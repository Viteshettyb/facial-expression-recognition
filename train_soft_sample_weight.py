"""
Phase 5C: controlled diagnosis experiment (expD).

Question
--------
Was expC's macro-F1 loss caused primarily by the interaction between FER+ soft
targets and the baseline's PER-CLASS weighting semantics?

expC applied w_c to every target-class component of the soft target, so rare-class
annotator mass scattered across thousands of rows was amplified by that class's
weight.  expD applies exactly ONE weight per sample, chosen by the sample's hard
target.  Everything else is identical to expC.

    q_i  = (1 - eps) * soft_i + eps / K
    y_i  = soft_i.argmax()            (== stored hard labels, 100% agreement)
    loss = sum_i [ w_{y_i} * sum_c (-q_ic * log p_ic) ] / sum_i w_{y_i}

Equivalence, measured not assumed
--------------------------------
With one-hot targets and eps=0 this is EXACTLY nn.CrossEntropyLoss(weight=w)
(relative difference 2.2e-16; asserted at startup).

With eps=0.05 it differs from nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
by ~0.72% relative, because the baseline weights the eps/K smoothing mass by each
class's own w_c while per-sample weighting scales it by w_y.  That gap is an
unavoidable consequence of the semantic change under test, and is reported rather
than papered over.

Therefore:
    expD vs expC   -> perfectly controlled, one variable (weighting semantics)
    expD vs baseline -> two variables (targets + weighting) plus the smoothing gap

Reads:  processed/fer_soft.npz
Writes: runs/expD_soft_labels_sample_weight/

Never reads the test split.  Never writes to runs/baseline_resnet18/ or
runs/expC_soft_labels/.  Architecture, transforms and metrics are IMPORTED from
train_baseline.py so they cannot drift.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_baseline import (BATCH_SIZE, IMG_SIZE, LABEL_SMOOTHING, LR_HEAD,
                            LR_MAIN, LR_MIN, MAX_EPOCHS, PATIENCE, SEED,
                            WARMUP_EPOCHS, WEIGHT_DECAY, WEIGHT_EXPONENT, Tee,
                            build_model, build_transforms, compute_metrics,
                            confusion_matrix, set_seed, write_confusion_csv)
# Dataset class reused verbatim from expC: identical batching and augmentation.
from train_soft import SoftFERDataset

ROOT = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(ROOT, "processed", "fer_soft.npz")
CLEAN = os.path.join(ROOT, "processed", "fer_clean.npz")
OUT = os.path.join(ROOT, "runs", "expD_soft_labels_sample_weight")
BASELINE_DIR = os.path.join(ROOT, "runs", "baseline_resnet18")
EXPC_DIR = os.path.join(ROOT, "runs", "expC_soft_labels")

EXPERIMENT = "expD_soft_labels_sample_weight"
EXPECTED_CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger",
                    "disgust", "fear"]
EXPECTED_SPLITS = {"train": 20481, "val": 4390, "test": 4388}


# ------------------------------------------------------------------- loss
def sample_weighted_soft_ce(logits, q, y, class_weights, eps=LABEL_SMOOTHING):
    """One class weight per SAMPLE, selected by the sample's hard target.

        q_s  = (1 - eps) * q + eps / K
        loss = sum_i [ w_{y_i} * sum_c (-q_s_ic * log p_ic) ] / sum_i w_{y_i}
    """
    k = logits.size(1)
    qs = (1.0 - eps) * q + eps / k
    logp = F.log_softmax(logits.float(), dim=1)
    per_sample = -(qs * logp).sum(dim=1)          # (B,)
    sw = class_weights[y]                          # (B,)
    return (sw * per_sample).sum() / sw.sum()


def loss_equivalence_report(class_weights, device, k):
    """Control 1: measure the reduction to the baseline loss on one-hot targets."""
    g = torch.Generator(device="cpu").manual_seed(0)
    out = {}
    for eps in (0.0, LABEL_SMOOTHING):
        devs = []
        for b in (3, 16, 64):
            logits = torch.randn(b, k, generator=g).double().to(device)
            y = torch.randint(0, k, (b,), generator=g).to(device)
            onehot = F.one_hot(y, k).double()
            mine = sample_weighted_soft_ce(logits, onehot, y,
                                           class_weights.double(), eps=eps)
            ref = nn.CrossEntropyLoss(weight=class_weights.double(),
                                      label_smoothing=eps)(logits, y)
            devs.append(abs(mine - ref).item() / ref.item())
        out[eps] = max(devs)
    # eps=0 must be an identity. The loss upcasts to float32 internally (needed
    # for bf16 training stability), so float32 epsilon is the achievable floor;
    # in pure float64 the same comparison lands at ~2e-16.
    assert out[0.0] < 1e-6, f"eps=0 identity broken: rel diff {out[0.0]:.3e}"
    return out


# -------------------------------------------------------------- run loops
@torch.no_grad()
def evaluate(model, loader, class_weights, device, k):
    model.eval()
    loss_sum, n = 0.0, 0
    trues, preds = [], []
    for x, q, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        q = q.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        loss = sample_weighted_soft_ce(out, q, y, class_weights)
        loss_sum += loss.item() * y.size(0)
        n += y.size(0)
        trues.append(y.cpu().numpy())
        preds.append(out.argmax(1).cpu().numpy())
    cm = confusion_matrix(np.concatenate(trues), np.concatenate(preds), k)
    return loss_sum / n, cm


def train_one_epoch(model, loader, class_weights, optimizer, device):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, q, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        q = q.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        loss = sample_weighted_soft_ce(out, q, y, class_weights)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()

    for protected in (BASELINE_DIR, EXPC_DIR):
        assert os.path.abspath(OUT) != os.path.abspath(protected), \
            f"refusing to write into {protected}"
    assert os.path.basename(OUT) == EXPERIMENT, f"unexpected output dir: {OUT}"

    os.makedirs(OUT, exist_ok=True)
    log = Tee(os.path.join(OUT, "train.log"))
    log("=" * 76)
    log("PHASE 5C - DIAGNOSIS EXPERIMENT expD (per-SAMPLE class weighting)")
    log("=" * 76)

    # ---- control 3/4: dataset comes straight from fer_soft.npz ----------
    assert os.path.exists(NPZ), f"missing dataset: {NPZ}"
    d = np.load(NPZ)
    for key in ("images", "labels", "split", "hash", "class_names",
                "votes", "vote_cols", "soft"):
        assert key in d.files, f"missing array: {key}"
    images, labels, split = d["images"], d["labels"], d["split"]
    hsh, soft = d["hash"], d["soft"].astype(np.float32)
    class_names = [str(c) for c in d["class_names"]]
    k = len(class_names)

    clean = np.load(CLEAN)
    for key in ("images", "labels", "split", "hash", "class_names"):
        assert np.array_equal(d[key], clean[key]), \
            f"{key} differs from fer_clean.npz - dataset was regenerated"
    assert images.shape == (29259, 48, 48) and soft.shape == (29259, 7)
    got = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    assert got == EXPECTED_SPLITS, f"split mismatch: {got}"
    assert class_names == EXPECTED_CLASSES, f"class order: {class_names}"
    assert np.allclose(soft.sum(1), 1.0, atol=1e-5)
    assert (soft.argmax(1) == labels.astype(np.int64)).all()

    log("\n--- EXPERIMENT MANIFEST ---")
    log(f"experiment        : {EXPERIMENT}")
    log(f"dataset           : processed/fer_soft.npz  "
        f"(images/labels/split/hash/class_names array-equal to fer_clean.npz)")
    log(f"samples           : {len(labels)}  " +
        " ".join(f"{s}={v}" for s, v in got.items()))
    log(f"class order       : {class_names}")
    log(f"supervision       : FER+ soft targets (identical to expC)")
    log(f"variable under test: class weighting  PER-CLASS (expC) -> PER-SAMPLE (expD)")
    log(f"one-hot rows      : {float((soft.max(1) >= 1 - 1e-6).mean()) * 100:.2f}%")

    assert torch.cuda.is_available(), "CUDA not available - refusing to run"
    device = torch.device("cuda")
    log(f"GPU               : {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    log(f"seed              : {SEED}")

    m_tr, m_va = (split == "train"), (split == "val")
    x_tr, s_tr, y_tr = images[m_tr], soft[m_tr], labels[m_tr]
    x_va, s_va, y_va = images[m_va], soft[m_va], labels[m_va]
    log(f"test split        : {int((split == 'test').sum())} rows - NEVER read")

    mean = float(x_tr.mean() / 255.0)
    std = float(x_tr.std() / 255.0)

    counts = np.bincount(y_tr.astype(np.int64), minlength=k).astype(np.float64)
    w = (counts.sum() / (k * np.maximum(counts, 1))) ** WEIGHT_EXPONENT
    w = w / w.mean()
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    # ---- control 2: class weights numerically identical to baseline -----
    with open(os.path.join(BASELINE_DIR, "config.json"), encoding="utf-8") as f:
        base_cfg = json.load(f)
    base_w = np.array(base_cfg["class_weights"], dtype=np.float64)
    max_dev = float(np.abs(np.round(w, 6) - base_w).max())
    assert max_dev < 1e-6, f"class weights differ from baseline by {max_dev}"
    assert base_cfg["norm_mean"] == mean and base_cfg["norm_std"] == std, \
        "normalisation statistics differ from baseline"
    log(f"norm mean/std     : {mean:.10f} / {std:.10f}  (identical to baseline)")
    log(f"class weights     : " + ", ".join(
        f"{c}={w[i]:.3f}" for i, c in enumerate(class_names)))
    log(f"                    max deviation vs baseline config = {max_dev:.2e}")

    # ---- control 1: loss equivalence ------------------------------------
    eqv = loss_equivalence_report(class_weights, device, k)
    log(f"\nloss equivalence on ONE-HOT targets vs nn.CrossEntropyLoss(weight, ls):")
    log(f"  eps=0.00 : max rel diff {eqv[0.0]:.3e}  -> EXACT identity (asserted)")
    log(f"  eps=0.05 : max rel diff {eqv[LABEL_SMOOTHING]:.3e}  -> differs only in how"
        f" the eps/K smoothing mass is weighted")

    t_train, t_eval = build_transforms(mean, std)
    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(SoftFERDataset(x_tr, s_tr, y_tr, t_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(SoftFERDataset(x_va, s_va, y_va, t_eval),
                            batch_size=256, shuffle=False, **common)

    model = build_model(k).to(device, memory_format=torch.channels_last)

    cfg = {
        "experiment": EXPERIMENT,
        "purpose": "Diagnosis: is expC's macro-F1 loss caused by the interaction "
                   "between FER+ soft targets and PER-CLASS weighting semantics?",
        "dataset": "processed/fer_soft.npz",
        "supervision": "FER+ annotator soft targets (identical to expC)",
        "variable_under_test": "class weighting: per-class (expC) -> per-sample (expD)",
        "compared_against": {
            "baseline": {"run": "runs/baseline_resnet18", "best_epoch": 53,
                         "val_macro_f1": 0.792805},
            "expC": {"run": "runs/expC_soft_labels", "best_epoch": 44,
                     "val_macro_f1": 0.750348},
        },
        "class_names": class_names,
        "seed": SEED, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "warmup_epochs": WARMUP_EPOCHS, "max_epochs": args.epochs,
        "lr_head": LR_HEAD, "lr_main": LR_MAIN, "lr_min": LR_MIN,
        "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
        "label_smoothing": LABEL_SMOOTHING,
        "precision": "bf16-autocast",
        "backbone": "resnet18/IMAGENET1K_V1 (conv1 collapsed to 1ch)",
        "augmentation": "identical to baseline (imported build_transforms)",
        "norm_mean": mean, "norm_std": std,
        "loss": {
            "name": "per-sample weighted soft-target cross-entropy",
            "formula": "sum_i [ w_{y_i} * sum_c (-q_ic * log p_ic) ] / sum_i w_{y_i}",
            "q": "(1 - eps) * soft_i + eps / K",
            "y": "soft_i.argmax() (== stored hard labels)",
            "equivalence_onehot_eps0_rel_diff": eqv[0.0],
            "equivalence_onehot_eps005_rel_diff": eqv[LABEL_SMOOTHING],
            "equivalence_note": "Exactly nn.CrossEntropyLoss(weight) on one-hot "
                                "targets when eps=0. At eps=0.05 it differs by "
                                "~0.7% because the baseline weights the eps/K "
                                "smoothing mass by each class w_c whereas "
                                "per-sample weighting scales it by w_y.",
        },
        "class_weights": {
            "exponent_alpha": WEIGHT_EXPONENT,
            "definition": "w_c = (N/(K*n_c))**alpha normalised to mean 1, from "
                          "TRAIN-split hard-label counts",
            "values": [round(float(v), 6) for v in w],
            "max_deviation_vs_baseline_config": max_dev,
            "train_counts": counts.astype(int).tolist(),
        },
        "metrics_computed_on": "hard ground-truth labels from fer_soft.npz",
        "selection_metric": "val_macro_f1",
        "test_used_for_selection": False,
        "test_split_read": False,
    }
    with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    hist_path = os.path.join(OUT, "history.csv")
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "phase", "lr", "train_loss", "train_acc",
                                "val_loss", "val_acc", "val_balanced_acc",
                                "val_macro_f1", "seconds"])

    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = None

    best_f1, best_epoch, bad_epochs, best_state = -1.0, -1, 0, None
    last_lr = LR_HEAD
    t_start = time.time()

    log("")
    for epoch in range(1, args.epochs + 1):
        if epoch == WARMUP_EPOCHS + 1:
            log("-- unfreezing backbone, switching to cosine schedule --")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAIN,
                                          weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - WARMUP_EPOCHS),
                eta_min=LR_MIN)

        phase = "warmup" if epoch <= WARMUP_EPOCHS else "main"
        lr_now = optimizer.param_groups[0]["lr"]
        last_lr = lr_now
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, class_weights,
                                          optimizer, device)
        va_loss, cm = evaluate(model, val_loader, class_weights, device, k)
        m = compute_metrics(cm, class_names)
        dt = time.time() - t0
        if scheduler is not None:
            scheduler.step()

        log(f"epoch {epoch:3d} [{phase}] lr {lr_now:.2e} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {m['accuracy']:.4f} "
            f"bal {m['balanced_accuracy']:.4f} macroF1 {m['macro_f1']:.4f} | "
            f"{dt:.1f}s")

        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, phase, f"{lr_now:.6e}", f"{tr_loss:.6f}", f"{tr_acc:.6f}",
                f"{va_loss:.6f}", f"{m['accuracy']:.6f}",
                f"{m['balanced_accuracy']:.6f}", f"{m['macro_f1']:.6f}",
                f"{dt:.1f}"])

        if m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = m["macro_f1"], epoch, 0
            best_state = {kk: vv.detach().cpu().clone()
                          for kk, vv in model.state_dict().items()}
            torch.save({"state_dict": best_state, "config": cfg,
                        "epoch": epoch, "val_metrics": m},
                       os.path.join(OUT, "best_model.pt"))
            with open(os.path.join(OUT, "metrics_val.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"epoch": epoch, "val_loss": va_loss, **m}, f, indent=2)
            write_confusion_csv(os.path.join(OUT, "confusion_matrix_val.csv"),
                                cm, class_names)
            log(f"    * new best val macro-F1 {best_f1:.6f} (checkpoint saved)")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE and phase == "main":
                log(f"early stopping: no val macro-F1 gain in {PATIENCE} epochs")
                break

    total_s = time.time() - t_start
    log("")
    log(f"best epoch {best_epoch} | val macro-F1 {best_f1:.6f}")
    log(f"training duration {total_s:.1f}s ({total_s / 60:.2f} min)")
    log(f"final learning rate {last_lr:.6e}")

    with open(os.path.join(OUT, "metrics_val.json"), encoding="utf-8") as f:
        best = json.load(f)
    log(f"{'class':<12}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for c, s in best["per_class"].items():
        log(f"{c:<12}{s['precision']:>8.4f}{s['recall']:>8.4f}"
            f"{s['f1']:>8.4f}{s['support']:>9d}")
    log(f"val accuracy {best['accuracy']:.6f} | "
        f"balanced {best['balanced_accuracy']:.6f} | "
        f"macro-F1 {best['macro_f1']:.6f}")

    cfg["run_summary"] = {"training_duration_s": total_s, "final_lr": last_lr,
                          "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                          "epochs_run": epoch}
    with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    log("\ntest split was never read")
    log(f"artifacts in {OUT}")


if __name__ == "__main__":
    main()
