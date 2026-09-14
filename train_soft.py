"""
Phase 5B: controlled soft-label experiment (expC).

Hypothesis
----------
Does replacing hard one-hot supervision with FER+ annotator soft-target
supervision improve validation macro-F1, with the rest of the baseline
protocol held fixed?

The ONLY intended variable is the target distribution q.

Loss
----
PyTorch's nn.CrossEntropyLoss(weight=w, label_smoothing=eps) was verified
numerically to compute

    loss = sum_i sum_c [ -w_c * q_ic * log p_ic ]  /  sum_i w_{y_i}

with q_i = (1 - eps) * onehot(y_i) + eps / K.  Note the class weight w_c
multiplies each TARGET-CLASS term inside the inner sum (not the sample as a
whole), and the denominator is the sum of TARGET-class weights.

This script uses the identical expression, generalised so that q comes from
the FER+ annotator votes:

    q_i = (1 - eps) * soft_i + eps / K
    y_i = soft_i.argmax()          (equals the stored hard labels exactly)

Substituting soft_i = onehot(y_i) reproduces the baseline loss bit-for-bit;
this is asserted at startup.

Label smoothing is retained at eps=0.05 deliberately.  32.20% of FER+ soft
targets are already one-hot; disabling smoothing would make those targets
SHARPER than the baseline's, changing regularisation on a third of the
training set for reasons unrelated to FER+ information.  Keeping eps means
those rows are identical to the baseline and the experiment isolates the
67.80% of rows where annotators actually disagreed.

Reads:  processed/fer_soft.npz
Writes: runs/expC_soft_labels/{best_model.pt, config.json, history.csv,
                               metrics_val.json, confusion_matrix_val.csv,
                               train.log}

Never reads the test split.  Never writes to runs/baseline_resnet18/.
Architecture, transforms and metrics are IMPORTED from train_baseline.py so
they cannot drift; train_baseline.py itself is not modified or executed.
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
from torch.utils.data import DataLoader, Dataset

# Single source of truth for architecture / transforms / metrics.
from train_baseline import (BATCH_SIZE, IMG_SIZE, LABEL_SMOOTHING, LR_HEAD,
                            LR_MAIN, LR_MIN, MAX_EPOCHS, PATIENCE, SEED,
                            WARMUP_EPOCHS, WEIGHT_DECAY, WEIGHT_EXPONENT, Tee,
                            build_model, build_transforms, compute_metrics,
                            confusion_matrix, set_seed, write_confusion_csv)

ROOT = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(ROOT, "processed", "fer_soft.npz")
OUT = os.path.join(ROOT, "runs", "expC_soft_labels")
BASELINE_DIR = os.path.join(ROOT, "runs", "baseline_resnet18")

EXPERIMENT = "expC_soft_labels"
EXPECTED_CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger",
                    "disgust", "fear"]
EXPECTED_SPLITS = {"train": 20481, "val": 4390, "test": 4388}


# --------------------------------------------------------------- dataset
class SoftFERDataset(Dataset):
    """Returns (image, soft_target, hard_label). Only the image is augmented."""

    def __init__(self, images, soft, labels, transform):
        self.images = images
        self.soft = torch.from_numpy(soft.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).unsqueeze(0)   # (1,48,48) uint8
        return self.transform(x), self.soft[i], self.labels[i]


# ------------------------------------------------------------------ loss
def soft_weighted_ce(logits, q, y, class_weights, eps=LABEL_SMOOTHING):
    """Weighted soft-target cross-entropy.

        q_smoothed = (1 - eps) * q + eps / K
        loss = sum_i sum_c [ -w_c * q_ic * log p_ic ] / sum_i w_{y_i}

    Exactly reproduces nn.CrossEntropyLoss(weight, label_smoothing=eps)
    when q is one-hot.
    """
    k = logits.size(1)
    qs = (1.0 - eps) * q + eps / k
    logp = F.log_softmax(logits.float(), dim=1)
    numerator = -(qs * logp * class_weights.view(1, -1)).sum()
    denominator = class_weights[y].sum()
    return numerator / denominator


def assert_loss_generalises_baseline(class_weights, device, k):
    """Prove soft_weighted_ce == baseline loss when the target is one-hot."""
    g = torch.Generator(device="cpu").manual_seed(0)
    for b in (3, 16, 64):
        logits = torch.randn(b, k, generator=g).to(device)
        y = torch.randint(0, k, (b,), generator=g).to(device)
        onehot = F.one_hot(y, k).float()
        mine = soft_weighted_ce(logits, onehot, y, class_weights)
        ref = nn.CrossEntropyLoss(weight=class_weights,
                                  label_smoothing=LABEL_SMOOTHING)(logits, y)
        assert torch.allclose(mine, ref, atol=1e-5), \
            f"loss mismatch at B={b}: {mine.item()} vs {ref.item()}"
    return True


# ------------------------------------------------------------ run helpers
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
        loss = soft_weighted_ce(out, q, y, class_weights)
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
        loss = soft_weighted_ce(out, q, y, class_weights)
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

    # ---- safety checks 11 & 12 BEFORE creating anything ------------------
    assert os.path.abspath(OUT) != os.path.abspath(BASELINE_DIR), \
        "refusing to write into the baseline run directory"
    assert os.path.basename(OUT) == "expC_soft_labels", \
        f"unexpected output directory: {OUT}"

    os.makedirs(OUT, exist_ok=True)
    log = Tee(os.path.join(OUT, "train.log"))
    log("=" * 72)
    log("PHASE 5B - CONTROLLED SOFT-LABEL EXPERIMENT (expC)")
    log("=" * 72)

    # ---- checks 1-7: dataset -------------------------------------------
    assert os.path.exists(NPZ), f"missing dataset: {NPZ}"
    d = np.load(NPZ)
    for key in ("images", "labels", "split", "hash", "class_names",
                "votes", "vote_cols", "soft"):
        assert key in d.files, f"missing array: {key}"
    images, labels, split = d["images"], d["labels"], d["split"]
    soft = d["soft"].astype(np.float32)
    class_names = [str(c) for c in d["class_names"]]
    k = len(class_names)

    assert images.shape == (29259, 48, 48), f"bad images shape {images.shape}"
    assert soft.shape == (29259, 7), f"bad soft shape {soft.shape}"
    assert labels.shape == (29259,), f"bad labels shape {labels.shape}"
    got = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    assert got == EXPECTED_SPLITS, f"split mismatch: {got}"
    assert class_names == EXPECTED_CLASSES, f"class order mismatch: {class_names}"
    assert np.allclose(soft.sum(1), 1.0, atol=1e-5), "soft rows do not sum to 1"
    assert (soft.argmax(1) == labels.astype(np.int64)).all(), \
        "soft.argmax != labels"
    log(f"\ndataset: {os.path.relpath(NPZ, ROOT)}")
    log(f"  samples {len(labels)} | " + " ".join(f"{s}={v}" for s, v in got.items()))
    log(f"  classes {class_names}")
    log(f"  soft sum max deviation {float(np.abs(soft.sum(1) - 1).max()):.2e}")
    log(f"  soft.argmax == labels: {int((soft.argmax(1) == labels).sum())}/{len(labels)}")
    onehot_frac = float((soft.max(1) >= 1 - 1e-6).mean())
    log(f"  one-hot rows {onehot_frac * 100:.2f}%  genuinely soft "
        f"{(1 - onehot_frac) * 100:.2f}%")

    # ---- checks 8-10: device and seed -----------------------------------
    assert torch.cuda.is_available(), "CUDA not available - refusing to run"
    device = torch.device("cuda")
    log(f"\nGPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    log(f"seed {SEED}")

    # ---- data ------------------------------------------------------------
    m_tr, m_va = (split == "train"), (split == "val")
    x_tr, s_tr, y_tr = images[m_tr], soft[m_tr], labels[m_tr]
    x_va, s_va, y_va = images[m_va], soft[m_va], labels[m_va]
    log(f"train {m_tr.sum()} | val {m_va.sum()} | test {int((split == 'test').sum())} (NEVER read)")

    mean = float(x_tr.mean() / 255.0)
    std = float(x_tr.std() / 255.0)
    log(f"train-split grayscale mean={mean:.4f} std={std:.4f}")

    counts = np.bincount(y_tr.astype(np.int64), minlength=k).astype(np.float64)
    w = (counts.sum() / (k * np.maximum(counts, 1))) ** WEIGHT_EXPONENT
    w = w / w.mean()
    log(f"class-weight exponent alpha={WEIGHT_EXPONENT} (identical to baseline)")
    log("class weights: " + ", ".join(
        f"{c}={w[i]:.3f}(n={int(counts[i])})" for i, c in enumerate(class_names)))
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    t_train, t_eval = build_transforms(mean, std)
    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(SoftFERDataset(x_tr, s_tr, y_tr, t_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(SoftFERDataset(x_va, s_va, y_va, t_eval),
                            batch_size=256, shuffle=False, **common)

    # ---- model and loss self-test ---------------------------------------
    model = build_model(k).to(device, memory_format=torch.channels_last)
    assert_loss_generalises_baseline(class_weights, device, k)
    log("\nloss self-test PASSED: soft_weighted_ce(one-hot) == "
        "nn.CrossEntropyLoss(weight, label_smoothing=0.05)")

    cfg = {
        "experiment": EXPERIMENT,
        "hypothesis": "FER+ soft-target supervision improves validation "
                      "macro-F1 vs hard-label supervision, all else fixed",
        "dataset": "processed/fer_soft.npz",
        "supervision": "FER+ annotator soft targets (7-class, renormalised)",
        "baseline_compared_against": {
            "run": "runs/baseline_resnet18", "best_epoch": 53,
            "val_macro_f1": 0.792805,
        },
        "class_names": class_names,
        "seed": SEED, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "warmup_epochs": WARMUP_EPOCHS, "max_epochs": args.epochs,
        "lr_head": LR_HEAD, "lr_main": LR_MAIN, "lr_min": LR_MIN,
        "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
        "precision": "bf16-autocast",
        "backbone": "resnet18/IMAGENET1K_V1 (conv1 collapsed to 1ch)",
        "augmentation": "identical to baseline (imported build_transforms)",
        "norm_mean": mean, "norm_std": std,
        "loss": {
            "name": "weighted soft-target cross-entropy",
            "formula": "sum_i sum_c [ -w_c * q_ic * log p_ic ] / sum_i w_{y_i}",
            "q": "(1 - eps) * soft_i + eps / K",
            "y": "soft_i.argmax() (== stored hard labels, 100% agreement)",
            "verified_equivalent_to_baseline_when_q_is_onehot": True,
            "pytorch_semantics_note": "class weight multiplies each target-class "
                                      "term inside the sum; denominator is the "
                                      "sum of target-class weights",
        },
        "label_smoothing": {
            "epsilon": LABEL_SMOOTHING,
            "disabled": False,
            "rationale": "Retained deliberately. 32.20% of FER+ soft targets are "
                         "already one-hot; disabling smoothing would make those "
                         "targets sharper than the baseline's, altering "
                         "regularisation on a third of the training set for "
                         "reasons unrelated to FER+ information. With eps kept, "
                         "unanimous rows are identical to the baseline and the "
                         "experiment isolates the 67.80% of rows where "
                         "annotators disagreed.",
        },
        "class_weights": {
            "exponent_alpha": WEIGHT_EXPONENT,
            "definition": "w_c = (N/(K*n_c))**alpha, normalised to mean 1, "
                          "from TRAIN-split hard-label counts (identical to baseline)",
            "values": [round(float(v), 6) for v in w],
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

    # ---- warmup: head only ----------------------------------------------
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

    summary = {"training_duration_s": total_s, "final_lr": last_lr,
               "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
               "epochs_run": epoch}
    cfg["run_summary"] = summary
    with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    log("\ntest split was never read")
    log(f"artifacts in {OUT}")


if __name__ == "__main__":
    main()
