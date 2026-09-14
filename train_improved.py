"""
Phase 6B: improved training pipeline on the recovered FER+ artifact.

Reads:  processed/fer_plus_v2.npz
Writes: runs/<run-name>/{config.json, best_model.pt, history.csv,
                         metrics_val.json, confusion_matrix_val.csv, train.log}

Three changes from the baseline, each one isolated so the sweep can attribute
the result to it:

1. TARGETS. The FER+ annotator distribution is the target, not its argmax.
   An image 5/10 sadness, 3/10 fear, 2/10 neutral trains as exactly that. The
   hard label is used only for class statistics, sampling and metrics.

       q_i  = (1 - eps) * soft_i + eps / K
       loss = sum_i w_i * CE(q_i, p_i) / sum_i w_i          w_i = w_{argmax soft_i}

   One weight per SAMPLE, keyed on its hard class - not one weight per target
   component. The difference is not cosmetic: weighting every component lets a
   rare class's weight amplify the scattered annotator mass it holds across
   thousands of unrelated rows, which is what collapsed disgust precision to
   0.25 in the earlier expC run. Verified at startup: with one-hot targets and
   eps=0 this loss equals nn.CrossEntropyLoss(weight=w) to floating-point noise.

2. DATA. processed/fer_plus_v2.npz carries the ambiguous rows that the v1
   filter discarded - which were overwhelmingly the hard classes (disgust
   doubles, fear +42%, anger +30%, against happiness +5%).

3. AUGMENTATION. The baseline's geometric and photometric set, plus mild blur
   and sensor noise. A webcam frame reaches the model through a BlazeFace crop
   and an INTER_AREA downscale to 48x48; it is softer and noisier than a FER+
   training image, and the model should have seen that.

Class imbalance is corrected in two mild halves rather than one aggressive one:
`--weight-exponent` on the loss and `--sampler-exponent` on a balanced sampler.
Both default to values that under-correct on purpose; over-correcting disgust
is what produced a 0.25-precision model last time.

Selection is on VALIDATION macro-F1. The test split is read only with
--eval-test, once, after selection is finished.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

from provenance_aug import ProvenanceRandomiser
from train_baseline import (BATCH_SIZE, IMG_SIZE, LABEL_SMOOTHING, LR_HEAD,
                            LR_MAIN, LR_MIN, MAX_EPOCHS, PATIENCE, SEED,
                            WARMUP_EPOCHS, WEIGHT_DECAY, Tee, build_model,
                            compute_metrics, confusion_matrix, set_seed,
                            write_confusion_csv)

ROOT = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(ROOT, "processed", "fer_plus_v2.npz")
RUNS = os.path.join(ROOT, "runs")

# Loss-side correction. 0.5 = sqrt-inverse-frequency, the baseline's value.
WEIGHT_EXPONENT = 0.5
# Sampler-side correction. 0 = natural class order, no oversampling.
SAMPLER_EXPONENT = 0.0
# Ceiling on how often a rare class may be oversampled. Without it, disgust
# (185 rows) would be drawn as often as neutral (9506) and the model would
# memorise those 185 images.
MAX_SAMPLER_WEIGHT = 8.0


# ------------------------------------------------------------------- dataset
class SoftFERDataset(Dataset):
    """uint8 images in RAM with their target distribution.

    `eps` is PER SAMPLE, not global. A FER+ row carries a real 10-annotator
    distribution; an ExpW row carries one person's single label. Smoothing the
    second harder is how that difference is expressed in the loss instead of
    being papered over - see --source-smoothing.

    `provenance` is applied to the 48x48 uint8 crop before any tensor transform,
    inside the exact path the DataLoader feeds the model.
    """

    def __init__(self, images, soft, hard, transform, eps=None, provenance=None):
        self.images = images
        self.soft = torch.from_numpy(soft.astype(np.float32))
        self.hard = torch.from_numpy(hard.astype(np.int64))
        self.transform = transform
        if eps is None:
            eps = np.full(len(hard), LABEL_SMOOTHING, dtype=np.float32)
        self.eps = torch.from_numpy(np.asarray(eps, dtype=np.float32))
        self.provenance = provenance

    def __len__(self):
        return len(self.hard)

    def __getitem__(self, i):
        img = self.images[i]
        if self.provenance is not None:
            img = self.provenance(img)
        x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)  # (1,48,48)
        return self.transform(x), self.soft[i], self.hard[i], self.eps[i]


class GaussianNoise(nn.Module):
    """Sensor noise, applied after scaling to float and before normalisation."""

    def __init__(self, std=0.02, p=0.3):
        super().__init__()
        self.std, self.p = std, p

    def forward(self, x):
        if torch.rand(()) >= self.p:
            return x
        return (x + torch.randn_like(x) * self.std).clamp_(0.0, 1.0)


def build_transforms_v2(mean, std):
    """Baseline set + mild blur and noise. Nothing here is unrealistic: no
    vertical flips (faces are not upside down), no large rotations, no hue
    shifts (the input is grayscale), no heavy occlusion."""
    train = v2.Compose([
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1),
                             antialias=True),
        v2.RandomHorizontalFlip(p=0.5),          # a mirrored face is a face
        v2.RandomRotation(degrees=12),           # head tilt, not head rotation
        v2.ColorJitter(brightness=0.30, contrast=0.30),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.25),
        v2.ToDtype(torch.float32, scale=True),
        GaussianNoise(std=0.02, p=0.30),
        v2.Normalize(mean=[mean], std=[std]),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.12), value=0.0),
    ])
    evaluate = v2.Compose([
        v2.Resize(IMG_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean], std=[std]),
    ])
    return train, evaluate


# ---------------------------------------------------------------------- loss
def soft_ce(logits, q, sample_w):
    """Weighted soft-target cross entropy, normalised by the weight mass."""
    per_sample = -(q * F.log_softmax(logits, dim=1)).sum(1)
    return (sample_w * per_sample).sum() / sample_w.sum().clamp_min(1e-12)


def _assert_loss_equivalence(k, device, log):
    """With one-hot targets and eps=0 this must BE weighted cross entropy."""
    g = torch.Generator(device="cpu").manual_seed(0)
    logits = torch.randn(64, k, generator=g).to(device)
    y = torch.randint(0, k, (64,), generator=g).to(device)
    w = (torch.rand(k, generator=g) + 0.5).to(device)
    q = F.one_hot(y, k).float()
    mine = soft_ce(logits, q, w[y])
    ref = nn.CrossEntropyLoss(weight=w)(logits, y)
    rel = abs(mine.item() - ref.item()) / abs(ref.item())
    assert rel < 1e-5, f"soft loss is not weighted CE (relative gap {rel:.2e})"
    log(f"loss equivalence check passed (relative gap {rel:.2e})")


# ------------------------------------------------------------------ run loop
@torch.no_grad()
def evaluate_split(model, loader, device, k, eps=None):
    model.eval()
    loss_sum, n = 0.0, 0
    trues, preds = [], []
    for x, q, y, e in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        q = q.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        e = e.to(device, non_blocking=True).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        qs = (1.0 - e) * q + e / k
        loss = soft_ce(out.float(), qs, torch.ones_like(y, dtype=torch.float32))
        loss_sum += loss.item() * y.size(0)
        n += y.size(0)
        trues.append(y.cpu().numpy())
        preds.append(out.argmax(1).cpu().numpy())
    cm = confusion_matrix(np.concatenate(trues), np.concatenate(preds), k)
    return loss_sum / n, cm


def train_one_epoch(model, loader, optimizer, device, k, eps, class_w):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, q, y, e in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        q = q.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        e = e.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        qs = (1.0 - e) * q + e / k
        loss = soft_ce(out.float(), qs, class_w[y])
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--weight-exponent", type=float, default=WEIGHT_EXPONENT,
                    help="alpha in loss weight w_c ~ (1/n_c)**alpha")
    ap.add_argument("--sampler-exponent", type=float, default=SAMPLER_EXPONENT,
                    help="beta in sampling weight s_c ~ (1/n_c)**beta; "
                         "0 disables the balanced sampler")
    ap.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    ap.add_argument("--patience", type=int, default=PATIENCE,
                    help="early-stopping patience on val macro-F1. The default "
                         "stops runs before the cosine schedule has annealed "
                         "when macro-F1 is noisy (val disgust support is 18); "
                         "raise it to let the full schedule run.")
    ap.add_argument("--dataset", type=str, default=NPZ,
                    help="training artifact. Defaults to the FER+-only one so "
                         "earlier runs stay reproducible.")
    ap.add_argument("--source-smoothing", type=float, default=None,
                    help="label smoothing for ExpW rows, which carry a single "
                         "annotator's label rather than a vote distribution. "
                         "Defaults to --label-smoothing when unset.")
    ap.add_argument("--provenance-aug", action="store_true",
                    help="randomised resampling chain (downscale/JPEG/interp) "
                         "on every training crop, matching the live camera's "
                         "own channel")
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--eval-test", action="store_true",
                    help="ONE-SHOT test evaluation with the best checkpoint.")
    args = ap.parse_args()

    out = os.path.join(RUNS, args.run_name)
    os.makedirs(out, exist_ok=True)
    log = Tee(os.path.join(out, "train.log"))
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA device not available - refusing to run on CPU.")
    log(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    log(f"run {args.run_name} | alpha={args.weight_exponent} "
        f"beta={args.sampler_exponent} eps={args.label_smoothing}")

    # ---- data ------------------------------------------------------------
    d = np.load(args.dataset)
    images, hard, soft, split = d["images"], d["labels"], d["soft"], d["split"]
    class_names = [str(c) for c in d["class_names"]]
    k = len(class_names)
    has_source = "source" in d.files
    source = d["source"] if has_source else np.array(["ferplus"] * len(hard))

    m_tr, m_va, m_te = (split == "train"), (split == "val"), (split == "test")
    m_ho = split == "expw_holdout"
    x_tr, y_tr, s_tr = images[m_tr], hard[m_tr], soft[m_tr]
    log(f"dataset {os.path.relpath(args.dataset, ROOT)}")
    log(f"train {m_tr.sum()} | val {m_va.sum()} | test {m_te.sum()} (untouched) "
        f"| expw_holdout {m_ho.sum()} (never trained on)")
    if has_source:
        for name in ("ferplus", "expw"):
            log(f"  train rows from {name}: {int((source[m_tr] == name).sum())}")

    # The mistakes that would invalidate everything, checked rather than assumed.
    assert not (source[m_va | m_te] == "expw").any(), \
        "an ExpW row is inside the FER+ evaluation splits"
    assert int(m_ho.sum()) == 0 or not (m_ho & m_tr).any(), \
        "the ExpW holdout overlaps the training split"
    if "hash" in d.files:
        tr_h, ev_h = set(d["hash"][m_tr]), set(d["hash"][m_va | m_te])
        assert not (tr_h & ev_h), "train and eval share an image hash"
        tr_d, ev_d = set(d["dhash"][m_tr]), set(d["dhash"][m_va | m_te])
        assert not (tr_d & ev_d), "train and eval share a near-duplicate image"

    mean = float(x_tr.mean() / 255.0)
    std = float(x_tr.std() / 255.0)
    log(f"train-split grayscale mean={mean:.4f} std={std:.4f}")

    counts = np.bincount(y_tr.astype(np.int64), minlength=k).astype(np.float64)

    # loss-side class weights
    alpha = args.weight_exponent
    w = (counts.sum() / (k * np.maximum(counts, 1))) ** alpha
    w = w / w.mean()
    log("loss weights:  " + ", ".join(
        f"{c}={w[i]:.3f}(n={int(counts[i])})" for i, c in enumerate(class_names)))
    class_w = torch.tensor(w, dtype=torch.float32, device=device)

    # sampler-side weights, capped so a 185-row class is not drawn to death
    beta = args.sampler_exponent
    sampler = None
    sw = np.ones(k)
    if beta > 0:
        sw = (counts.sum() / (k * np.maximum(counts, 1))) ** beta
        sw = np.minimum(sw / sw.mean(), MAX_SAMPLER_WEIGHT)
        log("sample weights: " + ", ".join(
            f"{c}={sw[i]:.3f}" for i, c in enumerate(class_names)))
        per_row = sw[y_tr.astype(np.int64)]
        g = torch.Generator().manual_seed(SEED)
        sampler = WeightedRandomSampler(
            torch.as_tensor(per_row, dtype=torch.double),
            num_samples=len(per_row), replacement=True, generator=g)

    t_train, t_eval = build_transforms_v2(mean, std)

    # Per-sample label smoothing, keyed on where the label came from.
    eps_base = args.label_smoothing
    eps_expw = args.source_smoothing if args.source_smoothing is not None else eps_base
    eps_tr = np.where(source[m_tr] == "expw", eps_expw, eps_base).astype(np.float32)
    log(f"label smoothing: ferplus={eps_base} expw={eps_expw} "
        f"(mean over train {float(eps_tr.mean()):.4f})")

    provenance = ProvenanceRandomiser(seed=SEED) if args.provenance_aug else None
    if provenance is not None:
        log("provenance randomisation ON: " + json.dumps(provenance.describe()))

    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(
        SoftFERDataset(x_tr, s_tr, y_tr, t_train, eps_tr, provenance),
        batch_size=BATCH_SIZE, shuffle=(sampler is None), sampler=sampler,
        drop_last=True, **common)
    # Evaluation is never augmented and never smoothed per-source: the metric
    # must not move because of how a row was labelled.
    val_loader = DataLoader(
        SoftFERDataset(images[m_va], soft[m_va], hard[m_va], t_eval),
        batch_size=256, shuffle=False, **common)

    # ---- model -----------------------------------------------------------
    model = build_model(k).to(device, memory_format=torch.channels_last)
    _assert_loss_equivalence(k, device, log)
    eps = args.label_smoothing

    cfg = {
        "seed": SEED, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "warmup_epochs": WARMUP_EPOCHS, "max_epochs": args.epochs,
        "lr_head": LR_HEAD, "lr_main": LR_MAIN, "lr_min": LR_MIN,
        "weight_decay": WEIGHT_DECAY, "label_smoothing": eps,
        "patience": args.patience, "precision": "bf16-autocast",
        "backbone": "resnet18/IMAGENET1K_V1 (conv1 collapsed to 1ch)",
        "class_names": class_names, "train_counts": counts.astype(int).tolist(),
        "class_weights": [round(float(v), 6) for v in w],
        "weight_exponent": alpha, "sampler_exponent": beta,
        "sampler_weights": [round(float(v), 6) for v in sw],
        "max_sampler_weight": MAX_SAMPLER_WEIGHT,
        "run_name": args.run_name,
        "norm_mean": mean, "norm_std": std,
        "dataset": os.path.relpath(args.dataset, ROOT).replace("\\", "/"),
        "train_rows": int(m_tr.sum()),
        "train_rows_by_source": ({n: int((source[m_tr] == n).sum())
                                  for n in ("ferplus", "expw")} if has_source
                                 else {"ferplus": int(m_tr.sum())}),
        "label_smoothing_ferplus": eps_base,
        "label_smoothing_expw": eps_expw,
        "provenance_augmentation": (provenance.describe() if provenance
                                    else {"enabled": False}),
        "targets": "FER+ annotator vote distribution (soft)",
        "augmentation": "RRC(0.75-1.0) + hflip + rot12 + jitter(0.30) + "
                        "blur(p0.25) + noise(p0.30,std0.02) + erasing(p0.25)",
        "selection_metric": "val_macro_f1", "test_used_for_selection": False,
    }
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    hist_path = os.path.join(out, "history.csv")
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "phase", "lr", "train_loss", "train_acc",
                                "val_loss", "val_acc", "val_balanced_acc",
                                "val_macro_f1", "seconds"])

    # ---- phase 1: frozen backbone, head only -----------------------------
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = None

    best_f1, best_epoch, bad_epochs, best_state = -1.0, -1, 0, None

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
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          device, k, eps, class_w)
        va_loss, cm = evaluate_split(model, val_loader, device, k)
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
            # Checkpoint layout is the one ml/model.py::EmotionClassifier reads.
            torch.save({"state_dict": best_state, "config": cfg,
                        "epoch": epoch, "val_metrics": m},
                       os.path.join(out, "best_model.pt"))
            with open(os.path.join(out, "metrics_val.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"epoch": epoch, "val_loss": va_loss, **m}, f, indent=2)
            write_confusion_csv(os.path.join(out, "confusion_matrix_val.csv"),
                                cm, class_names)
            log(f"    * new best val macro-F1 {best_f1:.4f} (checkpoint saved)")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience and phase == "main":
                log(f"early stopping: no val macro-F1 gain in {args.patience} epochs")
                break

    log("")
    log(f"best epoch {best_epoch} | val macro-F1 {best_f1:.4f}")
    model.load_state_dict(best_state)

    with open(os.path.join(out, "metrics_val.json"), encoding="utf-8") as f:
        best = json.load(f)
    log(f"{'class':<12}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for c, s in best["per_class"].items():
        log(f"{c:<12}{s['precision']:>8.4f}{s['recall']:>8.4f}"
            f"{s['f1']:>8.4f}{s['support']:>9d}")
    log(f"val accuracy {best['accuracy']:.4f} | "
        f"balanced {best['balanced_accuracy']:.4f} | macro-F1 {best['macro_f1']:.4f}")

    if args.eval_test:
        log("")
        log("*** ONE-SHOT TEST EVALUATION (best checkpoint) ***")
        test_loader = DataLoader(
            SoftFERDataset(images[m_te], soft[m_te], hard[m_te], t_eval),
            batch_size=256, shuffle=False, **common)
        te_loss, cm_te = evaluate_split(model, test_loader, device, k)
        mt = compute_metrics(cm_te, class_names)
        with open(os.path.join(out, "metrics_test.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"source_epoch": best_epoch, "test_loss": te_loss, **mt},
                      f, indent=2)
        write_confusion_csv(os.path.join(out, "confusion_matrix_test.csv"),
                            cm_te, class_names)
        log(f"test accuracy {mt['accuracy']:.4f} | balanced "
            f"{mt['balanced_accuracy']:.4f} | macro-F1 {mt['macro_f1']:.4f}")

    log(f"artifacts in {out}")


if __name__ == "__main__":
    main()
