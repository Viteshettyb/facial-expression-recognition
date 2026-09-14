"""
Phase 3 baseline: ResNet-18 (ImageNet-pretrained) on the validated FER+ artifact.

Reads (read-only):
    processed/fer_clean.npz

Writes:
    runs/baseline_resnet18/{config.json, best_model.pt, history.csv,
                            metrics_val.json, confusion_matrix_val.csv, train.log}

Model selection is on VALIDATION macro-F1 only. The test split is never read
unless --eval-test is passed explicitly.
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2

ROOT = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(ROOT, "processed", "fer_clean.npz")
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(RUNS, "baseline_resnet18")

SEED = 42
IMG_SIZE = 112
BATCH_SIZE = 128
WARMUP_EPOCHS = 3
MAX_EPOCHS = 60
LR_HEAD = 1e-3
LR_MAIN = 3e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
PATIENCE = 12
WEIGHT_EXPONENT = 0.5   # alpha in w_c ~ (1/n_c)**alpha; 0.5 = sqrt-inverse-freq


# ----------------------------------------------------------------- utilities
class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------- dataset
class FERDataset(Dataset):
    """Holds the uint8 images in RAM (~67 MB); no disk IO during training."""

    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).unsqueeze(0)   # (1,48,48) uint8
        return self.transform(x), self.labels[i]


def build_transforms(mean, std):
    train = v2.Compose([
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1),
                             antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=12),
        v2.ColorJitter(brightness=0.25, contrast=0.25),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean], std=[std]),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.12), value=0.0),
    ])
    evaluate = v2.Compose([
        v2.Resize(IMG_SIZE, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean], std=[std]),
    ])
    return train, evaluate


# --------------------------------------------------------------------- model
def build_model(num_classes):
    """ImageNet ResNet-18 with conv1 collapsed to a single grayscale channel.

    Summing conv1's weights across the RGB input dimension reproduces the
    response of feeding a 3x-replicated grayscale image, at 1/3 the stem cost.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    old = model.conv1
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
    model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ------------------------------------------------------------------- metrics
def confusion_matrix(y_true, y_pred, k):
    idx = y_true.astype(np.int64) * k + y_pred.astype(np.int64)
    return np.bincount(idx, minlength=k * k).reshape(k, k)


def compute_metrics(cm, class_names):
    k = cm.shape[0]
    support = cm.sum(1)
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(0).astype(np.float64)

    precision = np.divide(tp, pred_pos, out=np.zeros(k), where=pred_pos > 0)
    recall = np.divide(tp, support.astype(np.float64),
                       out=np.zeros(k), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(k), where=denom > 0)

    present = support > 0
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "per_class": {
            c: {"precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(class_names)
        },
    }


def write_confusion_csv(path, cm, class_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true|pred"] + list(class_names))
        for i, c in enumerate(class_names):
            w.writerow([c] + [int(v) for v in cm[i]])


# ------------------------------------------------------------------ run loop
@torch.no_grad()
def evaluate(model, loader, criterion, device, k):
    model.eval()
    loss_sum, n = 0.0, 0
    trues, preds = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * y.size(0)
        n += y.size(0)
        trues.append(y.cpu().numpy())
        preds.append(out.argmax(1).cpu().numpy())
    cm = confusion_matrix(np.concatenate(trues), np.concatenate(preds), k)
    return loss_sum / n, cm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = criterion(out, y)
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
                    help="alpha in w_c ~ (1/n_c)**alpha, normalised to mean 1. "
                         "0.5 reproduces the baseline exactly.")
    ap.add_argument("--run-name", type=str, default="baseline_resnet18",
                    help="output subdirectory under runs/")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training entirely: load runs/<run-name>/best_model.pt "
                         "and evaluate only. Requires --eval-test. Never writes "
                         "config.json, history.csv or best_model.pt.")
    ap.add_argument("--eval-test", action="store_true",
                    help="Run the ONE-SHOT test evaluation with the best "
                         "checkpoint. Off by default: never use for selection.")
    args = ap.parse_args()

    global OUT
    OUT = os.path.join(RUNS, args.run_name)
    os.makedirs(OUT, exist_ok=True)
    log = Tee(os.path.join(OUT, "train.log"))
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA device not available - refusing to run on CPU.")
    log(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    # ---- data ------------------------------------------------------------
    d = np.load(NPZ)
    images, labels, split = d["images"], d["labels"], d["split"]
    class_names = [str(c) for c in d["class_names"]]
    k = len(class_names)

    m_tr, m_va, m_te = (split == "train"), (split == "val"), (split == "test")
    x_tr, y_tr = images[m_tr], labels[m_tr]
    x_va, y_va = images[m_va], labels[m_va]
    log(f"train {m_tr.sum()} | val {m_va.sum()} | test {m_te.sum()} (untouched)")

    # normalisation from the TRAIN split only
    mean = float(x_tr.mean() / 255.0)
    std = float(x_tr.std() / 255.0)
    log(f"train-split grayscale mean={mean:.4f} std={std:.4f}")

    # inverse-frequency class weights ^alpha, normalised to mean 1
    alpha = args.weight_exponent
    counts = np.bincount(y_tr.astype(np.int64), minlength=k).astype(np.float64)
    w = (counts.sum() / (k * np.maximum(counts, 1))) ** alpha
    w = w / w.mean()
    log(f"class-weight exponent alpha={alpha}")
    log("class weights: " + ", ".join(
        f"{c}={w[i]:.3f}(n={int(counts[i])})" for i, c in enumerate(class_names)))
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    t_train, t_eval = build_transforms(mean, std)
    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(FERDataset(x_tr, y_tr, t_train), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(FERDataset(x_va, y_va, t_eval), batch_size=256,
                            shuffle=False, **common)

    # ---- model / loss ----------------------------------------------------
    model = build_model(k).to(device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=LABEL_SMOOTHING)

    # ---- eval-only: existing checkpoint, no training, no overwrites -------
    if args.eval_only:
        if not args.eval_test:
            raise SystemExit("--eval-only requires --eval-test.")
        ckpt_path = os.path.join(OUT, "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        c0 = ckpt["config"]
        # guard: the pipeline reconstructed here must match the trained one
        assert c0["class_names"] == class_names, "class name mismatch"
        assert abs(c0["norm_mean"] - mean) < 1e-9, "norm mean mismatch"
        assert abs(c0["norm_std"] - std) < 1e-9, "norm std mismatch"
        assert np.allclose(np.array(c0["class_weights"]),
                           np.round(w, 6), atol=1e-6), "class weight mismatch"
        assert c0["img_size"] == IMG_SIZE, "img size mismatch"
        model.load_state_dict(ckpt["state_dict"])
        log(f"eval-only: loaded {ckpt_path} (epoch {ckpt['epoch']}, "
            f"val macro-F1 {ckpt['val_metrics']['macro_f1']:.4f})")
        log("pipeline guards passed; no training performed")
        test_loader = DataLoader(FERDataset(images[m_te], labels[m_te], t_eval),
                                 batch_size=256, shuffle=False, **common)
        te_loss, cm_te = evaluate(model, test_loader, criterion, device, k)
        mt = compute_metrics(cm_te, class_names)
        with open(os.path.join(OUT, "metrics_test.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"source_epoch": ckpt["epoch"], "eval_only": True,
                       "checkpoint": "best_model.pt", "test_loss": te_loss,
                       **mt}, f, indent=2)
        write_confusion_csv(os.path.join(OUT, "confusion_matrix_test.csv"),
                            cm_te, class_names)
        log(f"test accuracy {mt['accuracy']:.4f} | balanced "
            f"{mt['balanced_accuracy']:.4f} | macro-F1 {mt['macro_f1']:.4f}")
        log(f"artifacts in {OUT}")
        return

    cfg = {
        "seed": SEED, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "warmup_epochs": WARMUP_EPOCHS, "max_epochs": args.epochs,
        "lr_head": LR_HEAD, "lr_main": LR_MAIN, "lr_min": LR_MIN,
        "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTHING,
        "patience": PATIENCE, "precision": "bf16-autocast",
        "backbone": "resnet18/IMAGENET1K_V1 (conv1 collapsed to 1ch)",
        "class_names": class_names, "train_counts": counts.astype(int).tolist(),
        "class_weights": [round(float(v), 6) for v in w],
        "weight_exponent": alpha, "run_name": args.run_name,
        "norm_mean": mean, "norm_std": std,
        "selection_metric": "val_macro_f1", "test_used_for_selection": False,
    }
    with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    hist_path = os.path.join(OUT, "history.csv")
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

    best_f1, best_epoch, bad_epochs = -1.0, -1, 0
    best_state = None

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
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device)
        va_loss, cm = evaluate(model, val_loader, criterion, device, k)
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

        # ---- selection on val macro-F1 only ------------------------------
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
            log(f"    * new best val macro-F1 {best_f1:.4f} (checkpoint saved)")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE and phase == "main":
                log(f"early stopping: no val macro-F1 gain in {PATIENCE} epochs")
                break

    log("")
    log(f"best epoch {best_epoch} | val macro-F1 {best_f1:.4f}")
    model.load_state_dict(best_state)

    with open(os.path.join(OUT, "metrics_val.json"), encoding="utf-8") as f:
        best = json.load(f)
    log(f"{'class':<12}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for c, s in best["per_class"].items():
        log(f"{c:<12}{s['precision']:>8.4f}{s['recall']:>8.4f}"
            f"{s['f1']:>8.4f}{s['support']:>9d}")
    log(f"val accuracy {best['accuracy']:.4f} | "
        f"balanced {best['balanced_accuracy']:.4f} | "
        f"macro-F1 {best['macro_f1']:.4f}")

    # ---- one-shot test evaluation, opt-in only ---------------------------
    if args.eval_test:
        log("")
        log("*** ONE-SHOT TEST EVALUATION (best checkpoint) ***")
        test_loader = DataLoader(FERDataset(images[m_te], labels[m_te], t_eval),
                                 batch_size=256, shuffle=False, **common)
        te_loss, cm_te = evaluate(model, test_loader, criterion, device, k)
        mt = compute_metrics(cm_te, class_names)
        with open(os.path.join(OUT, "metrics_test.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"source_epoch": best_epoch, "test_loss": te_loss, **mt},
                      f, indent=2)
        write_confusion_csv(os.path.join(OUT, "confusion_matrix_test.csv"),
                            cm_te, class_names)
        log(f"test accuracy {mt['accuracy']:.4f} | balanced "
            f"{mt['balanced_accuracy']:.4f} | macro-F1 {mt['macro_f1']:.4f}")
    else:
        log("")
        log("test split not evaluated (pass --eval-test when you are ready)")

    log(f"artifacts in {OUT}")


if __name__ == "__main__":
    main()
