"""
Phase 7C: the honesty check. Can a model tell FER+ from ExpW?

Reads:  processed/fer_expw_v1.npz
Writes: processed/domain_probe.json

WHY THIS EXISTS
---------------
Merging two corpora is only safe if the corpus itself is not a usable feature.
Our disgust class ends up ~95% ExpW-sourced simply because 185 FER+ disgust
images is all that exists - so IF a network can tell the two corpora apart from
a 48x48 crop, it has a shortcut to "disgust" that has nothing to do with the
face, and the merged model's rare-class numbers would be measuring the shortcut.

The defence is structural: every ExpW image is reduced through the same
BlazeFace detection, the same Phase 4A.1-calibrated square crop and the same
BT.601 grayscale 48x48 bottleneck as FER+, which destroys colour, resolution,
background framing and camera signature on purpose. This script checks whether
that actually worked instead of assuming it did.

METHOD
------
Train the same ResNet-18 used for the real task, on the same augmentation, to
predict the SOURCE (ferplus vs expw) rather than the emotion. To stop it from
answering via class priors - ExpW is disgust-heavy, FER+ is neutral-heavy, so
"looks like disgust" alone would leak the source - the probe is trained on a
CLASS-BALANCED subset: for every emotion, an equal number of FER+ and ExpW rows.
Then emotion carries zero information about source, and any accuracy above 50%
is genuine corpus signal.

READING THE RESULT
------------------
  ~0.50-0.60   the bottleneck worked; the corpora are not separable
  ~0.60-0.75   some residual signal; interpret rare-class gains cautiously
  >0.75        the shortcut is real; merged disgust/fear numbers are suspect

This reports the number. It does not decide for you.
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from provenance_aug import (ExpressionPreservingBlur,                 # noqa: E402
                            ProvenanceRandomiser)
from train_baseline import (BATCH_SIZE, IMG_SIZE, SEED, build_model,  # noqa: E402
                            confusion_matrix, set_seed)
from train_improved import build_transforms_v2                        # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "processed")
NPZ = os.path.join(OUT, "fer_expw_v1.npz")


class DomainDataset(Dataset):
    """The provenance randomiser runs HERE, on the 48x48 uint8 crop, before any
    tensor transform - inside the exact code path the DataLoader feeds the
    classifier, not a preprocessing step applied somewhere else."""

    def __init__(self, images, domain, transform, provenance=None):
        self.images = images
        self.domain = torch.from_numpy(domain.astype(np.int64))
        self.transform = transform
        self.provenance = provenance

    def __len__(self):
        return len(self.domain)

    def __getitem__(self, i):
        img = self.images[i]
        if self.provenance is not None:
            img = self.provenance(img)
        x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)
        return self.transform(x), self.domain[i]



def verify_augmentation(images, domain, t_eval, make_aug, n=512):
    """Prove the augmentation is (or is not) present in the tensors the model
    consumes.

    Compares the SAME indices through the SAME Dataset.__getitem__ with and
    without the randomiser, under the DETERMINISTIC eval transform, so the only
    thing that can differ between the two tensors is the provenance chain.
    Also reports whether the FER+/ExpW texture gap - the statistic the probe was
    exploiting - actually narrows.
    """
    idx = np.arange(min(n, len(domain)))
    plain = DomainDataset(images, domain, t_eval, None)
    prov = make_aug(1234)
    enabled = prov is not None
    augmented = DomainDataset(images, domain, t_eval, prov)

    a = torch.stack([plain[i][0] for i in idx])
    b = torch.stack([augmented[i][0] for i in idx])
    diff_mask = (a != b).flatten(1).any(dim=1)
    differing = int(diff_mask.sum())
    mean_abs = float((a - b).abs().mean())
    calls_after_tensors = prov.calls if prov else 0

    def lap_var(arr_u8):
        if len(arr_u8) == 0:
            return 0.0
        return float(np.mean([cv2.Laplacian(im, cv2.CV_64F).var() for im in arr_u8]))

    raw = images[idx]
    raw_aug = np.stack([prov(im) for im in raw]) if enabled else raw
    is_expw = domain[idx] == 1
    gap_before = abs(lap_var(raw[~is_expw]) - lap_var(raw[is_expw]))
    gap_after = abs(lap_var(raw_aug[~is_expw]) - lap_var(raw_aug[is_expw]))

    result = {
        "augmentation_enabled": bool(enabled),
        "augmentation_class": type(prov).__name__ if enabled else None,
        "samples_checked": int(len(idx)),
        "tensors_differing_from_unaugmented": differing,
        "fraction_differing": round(differing / len(idx), 4),
        "mean_absolute_tensor_difference": round(mean_abs, 6),
        "randomiser_calls_while_building_those_tensors": calls_after_tensors,
        "texture_gap_ferplus_vs_expw_laplacian_variance": {
            "unaugmented": round(gap_before, 1),
            "augmented": round(gap_after, 1),
        },
    }
    if enabled:
        assert differing == len(idx), (
            "the randomiser did not alter every input the model sees "
            f"({differing}/{len(idx)})")
        assert calls_after_tensors == len(idx), (
            f"expected {len(idx)} randomiser calls while building the tensors, "
            f"got {calls_after_tensors}")
        assert mean_abs > 0, "tensors identical despite augmentation"
    else:
        assert differing == 0, "inputs changed with the augmentation disabled"
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--provenance-aug", action="store_true",
                    help="randomise the resampling history of BOTH corpora")
    ap.add_argument("--blur", action="store_true",
                    help="heavy expression-preserving low-pass on BOTH corpora; "
                         "tests whether the separability is low-level texture "
                         "or higher-level content")
    ap.add_argument("--blur-sigma", type=float, default=1.0,
                    help="Gaussian sigma in pixels on the 48x48 crop. 1.0 was "
                         "selected by measurement: it removes 94.7%% of the "
                         "FER+/ExpW texture gap while the emotion checkpoint "
                         "still reads the expression (see "
                         "processed/blur_sigma_selection.json)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filename, so the control and "
                         "the treatment results sit side by side")
    args = ap.parse_args()

    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA device not available - refusing to run on CPU.")

    d = np.load(NPZ)
    images, labels, split, source = (d["images"], d["labels"].astype(np.int64),
                                     d["split"], d["source"])
    classes = [str(c) for c in d["class_names"]]

    tr = split == "train"
    rng = np.random.default_rng(SEED)

    # Class-balanced pairing: equal FER+ and ExpW rows within every emotion, so
    # the emotion label cannot be used to infer the source.
    picked = []
    per_class = {}
    for i, c in enumerate(classes):
        f_idx = np.flatnonzero(tr & (source == "ferplus") & (labels == i))
        e_idx = np.flatnonzero(tr & (source == "expw") & (labels == i))
        n = min(len(f_idx), len(e_idx))
        per_class[c] = {"ferplus_available": int(len(f_idx)),
                        "expw_available": int(len(e_idx)), "paired": int(n)}
        if n == 0:
            continue
        picked.append(rng.choice(f_idx, n, replace=False))
        picked.append(rng.choice(e_idx, n, replace=False))
    idx = np.concatenate(picked)
    rng.shuffle(idx)

    x = images[idx]
    y = (source[idx] == "expw").astype(np.int64)          # 0 = FER+, 1 = ExpW
    emo = labels[idx]
    assert 0 < y.mean() < 1, "the probe set is single-class"

    n_val = int(round(len(y) * args.val_fraction))
    x_va, y_va, emo_va = x[:n_val], y[:n_val], emo[:n_val]
    x_tr, y_tr = x[n_val:], y[n_val:]
    print(f"probe set {len(y)} rows ({y.mean()*100:.1f}% ExpW) | "
          f"train {len(y_tr)} val {len(y_va)}")

    mean = float(x_tr.mean() / 255.0)
    std = float(x_tr.std() / 255.0)
    t_train, t_eval = build_transforms_v2(mean, std)

    assert not (args.provenance_aug and args.blur), \
        "run one probe input transform at a time, or the result attributes to neither"

    def make_aug(seed):
        """The single place a probe input transform is constructed, so the
        verification below and the loaders below cannot drift apart."""
        if args.provenance_aug:
            return ProvenanceRandomiser(seed=seed)
        if args.blur:
            return ExpressionPreservingBlur(args.blur_sigma)
        return None

    prov_tr = make_aug(SEED)
    prov_va = make_aug(SEED + 1)

    # ---- proof that the augmentation reaches the classifier's inputs -------
    verification = verify_augmentation(x_va, y_va, t_eval, make_aug)
    print(json.dumps(verification, indent=2), flush=True)

    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(DomainDataset(x_tr, y_tr, t_train, prov_tr),
                              batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(DomainDataset(x_va, y_va, t_eval, prov_va),
                            batch_size=256, shuffle=False, **common)

    model = build_model(2).to(device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_acc, best_cm, epoch_accs = 0.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        trues, preds = [], []
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(xb)
                trues.append(yb.numpy())
                preds.append(out.argmax(1).cpu().numpy())
        trues, preds = np.concatenate(trues), np.concatenate(preds)
        acc = float((trues == preds).mean())
        cm = confusion_matrix(trues, preds, 2)
        epoch_accs.append(round(acc, 4))
        print(f"  epoch {epoch}: domain accuracy {acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc, best_cm = acc, cm

    # The probe's best epoch is the adversary's best shot, which is the number
    # that matters: we want to know the WORST case for separability.
    if best_acc >= 0.75:
        verdict = ("SEPARABLE - the corpora are distinguishable from the 48x48 "
                   "crop. A merged model can use source as a shortcut to the "
                   "ExpW-dominated classes; treat merged disgust/fear numbers "
                   "as unreliable and prefer the ExpW holdout for those.")
    elif best_acc >= 0.60:
        verdict = ("PARTIALLY SEPARABLE - some residual corpus signal survives "
                   "the bottleneck. Rare-class gains are probably real but "
                   "inflated; confirm on the ExpW holdout and on real webcam "
                   "frames before trusting them.")
    else:
        verdict = ("NOT SEPARABLE - the shared BlazeFace + calibrated-crop + "
                   "48x48 grayscale bottleneck removed the corpus signature. "
                   "Merging is safe on this axis.")

    result = {
        "phase": ("7D" if args.provenance_aug
                  else "7E" if args.blur else "7C"),
        "question": "can a ResNet-18 tell FER+ from ExpW given only the 48x48 crop?",
        "method": "class-balanced pairing per emotion, so the emotion label "
                  "carries no information about the source; chance = 0.50",
        "probe_rows": int(len(y)),
        "val_rows": int(n_val),
        "epochs": args.epochs,
        "per_class_pairing": per_class,
        "chance_accuracy": 0.5,
        "probe_input_transform": (
            prov_tr.describe() if prov_tr is not None else {"enabled": False}),
        "probe_input_transform_class": (
            type(prov_tr).__name__ if prov_tr is not None else None),
        "augmentation_verification": verification,
        "epoch_accuracies": epoch_accs,
        "median_domain_accuracy": round(float(np.median(epoch_accs)), 4),
        "best_domain_accuracy": round(best_acc, 4),
        "confusion_matrix": best_cm.astype(int).tolist(),
        "confusion_matrix_axes": {"rows": ["true_ferplus", "true_expw"],
                                  "cols": ["pred_ferplus", "pred_expw"]},
        "verdict": verdict,
    }
    name = "domain_probe%s.json" % (("_" + args.tag) if args.tag else "")
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"epoch accuracies: {epoch_accs}")
    print(f"best domain accuracy {best_acc:.4f} | median "
          f"{float(np.median(epoch_accs)):.4f} | chance 0.5000")
    print(verdict)
    print("wrote:", os.path.join(OUT, name))


if __name__ == "__main__":
    main()
