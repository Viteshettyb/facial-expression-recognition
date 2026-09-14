"""Emotion classifier: checkpoint loading and batched GPU inference.

Every runtime constant - class names, class ordering, image size, normalisation
mean/std - is read FROM THE CHECKPOINT, never hardcoded. Hardcoding them is the
one failure mode that produces silently wrong answers: the state dict still
loads, the softmax still sums to 1, and nothing raises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import v2


@dataclass
class ModelInfo:
    checkpoint: str
    epoch: int | None
    val_macro_f1: float | None
    val_accuracy: float | None
    val_balanced_accuracy: float | None
    architecture: str
    class_names: list[str]
    img_size: int
    norm_mean: float
    norm_std: float
    device: str


def _build_resnet18_gray(num_classes: int) -> nn.Module:
    # weights=None: ImageNet weights would be overwritten by the checkpoint
    # anyway, and downloading them makes startup depend on network access.
    model = resnet18(weights=None)
    old = model.conv1
    model.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class EmotionClassifier:
    """Loaded once at process start and reused for every request."""

    def __init__(self, checkpoint_path: str, device: torch.device | None = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        cfg = ckpt["config"]

        self.class_names: list[str] = [str(c) for c in cfg["class_names"]]
        self.img_size: int = int(cfg["img_size"])
        self.norm_mean: float = float(cfg["norm_mean"])
        self.norm_std: float = float(cfg["norm_std"])
        if len(self.class_names) != 7:
            raise ValueError(f"expected 7 classes, got {len(self.class_names)}")

        self.model = _build_resnet18_gray(len(self.class_names))
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.eval().to(self.device, memory_format=torch.channels_last)

        # Identical to train_baseline.build_transforms(...)[1].
        self.transform = v2.Compose([
            v2.Resize(self.img_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[self.norm_mean], std=[self.norm_std]),
        ])

        vm = ckpt.get("val_metrics") or {}
        self.info = ModelInfo(
            checkpoint=checkpoint_path,
            epoch=ckpt.get("epoch"),
            val_macro_f1=vm.get("macro_f1"),          # correct key
            val_accuracy=vm.get("accuracy"),
            val_balanced_accuracy=vm.get("balanced_accuracy"),
            architecture=str(cfg.get("backbone", "resnet18")),
            class_names=self.class_names,
            img_size=self.img_size,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            device=str(self.device),
        )

    @torch.inference_mode()
    def predict(self, faces48: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """(N,48,48) uint8 crops -> (N,7) float32 probabilities."""
        if not faces48:
            return np.zeros((0, len(self.class_names)), dtype=np.float32)

        out = []
        use_amp = self.device.type == "cuda"
        for i in range(0, len(faces48), batch_size):
            chunk = faces48[i:i + batch_size]
            batch = torch.from_numpy(
                np.stack(chunk).astype(np.uint8)).unsqueeze(1)   # (B,1,48,48)
            batch = self.transform(batch).to(
                self.device, non_blocking=True).to(memory_format=torch.channels_last)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(batch)
            else:
                logits = self.model(batch)
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
