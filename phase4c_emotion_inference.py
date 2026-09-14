from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "phase4b_video_pipeline"
    / "crops"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "baseline_resnet18"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "phase4c_emotion_inference"
)

OUTPUT_CSV = OUTPUT_DIR / "frame_emotions.csv"
OUTPUT_JSON = OUTPUT_DIR / "inference_stats.json"


# =====================================================================
# MODEL CONFIGURATION
# Must match the trained baseline exactly.
# =====================================================================

IMG_SIZE = 112

MEAN = 0.5043038587546841
STD = 0.25457988035377677

CLASS_NAMES = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
]


# =====================================================================
# EVALUATION PREPROCESSING
# =====================================================================

def build_eval_transform():
    """
    Exact inference/evaluation preprocessing.

    Phase 4B crop:
        48x48 grayscale

    Then:
        PIL image
        -> torchvision Image
        -> resize to 112x112
        -> float32 in [0, 1]
        -> normalization
    """

    return v2.Compose([
        v2.ToImage(),
        v2.Resize(
            IMG_SIZE,
            antialias=True,
        ),
        v2.ToDtype(
            torch.float32,
            scale=True,
        ),
        v2.Normalize(
            mean=[MEAN],
            std=[STD],
        ),
    ])


# =====================================================================
# MODEL
# =====================================================================

def build_model(num_classes: int) -> nn.Module:
    """
    Reconstruct the exact ResNet18 architecture used for training.

    Original:
        ImageNet ResNet18

    Modification:
        RGB conv1 -> 1-channel grayscale conv1

    The RGB weights are summed across the input-channel dimension,
    matching the training implementation.
    """

    model = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )

    old_conv1 = model.conv1

    new_conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=False,
    )

    with torch.no_grad():
        new_conv1.weight.copy_(
            old_conv1.weight.sum(
                dim=1,
                keepdim=True,
            )
        )

    model.conv1 = new_conv1

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


# =====================================================================
# CHECKPOINT LOADING
# =====================================================================

def load_checkpoint(
    model: nn.Module,
    device: torch.device,
):
    """
    Load the already-trained best checkpoint.

    No training is performed.
    """

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    epoch = None
    val_macro_f1 = None

    # -------------------------------------------------------------
    # Case 1: checkpoint contains model_state
    # -------------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state" in checkpoint
    ):

        state_dict = checkpoint["model_state"]

        epoch = checkpoint.get("epoch")
        val_macro_f1 = checkpoint.get(
            "val_macro_f1"
        )

    # -------------------------------------------------------------
    # Case 2: checkpoint contains state_dict
    # -------------------------------------------------------------

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint["state_dict"]

        epoch = checkpoint.get("epoch")
        val_macro_f1 = checkpoint.get(
            "val_macro_f1"
        )

    # -------------------------------------------------------------
    # Case 3: checkpoint itself is state_dict
    # -------------------------------------------------------------

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)

    model.eval()

    return epoch, val_macro_f1


# =====================================================================
# FRAME NUMBER EXTRACTION
# =====================================================================

def extract_frame_number(filename: str) -> int:
    """
    Extract original frame number.

    Example:

        sample_0000_frame_000000.png

    returns:

        0
    """

    marker = "_frame_"

    if marker not in filename:
        return -1

    value = filename.split(
        marker,
        1,
    )[1]

    value = value.rsplit(
        ".",
        1,
    )[0]

    try:
        return int(value)

    except ValueError:
        return -1


# =====================================================================
# SAFE JSON NUMBER
# =====================================================================

def safe_json_number(value):
    """Convert NumPy/Python numeric values into JSON-safe values."""
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value

# =====================================================================
# MAIN
# =====================================================================

def main():

    # -------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Validate input directory
    # -------------------------------------------------------------

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"\nInput crop directory does not exist:\n"
            f"{INPUT_DIR}\n"
        )

    # -------------------------------------------------------------
    # Validate checkpoint
    # -------------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel checkpoint does not exist:\n"
            f"{MODEL_PATH}\n"
        )

    # -------------------------------------------------------------
    # Select device
    # -------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -------------------------------------------------------------
    # Display device
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 4C — EMOTION INFERENCE")
    print("=" * 72)

    print(
        f"device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------

    print()
    print("loading model...")

    model = build_model(
        len(CLASS_NAMES)
    )

    epoch, val_macro_f1 = load_checkpoint(
        model,
        device,
    )

    print(
        f"checkpoint: {MODEL_PATH}"
    )

    print(
        f"checkpoint epoch: {epoch}"
    )

    print(
        f"validation macro-F1: {val_macro_f1}"
    )

    # -------------------------------------------------------------
    # Build preprocessing
    # -------------------------------------------------------------

    transform = build_eval_transform()

    # -------------------------------------------------------------
    # Find Phase 4B crops
    # -------------------------------------------------------------

    image_files = sorted(
        INPUT_DIR.glob(
            "sample_*_frame_*.png"
        ),
        key=lambda path: extract_frame_number(
            path.name
        ),
    )

    if not image_files:

        raise RuntimeError(
            f"\nNo Phase 4B crops found in:\n"
            f"{INPUT_DIR}\n"
        )

    print()
    print(
        f"input crops: {len(image_files)}"
    )

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------

    rows = []

    print()
    print("running inference...")

    with torch.inference_mode():

        for sample_index, image_path in enumerate(
            image_files
        ):

            # -------------------------------------------------
            # Load grayscale crop
            # -------------------------------------------------

            image = Image.open(
                image_path
            ).convert("L")

            # -------------------------------------------------
            # Apply evaluation transform
            # -------------------------------------------------

            tensor = transform(image)

            # -------------------------------------------------
            # Add batch dimension
            #
            # [1, H, W]
            #      ↓
            # [1, 1, H, W]
            # -------------------------------------------------

            tensor = tensor.unsqueeze(0)

            tensor = tensor.to(device)

            # -------------------------------------------------
            # Model inference
            # -------------------------------------------------

            logits = model(tensor)

            # -------------------------------------------------
            # Convert logits to probabilities
            # -------------------------------------------------

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

            # -------------------------------------------------
            # Predicted class
            # -------------------------------------------------

            predicted_index = int(
                torch.argmax(
                    probabilities
                ).item()
            )

            predicted_emotion = (
                CLASS_NAMES[predicted_index]
            )

            confidence = float(
                probabilities[
                    predicted_index
                ].item()
            )

            # -------------------------------------------------
            # Original video frame
            # -------------------------------------------------

            frame_number = extract_frame_number(
                image_path.name
            )

            # -------------------------------------------------
            # Create row
            # -------------------------------------------------

            row = {
                "sample_index": sample_index,
                "frame_number": frame_number,
                "image": image_path.name,
                "predicted_emotion": predicted_emotion,
                "confidence": confidence,
            }

            # -------------------------------------------------
            # Store every class probability
            # -------------------------------------------------

            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):

                row[
                    f"prob_{class_name}"
                ] = float(
                    probabilities[
                        class_index
                    ].item()
                )

            rows.append(row)

    # =================================================================
    # WRITE CSV
    # =================================================================

    fieldnames = [
        "sample_index",
        "frame_number",
        "image",
        "predicted_emotion",
        "confidence",
    ]

    fieldnames.extend(
        f"prob_{class_name}"
        for class_name in CLASS_NAMES
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    # =================================================================
    # CALCULATE BASIC STATISTICS
    # =================================================================

    prediction_counts = {
        emotion: 0
        for emotion in CLASS_NAMES
    }

    confidences = []

    for row in rows:

        prediction_counts[
            row["predicted_emotion"]
        ] += 1

        confidences.append(
            row["confidence"]
        )

    confidence_array = np.asarray(
        confidences,
        dtype=np.float64,
    )

    # =================================================================
    # STATISTICS
    # =================================================================

    stats = {

        "input": {

            "directory": str(
                INPUT_DIR
            ),

            "frames": len(rows),
        },

        "model": {

            "checkpoint": str(
                MODEL_PATH
            ),

            "epoch": epoch,

            "val_macro_f1":
                safe_json_number(
                    val_macro_f1
                ),

            "device": str(
                device
            ),

            "architecture":
                "ResNet18",
        },

        "preprocessing": {

            "input_crop":
                "48x48 grayscale",

            "resize":
                "112x112",

            "mean":
                MEAN,

            "std":
                STD,

            "augmentation":
                False,
        },

        "predictions": {

            "counts":
                prediction_counts,

            "confidence_min":
                safe_json_number(
                    confidence_array.min()
                ),

            "confidence_p05":
                safe_json_number(
                    np.percentile(
                        confidence_array,
                        5,
                    )
                ),

            "confidence_median":
                safe_json_number(
                    np.median(
                        confidence_array
                    )
                ),

            "confidence_mean":
                safe_json_number(
                    confidence_array.mean()
                ),

            "confidence_p95":
                safe_json_number(
                    np.percentile(
                        confidence_array,
                        95,
                    )
                ),

            "confidence_max":
                safe_json_number(
                    confidence_array.max()
                ),
        },
    }

    # =================================================================
    # WRITE JSON
    # =================================================================

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            stats,
            file,
            indent=2,
        )

    # =================================================================
    # CONSOLE SUMMARY
    # =================================================================

    print()
    print("=" * 72)
    print("INFERENCE COMPLETE")
    print("=" * 72)

    print()
    print(
        f"frames processed: {len(rows)}"
    )

    print()
    print("prediction counts:")

    for emotion in CLASS_NAMES:

        print(
            f"  {emotion:10s}: "
            f"{prediction_counts[emotion]}"
        )

    print()
    print("confidence:")

    print(
        f"  min:    "
        f"{confidence_array.min():.4f}"
    )

    print(
        f"  p05:    "
        f"{np.percentile(confidence_array, 5):.4f}"
    )

    print(
        f"  median: "
        f"{np.median(confidence_array):.4f}"
    )

    print(
        f"  mean:   "
        f"{confidence_array.mean():.4f}"
    )

    print(
        f"  p95:    "
        f"{np.percentile(confidence_array, 95):.4f}"
    )

    print(
        f"  max:    "
        f"{confidence_array.max():.4f}"
    )

    print()
    print("artifacts:")

    print(
        f"  CSV:  {OUTPUT_CSV}"
    )

    print(
        f"  JSON: {OUTPUT_JSON}"
    )

    print()
    print("=" * 72)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()