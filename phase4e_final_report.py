from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_INFERENCE_DIR = (
    PROJECT_ROOT
    / "runs"
    / "phase4c_emotion_inference"
)

DEFAULT_TEMPORAL_DIR = (
    PROJECT_ROOT
    / "runs"
    / "phase4d_temporal_analysis"
)

DEFAULT_VIDEO_PIPELINE_DIR = (
    PROJECT_ROOT
    / "runs"
    / "phase4b_video_pipeline"
)

DEFAULT_TRAINING_DIR = (
    PROJECT_ROOT
    / "runs"
    / "baseline_resnet18"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "phase4e_final_report"
)


# =====================================================================
# EMOTIONS
# =====================================================================

EMOTIONS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
]


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def safe_float(value: Any) -> float | None:
    """
    Convert a value to a normal Python float.

    Returns None when the value is missing or invalid.
    """
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    """
    Convert a value to a normal Python integer.

    Returns None when the value is missing or invalid.
    """
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file does not exist:\n{path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object in:\n{path}"
        )

    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    """
    Save JSON using readable formatting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# =====================================================================
# CSV
# =====================================================================

def count_frame_results(csv_path: Path) -> int:
    """
    Count rows in frame_emotions.csv.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Frame results CSV does not exist:\n{csv_path}"
        )

    count = 0

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for _ in reader:
            count += 1

    return count


# =====================================================================
# TRAINING HISTORY
# =====================================================================

def load_best_training_metrics(
    history_path: Path,
) -> dict[str, Any]:
    """
    Read training history and find the epoch with the highest
    validation macro-F1.

    Expected columns include:

        epoch
        phase
        lr
        train_loss
        train_acc
        val_loss
        val_acc
        val_balanced_acc
        val_macro_f1
        seconds
    """

    if not history_path.exists():
        return {
            "available": False,
            "best_epoch": None,
            "best_validation_macro_f1": None,
            "history_path": str(history_path),
        }

    rows: list[dict[str, str]] = []

    with history_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    if not rows:
        return {
            "available": False,
            "best_epoch": None,
            "best_validation_macro_f1": None,
            "history_path": str(history_path),
        }

    best_row: dict[str, str] | None = None
    best_f1: float | None = None

    for row in rows:

        f1 = safe_float(
            row.get("val_macro_f1")
        )

        if f1 is None:
            continue

        if best_f1 is None or f1 > best_f1:
            best_f1 = f1
            best_row = row

    if best_row is None:
        return {
            "available": False,
            "best_epoch": None,
            "best_validation_macro_f1": None,
            "history_path": str(history_path),
        }

    return {
        "available": True,
        "best_epoch": safe_int(
            best_row.get("epoch")
        ),
        "best_validation_macro_f1": best_f1,
        "history_path": str(history_path),
        "best_phase": best_row.get("phase"),
        "best_learning_rate": safe_float(
            best_row.get("lr")
        ),
        "best_train_loss": safe_float(
            best_row.get("train_loss")
        ),
        "best_train_accuracy": safe_float(
            best_row.get("train_acc")
        ),
        "best_validation_loss": safe_float(
            best_row.get("val_loss")
        ),
        "best_validation_accuracy": safe_float(
            best_row.get("val_acc")
        ),
        "best_validation_balanced_accuracy": safe_float(
            best_row.get("val_balanced_acc")
        ),
        "best_seconds": safe_float(
            best_row.get("seconds")
        ),
    }


# =====================================================================
# PHASE 4C EXTRACTION
# =====================================================================

def extract_model_info(
    inference_stats: dict[str, Any],
    training_metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a consistent model section.

    Phase 4C stores model information under:

        inference_stats["model"]

    while training history provides the authoritative
    best validation macro-F1.
    """

    model_data = inference_stats.get(
        "model",
        {}
    )

    if not isinstance(model_data, dict):
        model_data = {}

    checkpoint = (
        model_data.get("checkpoint")
        or inference_stats.get("checkpoint")
        or str(
            DEFAULT_TRAINING_DIR
            / "best_model.pt"
        )
    )

    architecture = (
        model_data.get("architecture")
        or inference_stats.get("architecture")
        or "ResNet18"
    )

    device = (
        model_data.get("device")
        or inference_stats.get("device")
    )

    checkpoint_epoch = (
        model_data.get("epoch")
        or model_data.get("checkpoint_epoch")
        or inference_stats.get("checkpoint_epoch")
        or training_metrics.get("best_epoch")
    )

    validation_macro_f1 = (
        model_data.get("val_macro_f1")
        if model_data.get("val_macro_f1") is not None
        else inference_stats.get("validation_macro_f1")
    )

    # The training history is authoritative if Phase 4C contains None.
    if validation_macro_f1 is None:
        validation_macro_f1 = (
            training_metrics.get(
                "best_validation_macro_f1"
            )
        )

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": safe_int(
            checkpoint_epoch
        ),
        "validation_macro_f1": safe_float(
            validation_macro_f1
        ),
        "architecture": architecture,
        "device": device,
        "training_history": (
            training_metrics.get("history_path")
        ),
    }


# =====================================================================
# VIDEO INFORMATION
# =====================================================================

def extract_video_info(
    pipeline_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Extract useful video information from Phase 4B.

    Phase 4B structure can vary, so this function intentionally
    handles multiple possible field names.
    """

    if not isinstance(pipeline_stats, dict):
        return {}

    result: dict[str, Any] = {}

    video_data = pipeline_stats.get(
        "video",
        {}
    )

    if isinstance(video_data, dict):
        result.update(video_data)

    # Copy common top-level fields if present.
    candidate_fields = [
        "fps",
        "frame_count",
        "duration_seconds",
        "duration_s",
        "width",
        "height",
        "input",
        "input_video",
        "video_path",
    ]

    for key in candidate_fields:
        if key in pipeline_stats and key not in result:
            result[key] = pipeline_stats[key]

    return result


# =====================================================================
# FINAL REPORT
# =====================================================================

def build_final_report(
    inference_stats: dict[str, Any],
    temporal_report: dict[str, Any],
    pipeline_stats: dict[str, Any] | None,
    frame_count: int,
    training_metrics: dict[str, Any],
) -> dict[str, Any]:

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    model = extract_model_info(
        inference_stats,
        training_metrics,
    )

    # -----------------------------------------------------------------
    # Temporal analysis
    # -----------------------------------------------------------------

    dominant_emotion = temporal_report.get(
        "dominant_emotion"
    )

    emotion_distribution = temporal_report.get(
        "emotion_distribution_percent",
        {},
    )

    temporal_label_distribution = temporal_report.get(
        "temporal_label_distribution_percent",
        {},
    )

    confidence = temporal_report.get(
        "confidence",
        {},
    )

    segments = temporal_report.get(
        "segments",
        [],
    )

    transitions = temporal_report.get(
        "transitions",
        [],
    )

    transition_count = temporal_report.get(
        "transition_count"
    )

    if transition_count is None:
        transition_count = len(transitions)

    # -----------------------------------------------------------------
    # Phase 4C prediction information
    # -----------------------------------------------------------------

    predictions = inference_stats.get(
        "predictions",
        {},
    )

    if not isinstance(predictions, dict):
        predictions = {}

    prediction_counts = predictions.get(
        "counts",
        {},
    )

    if not isinstance(prediction_counts, dict):
        prediction_counts = {}

    # -----------------------------------------------------------------
    # Video information
    # -----------------------------------------------------------------

    video = extract_video_info(
        pipeline_stats
    )

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------

    report: dict[str, Any] = {

        "phase": "4E",

        "description": (
            "Final video-based facial expression "
            "and temporal emotion analysis report"
        ),

        "status": "complete",

        # -------------------------------------------------------------
        # Input
        # -------------------------------------------------------------

        "input": {
            "frames_analyzed": frame_count,

            "video": video,
        },

        # -------------------------------------------------------------
        # Model
        # -------------------------------------------------------------

        "model": model,

        # -------------------------------------------------------------
        # Frame-level inference
        # -------------------------------------------------------------

        "frame_level_inference": {

            "prediction_counts": prediction_counts,

            "confidence": {
                "min": safe_float(
                    predictions.get(
                        "confidence_min"
                    )
                ),

                "p05": safe_float(
                    predictions.get(
                        "confidence_p05"
                    )
                ),

                "median": safe_float(
                    predictions.get(
                        "confidence_median"
                    )
                ),

                "mean": safe_float(
                    predictions.get(
                        "confidence_mean"
                    )
                ),

                "p95": safe_float(
                    predictions.get(
                        "confidence_p95"
                    )
                ),

                "max": safe_float(
                    predictions.get(
                        "confidence_max"
                    )
                ),
            },
        },

        # -------------------------------------------------------------
        # Temporal analysis
        # -------------------------------------------------------------

        "temporal_analysis": {

            "dominant_emotion": dominant_emotion,

            "emotion_distribution_percent": (
                emotion_distribution
            ),

            "temporal_label_distribution_percent": (
                temporal_label_distribution
            ),

            "confidence": confidence,

            "segments": segments,

            "transitions": transitions,

            "segment_count": len(segments),

            "transition_count": int(
                transition_count
            ),
        },

        # -------------------------------------------------------------
        # Interpretation
        # -------------------------------------------------------------

        "interpretation": {

            "dominant_emotion_definition": (
                "The dominant emotion is the emotion "
                "with the highest temporally smoothed "
                "probability distribution across the "
                "analyzed video."
            ),

            "emotion_distribution_definition": (
                "Emotion percentages represent the "
                "temporal distribution of model "
                "probabilities after smoothing, rather "
                "than raw frame-label counts."
            ),

            "temporal_label_distribution_definition": (
                "Temporal label percentages represent "
                "the proportion of analyzed samples "
                "assigned to each emotion after temporal "
                "smoothing."
            ),

            "confidence_definition": (
                "Confidence describes the model's "
                "probability for its selected class "
                "and should not be interpreted as "
                "classification accuracy."
            ),

            "limitation": (
                "Facial-expression classification "
                "estimates visible facial expression "
                "patterns. It does not establish a "
                "person's true internal emotional state."
            ),
        },

        # -------------------------------------------------------------
        # Artifacts
        # -------------------------------------------------------------

        "artifacts": {

            "phase4c_inference_stats": str(
                DEFAULT_INFERENCE_DIR
                / "inference_stats.json"
            ),

            "phase4c_frame_results": str(
                DEFAULT_INFERENCE_DIR
                / "frame_emotions.csv"
            ),

            "phase4d_temporal_report": str(
                DEFAULT_TEMPORAL_DIR
                / "temporal_report.json"
            ),

            "phase4d_temporal_results": str(
                DEFAULT_TEMPORAL_DIR
                / "temporal_emotions.csv"
            ),

            "phase4b_pipeline_stats": str(
                DEFAULT_VIDEO_PIPELINE_DIR
                / "pipeline_stats.json"
            ),

            "training_history": str(
                DEFAULT_TRAINING_DIR
                / "history.csv"
            ),

            "model_checkpoint": str(
                model["checkpoint"]
            ),
        },
    }

    return report


# =====================================================================
# CONSOLE OUTPUT
# =====================================================================

def print_report(
    report: dict[str, Any],
) -> None:

    print()
    print("=" * 72)
    print("PHASE 4E — FINAL EMOTION REPORT")
    print("=" * 72)

    print()
    print("analysis status:")
    print(
        f"  {report.get('status')}"
    )

    print()
    print("frames analyzed:")
    print(
        f"  {report['input']['frames_analyzed']}"
    )

    # -----------------------------------------------------------------
    # Dominant emotion
    # -----------------------------------------------------------------

    temporal = report["temporal_analysis"]

    print()
    print("dominant emotion:")
    print(
        f"  {temporal['dominant_emotion']}"
    )

    # -----------------------------------------------------------------
    # Emotion distribution
    # -----------------------------------------------------------------

    print()
    print("emotion distribution:")

    distribution = (
        temporal[
            "emotion_distribution_percent"
        ]
    )

    for emotion in EMOTIONS:

        value = distribution.get(
            emotion,
            0.0,
        )

        numeric = safe_float(value)

        if numeric is None:
            numeric = 0.0

        print(
            f"  {emotion:<10}: "
            f"{numeric:.2f}%"
        )

    # -----------------------------------------------------------------
    # Temporal labels
    # -----------------------------------------------------------------

    print()
    print("temporal label distribution:")

    labels = (
        temporal[
            "temporal_label_distribution_percent"
        ]
    )

    for emotion in EMOTIONS:

        value = labels.get(
            emotion,
            0.0,
        )

        numeric = safe_float(value)

        if numeric is None:
            numeric = 0.0

        print(
            f"  {emotion:<10}: "
            f"{numeric:.2f}%"
        )

    # -----------------------------------------------------------------
    # Temporal confidence
    # -----------------------------------------------------------------

    confidence = temporal.get(
        "confidence",
        {},
    )

    print()
    print("temporal confidence:")

    min_conf = safe_float(
        confidence.get("min")
    )

    mean_conf = safe_float(
        confidence.get("mean")
    )

    max_conf = safe_float(
        confidence.get("max")
    )

    print(
        f"  min:  "
        f"{min_conf:.4f}"
        if min_conf is not None
        else "  min:  None"
    )

    print(
        f"  mean: "
        f"{mean_conf:.4f}"
        if mean_conf is not None
        else "  mean: None"
    )

    print(
        f"  max:  "
        f"{max_conf:.4f}"
        if max_conf is not None
        else "  max:  None"
    )

    # -----------------------------------------------------------------
    # Temporal structure
    # -----------------------------------------------------------------

    print()
    print("temporal structure:")

    print(
        f"  segments: "
        f"{temporal['segment_count']}"
    )

    print(
        f"  transitions: "
        f"{temporal['transition_count']}"
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    print()
    print("model:")

    model = report["model"]

    print(
        f"  architecture: "
        f"{model.get('architecture')}"
    )

    print(
        f"  checkpoint epoch: "
        f"{model.get('checkpoint_epoch')}"
    )

    f1 = model.get(
        "validation_macro_f1"
    )

    if f1 is None:
        print(
            "  validation macro-F1: None"
        )
    else:
        print(
            f"  validation macro-F1: "
            f"{f1:.6f}"
        )

    print(
        f"  training history: "
        f"{model.get('training_history')}"
    )

    # -----------------------------------------------------------------
    # Frame-level confidence
    # -----------------------------------------------------------------

    frame_inf = report[
        "frame_level_inference"
    ]

    frame_conf = frame_inf[
        "confidence"
    ]

    print()
    print("frame-level confidence:")

    print(
        f"  min:    "
        f"{frame_conf['min']:.4f}"
    )

    print(
        f"  median: "
        f"{frame_conf['median']:.4f}"
    )

    print(
        f"  mean:   "
        f"{frame_conf['mean']:.4f}"
    )

    print(
        f"  max:    "
        f"{frame_conf['max']:.4f}"
    )

    # -----------------------------------------------------------------
    # End
    # -----------------------------------------------------------------

    print()
    print("=" * 72)


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Phase 4E — build the final "
            "emotion analysis report."
        )
    )

    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=DEFAULT_INFERENCE_DIR,
    )

    parser.add_argument(
        "--temporal-dir",
        type=Path,
        default=DEFAULT_TEMPORAL_DIR,
    )

    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DEFAULT_VIDEO_PIPELINE_DIR,
    )

    parser.add_argument(
        "--training-dir",
        type=Path,
        default=DEFAULT_TRAINING_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    args = parse_args()

    inference_dir = args.inference_dir
    temporal_dir = args.temporal_dir
    pipeline_dir = args.pipeline_dir
    training_dir = args.training_dir
    output_dir = args.output_dir

    # -----------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------

    inference_stats_path = (
        inference_dir
        / "inference_stats.json"
    )

    frame_results_path = (
        inference_dir
        / "frame_emotions.csv"
    )

    temporal_report_path = (
        temporal_dir
        / "temporal_report.json"
    )

    pipeline_stats_path = (
        pipeline_dir
        / "pipeline_stats.json"
    )

    history_path = (
        training_dir
        / "history.csv"
    )

    output_path = (
        output_dir
        / "final_report.json"
    )

    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 4E — FINAL EMOTION REPORT")
    print("=" * 72)

    # -----------------------------------------------------------------
    # Load Phase 4C
    # -----------------------------------------------------------------

    print()
    print("loading Phase 4C...")
    print(
        f"  {inference_stats_path}"
    )

    inference_stats = load_json(
        inference_stats_path
    )

    # -----------------------------------------------------------------
    # Load frame results
    # -----------------------------------------------------------------

    print("loading frame results...")
    print(
        f"  {frame_results_path}"
    )

    frame_count = count_frame_results(
        frame_results_path
    )

    print(
        f"  frames: {frame_count}"
    )

    # -----------------------------------------------------------------
    # Load Phase 4D
    # -----------------------------------------------------------------

    print()
    print("loading Phase 4D...")
    print(
        f"  {temporal_report_path}"
    )

    temporal_report = load_json(
        temporal_report_path
    )

    # -----------------------------------------------------------------
    # Load Phase 4B
    # -----------------------------------------------------------------

    print()
    print("loading Phase 4B...")

    pipeline_stats: dict[str, Any] | None = None

    if pipeline_stats_path.exists():

        print(
            f"  {pipeline_stats_path}"
        )

        pipeline_stats = load_json(
            pipeline_stats_path
        )

    else:

        print(
            f"  not found: "
            f"{pipeline_stats_path}"
        )

        print(
            "  continuing without Phase 4B "
            "pipeline statistics"
        )

    # -----------------------------------------------------------------
    # Load training history
    # -----------------------------------------------------------------

    print()
    print("loading training history...")
    print(
        f"  {history_path}"
    )

    training_metrics = (
        load_best_training_metrics(
            history_path
        )
    )

    if training_metrics["available"]:

        print(
            f"  best epoch: "
            f"{training_metrics['best_epoch']}"
        )

        print(
            f"  best validation macro-F1: "
            f"{training_metrics['best_validation_macro_f1']:.6f}"
        )

    else:

        print(
            "  training history metrics "
            "unavailable"
        )

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    print()
    print("building final report...")

    report = build_final_report(
        inference_stats=inference_stats,
        temporal_report=temporal_report,
        pipeline_stats=pipeline_stats,
        frame_count=frame_count,
        training_metrics=training_metrics,
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        output_path,
        report,
    )

    # -----------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------

    print_report(
        report
    )

    print()
    print("artifacts:")
    print(
        f"  JSON: {output_path}"
    )

    print()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()