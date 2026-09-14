from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


EMOTIONS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
]

PROB_COLUMNS = {
    emotion: f"prob_{emotion}"
    for emotion in EMOTIONS
}


# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------

def safe_json_number(value):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if value != value:
        return None

    if value == float("inf") or value == float("-inf"):
        return None

    return value


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------

def load_frame_results(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {
            "sample_index",
            "frame_number",
            "image",
            "predicted_emotion",
            "confidence",
            *PROB_COLUMNS.values(),
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            probabilities = {}

            for emotion, column in PROB_COLUMNS.items():
                probabilities[emotion] = float(row[column])

            probability_sum = sum(probabilities.values())

            if probability_sum <= 0:
                raise ValueError(
                    f"Invalid probability vector at sample "
                    f"{row['sample_index']}"
                )

            # Normalize defensively in case of tiny floating-point drift.
            probabilities = {
                emotion: value / probability_sum
                for emotion, value in probabilities.items()
            }

            rows.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "frame_number": int(row["frame_number"]),
                    "image": row["image"],
                    "predicted_emotion": row["predicted_emotion"],
                    "confidence": float(row["confidence"]),
                    "probabilities": probabilities,
                }
            )

    if not rows:
        raise ValueError("CSV contains no frame results.")

    rows.sort(key=lambda x: x["sample_index"])

    return rows


# ---------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------

def moving_average(
    rows: list[dict],
    window: int,
) -> list[dict]:
    """
    Centered moving-average smoothing over emotion probabilities.

    Example:
        window=5 at ~5 FPS ≈ 1 second temporal context.
    """

    if window < 1:
        raise ValueError("Smoothing window must be >= 1.")

    if window % 2 == 0:
        raise ValueError("Smoothing window must be odd.")

    n = len(rows)
    half = window // 2

    smoothed = []

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)

        neighborhood = rows[start:end]

        averaged = {}

        for emotion in EMOTIONS:
            averaged[emotion] = mean(
                item["probabilities"][emotion]
                for item in neighborhood
            )

        total = sum(averaged.values())

        averaged = {
            emotion: value / total
            for emotion, value in averaged.items()
        }

        predicted = max(
            EMOTIONS,
            key=lambda emotion: averaged[emotion],
        )

        smoothed.append(
            {
                "sample_index": rows[i]["sample_index"],
                "frame_number": rows[i]["frame_number"],
                "image": rows[i]["image"],
                "original_prediction": rows[i]["predicted_emotion"],
                "original_confidence": rows[i]["confidence"],
                "smoothed_probabilities": averaged,
                "smoothed_emotion": predicted,
                "smoothed_confidence": averaged[predicted],
            }
        )

    return smoothed


# ---------------------------------------------------------------------
# Short-run cleanup
# ---------------------------------------------------------------------

def remove_short_runs(
    smoothed: list[dict],
    min_run: int = 2,
) -> list[dict]:
    """
    Removes very short one-sample emotion spikes.

    A short run is replaced by the emotion immediately before it
    when possible; otherwise by the following emotion.
    """

    if min_run <= 1:
        return smoothed

    labels = [
        item["smoothed_emotion"]
        for item in smoothed
    ]

    n = len(labels)
    cleaned = labels.copy()

    i = 0

    while i < n:
        j = i + 1

        while j < n and labels[j] == labels[i]:
            j += 1

        run_length = j - i

        if run_length < min_run:
            previous_label = (
                labels[i - 1]
                if i > 0
                else None
            )

            next_label = (
                labels[j]
                if j < n
                else None
            )

            replacement = previous_label or next_label

            if replacement is not None:
                for k in range(i, j):
                    cleaned[k] = replacement

        i = j

    result = []

    for index, item in enumerate(smoothed):
        updated = dict(item)
        updated["temporal_emotion"] = cleaned[index]
        result.append(updated)

    return result


# ---------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------

def build_segments(
    rows: list[dict],
    fps: float,
) -> list[dict]:

    if not rows:
        return []

    segments = []

    start_index = 0
    current_emotion = rows[0]["temporal_emotion"]

    for i in range(1, len(rows)):
        emotion = rows[i]["temporal_emotion"]

        if emotion != current_emotion:
            segment_rows = rows[start_index:i]

            segments.append(
                make_segment(
                    segment_rows,
                    current_emotion,
                    fps,
                )
            )

            start_index = i
            current_emotion = emotion

    segment_rows = rows[start_index:]

    segments.append(
        make_segment(
            segment_rows,
            current_emotion,
            fps,
        )
    )

    return segments


def make_segment(
    rows: list[dict],
    emotion: str,
    fps: float,
) -> dict:

    start_frame = rows[0]["frame_number"]
    end_frame = rows[-1]["frame_number"]

    start_time = start_frame / fps
    end_time = end_frame / fps

    duration = max(
        1.0 / fps,
        end_time - start_time + (1.0 / fps),
    )

    confidence_values = [
        row["smoothed_probabilities"][emotion]
        for row in rows
    ]

    return {
        "emotion": emotion,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_s": safe_json_number(start_time),
        "end_time_s": safe_json_number(end_time),
        "duration_s": safe_json_number(duration),
        "frames": len(rows),
        "mean_confidence": safe_json_number(
            mean(confidence_values)
        ),
    }


# ---------------------------------------------------------------------
# Overall report
# ---------------------------------------------------------------------

def build_report(
    original: list[dict],
    smoothed: list[dict],
    segments: list[dict],
    fps: float,
    smoothing_window: int,
    min_run: int,
) -> dict:

    total_frames = len(smoothed)

    # Overall distribution is based on the temporally smoothed
    # probability vectors.
    distribution = {}

    for emotion in EMOTIONS:
        distribution[emotion] = mean(
            row["smoothed_probabilities"][emotion]
            for row in smoothed
        )

    distribution_percent = {
        emotion: safe_json_number(
            distribution[emotion] * 100.0
        )
        for emotion in EMOTIONS
    }

    dominant_emotion = max(
        EMOTIONS,
        key=lambda emotion: distribution[emotion],
    )

    dominant_percentage = distribution_percent[dominant_emotion]

    confidence_values = [
        row["smoothed_confidence"]
        for row in smoothed
    ]

    temporal_labels = [
        row["temporal_emotion"]
        for row in smoothed
    ]

    transitions = []

    for i in range(1, len(temporal_labels)):
        previous = temporal_labels[i - 1]
        current = temporal_labels[i]

        if previous != current:
            transitions.append(
                {
                    "from": previous,
                    "to": current,
                    "frame_number": smoothed[i]["frame_number"],
                    "time_s": safe_json_number(
                        smoothed[i]["frame_number"] / fps
                    ),
                }
            )

    frame_counts = {
        emotion: temporal_labels.count(emotion)
        for emotion in EMOTIONS
    }

    frame_percentages = {
        emotion: safe_json_number(
            frame_counts[emotion] / total_frames * 100.0
        )
        for emotion in EMOTIONS
    }

    return {
        "phase": "4D",
        "description": "Temporal emotion analysis",

        "input": {
            "frames": total_frames,
            "fps": safe_json_number(fps),
        },

        "temporal_smoothing": {
            "method": "centered moving average",
            "window_samples": smoothing_window,
            "approx_window_seconds": safe_json_number(
                smoothing_window / fps
            ),
            "short_run_min_samples": min_run,
        },

        "dominant_emotion": dominant_emotion,

        "emotion_distribution_percent": distribution_percent,

        "temporal_label_distribution_percent": frame_percentages,

        "confidence": {
            "min": safe_json_number(min(confidence_values)),
            "mean": safe_json_number(mean(confidence_values)),
            "max": safe_json_number(max(confidence_values)),
        },

        "frame_counts_after_temporal_smoothing": frame_counts,

        "segments": segments,

        "transitions": transitions,

        "transition_count": len(transitions),
    }


# ---------------------------------------------------------------------
# Output CSV
# ---------------------------------------------------------------------

def write_temporal_csv(
    rows: list[dict],
    output_path: Path,
    fps: float,
):

    fieldnames = [
        "sample_index",
        "frame_number",
        "time_s",
        "image",
        "original_prediction",
        "original_confidence",
        "temporal_emotion",
        "smoothed_confidence",
        *[
            f"smoothed_prob_{emotion}"
            for emotion in EMOTIONS
        ],
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            output_row = {
                "sample_index": row["sample_index"],
                "frame_number": row["frame_number"],
                "time_s": safe_json_number(
                    row["frame_number"] / fps
                ),
                "image": row["image"],
                "original_prediction": row[
                    "original_prediction"
                ],
                "original_confidence": safe_json_number(
                    row["original_confidence"]
                ),
                "temporal_emotion": row[
                    "temporal_emotion"
                ],
                "smoothed_confidence": safe_json_number(
                    row["smoothed_confidence"]
                ),
            }

            for emotion in EMOTIONS:
                output_row[
                    f"smoothed_prob_{emotion}"
                ] = safe_json_number(
                    row["smoothed_probabilities"][emotion]
                )

            writer.writerow(output_row)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Phase 4D temporal emotion analysis"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "runs/phase4c_emotion_inference/"
            "frame_emotions.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/phase4d_temporal_analysis"
        ),
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=23.976023976023978,
        help="Original video FPS.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Odd temporal smoothing window in samples.",
    )

    parser.add_argument(
        "--min-run",
        type=int,
        default=2,
        help="Minimum number of samples for a temporal run.",
    )

    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than zero.")

    if args.window < 1 or args.window % 2 == 0:
        parser.error("--window must be a positive odd number.")

    if args.min_run < 1:
        parser.error("--min-run must be >= 1.")

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("PHASE 4D — TEMPORAL EMOTION ANALYSIS")
    print("=" * 72)

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"video FPS: {args.fps:.6f}")
    print(f"smoothing window: {args.window} samples")
    print(
        f"approx smoothing window: "
        f"{args.window / args.fps:.2f}s"
    )
    print(f"minimum temporal run: {args.min_run}")

    print()
    print("loading frame results...")

    original = load_frame_results(args.input)

    print(f"frames loaded: {len(original)}")

    print()
    print("applying temporal smoothing...")

    smoothed = moving_average(
        original,
        window=args.window,
    )

    smoothed = remove_short_runs(
        smoothed,
        min_run=args.min_run,
    )

    print("temporal smoothing complete")

    print()
    print("building emotion segments...")

    segments = build_segments(
        smoothed,
        fps=args.fps,
    )

    report = build_report(
        original=original,
        smoothed=smoothed,
        segments=segments,
        fps=args.fps,
        smoothing_window=args.window,
        min_run=args.min_run,
    )

    temporal_csv = (
        args.output / "temporal_emotions.csv"
    )

    report_json = (
        args.output / "temporal_report.json"
    )

    write_temporal_csv(
        smoothed,
        temporal_csv,
        fps=args.fps,
    )

    with report_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("=" * 72)

    print()
    print("dominant emotion:")
    print(
        f"  {report['dominant_emotion']}"
        f" ({report['emotion_distribution_percent'][report['dominant_emotion']]:.2f}%)"
    )

    print()
    print("emotion distribution:")

    for emotion in EMOTIONS:
        percentage = report[
            "emotion_distribution_percent"
        ][emotion]

        print(
            f"  {emotion:<10}: "
            f"{percentage:.2f}%"
        )

    print()
    print("temporal confidence:")

    confidence = report["confidence"]

    print(f"  min:  {confidence['min']:.4f}")
    print(f"  mean: {confidence['mean']:.4f}")
    print(f"  max:  {confidence['max']:.4f}")

    print()
    print(f"segments: {len(segments)}")
    print(
        f"transitions: "
        f"{report['transition_count']}"
    )

    print()
    print("artifacts:")
    print(f"  CSV:  {temporal_csv}")
    print(f"  JSON: {report_json}")


if __name__ == "__main__":
    main()