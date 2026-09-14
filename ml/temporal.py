"""Temporal aggregation: gap-aware smoothing, segments, transitions, confidence.

Aggregation is done in PROBABILITY space, never by counting argmax labels. A
frame that is 45/40 neutral/sadness contributes its genuine ambiguity instead of
casting a full vote.

Smoothing is gap-aware: samples with no usable face break the sequence into
runs, and the moving average never spans a gap longer than `max_gap`. Averaging
across a three-second hole would silently invent continuity that is not there.
"""

from __future__ import annotations

import math

import numpy as np


def contiguous_runs(sample_positions: list[int], max_gap: int) -> list[list[int]]:
    """Split indices into runs whose neighbours are within `max_gap` samples."""
    runs: list[list[int]] = []
    current: list[int] = []
    prev = None
    for idx, pos in enumerate(sample_positions):
        if prev is not None and (pos - prev) > max_gap + 1:
            runs.append(current)
            current = []
        current.append(idx)
        prev = pos
    if current:
        runs.append(current)
    return runs


def smooth_probabilities(probs: np.ndarray, sample_positions: list[int],
                         window: int, max_gap: int) -> np.ndarray:
    """Centred moving average over probability vectors, within runs only."""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number")
    n = len(probs)
    if n == 0:
        return probs.copy()
    half = window // 2
    out = np.empty_like(probs)
    for run in contiguous_runs(sample_positions, max_gap):
        for local, i in enumerate(run):
            lo = max(0, local - half)
            hi = min(len(run), local + half + 1)
            block = probs[[run[j] for j in range(lo, hi)]]
            avg = block.mean(axis=0)
            total = avg.sum()
            out[i] = avg / total if total > 0 else avg
    return out


def remove_short_runs(labels: list[int], min_run: int) -> list[int]:
    """Replace runs shorter than `min_run` with the neighbouring label."""
    if min_run <= 1 or not labels:
        return list(labels)
    out = list(labels)
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and out[j + 1] == out[i]:
            j += 1
        length = j - i + 1
        if length < min_run:
            replacement = out[i - 1] if i > 0 else (
                out[j + 1] if j + 1 < len(out) else out[i])
            for k in range(i, j + 1):
                out[k] = replacement
        i = j + 1
    return out


def build_segments(labels: list[int], timestamps: list[float],
                   frame_numbers: list[int], class_names: list[str],
                   confidences: list[float]) -> list[dict]:
    segments: list[dict] = []
    if not labels:
        return segments
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            end = i - 1
            segments.append({
                "emotion": class_names[labels[start]],
                "start_time_s": round(float(timestamps[start]), 3),
                "end_time_s": round(float(timestamps[end]), 3),
                "duration_s": round(float(timestamps[end] - timestamps[start]), 3),
                "start_frame": int(frame_numbers[start]),
                "end_frame": int(frame_numbers[end]),
                "sample_count": end - start + 1,
                "mean_confidence": round(
                    float(np.mean(confidences[start:end + 1])), 4),
            })
            start = i
    return segments


def build_transitions(segments: list[dict]) -> list[dict]:
    return [{
        "from": segments[i]["emotion"],
        "to": segments[i + 1]["emotion"],
        "time_s": segments[i + 1]["start_time_s"],
        "frame": segments[i + 1]["start_frame"],
    } for i in range(len(segments) - 1)]


def compute_confidence(smoothed: np.ndarray, labels: list[int],
                       dominant_index: int, usable: int, sampled: int) -> dict:
    """Composite confidence in the RESULT - explicitly not model accuracy.

    Frames are temporally correlated, so a binomial interval over frame counts
    would be badly overstated. These four observable components are reported
    individually and reduced to a coarse band rather than a false precision.
    """
    if len(smoothed) == 0:
        return {"level": "none", "score": None, "components": {
            "margin": None, "stability": None, "coverage": 0.0,
            "mean_entropy_normalised": None},
            "note": "No usable frames; confidence is undefined."}

    ordered = np.sort(smoothed, axis=1)
    margin = float(np.mean(ordered[:, -1] - ordered[:, -2]))
    stability = float(np.mean([lab == dominant_index for lab in labels]))
    coverage = float(usable / sampled) if sampled else 0.0
    eps = 1e-12
    entropy = -np.sum(smoothed * np.log(smoothed + eps), axis=1)
    norm_entropy = float(np.mean(entropy) / math.log(smoothed.shape[1]))

    score = float(np.clip(
        0.35 * stability + 0.25 * coverage + 0.25 * (1.0 - norm_entropy)
        + 0.15 * margin, 0.0, 1.0))
    level = "high" if score >= 0.70 else ("medium" if score >= 0.45 else "low")

    return {
        "level": level,
        "score": round(score, 4),
        "components": {
            "margin": round(margin, 4),
            "stability": round(stability, 4),
            "coverage": round(coverage, 4),
            "mean_entropy_normalised": round(norm_entropy, 4),
        },
        "note": ("Confidence describes agreement and coverage within THIS video. "
                 "It is not the model's accuracy and is not a probability that "
                 "the dominant emotion is correct."),
    }
