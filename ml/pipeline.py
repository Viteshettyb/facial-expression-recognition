"""End-to-end video -> emotion report, in one process, no file round-trips."""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configuration.settings import SETTINGS, Settings          # noqa: E402
from ml.detector import FaceDetector                            # noqa: E402
from ml.model import EmotionClassifier                          # noqa: E402
from ml.preprocessing import square_face_crop, to_fer48         # noqa: E402
from ml.temporal import (build_segments, build_transitions,     # noqa: E402
                         compute_confidence, remove_short_runs,
                         smooth_probabilities)
from utilities.safe_json import clean, safe_number              # noqa: E402


def probe_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    if info["fps"] <= 0 or info["frame_count"] <= 0:
        raise ValueError("Video reports invalid fps or frame count")
    info["duration_s"] = info["frame_count"] / info["fps"]
    return info


class VideoEmotionPipeline:
    """Built once at startup; `analyze` is called per request."""

    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings
        self.classifier = EmotionClassifier(settings.checkpoint)
        self.detector = FaceDetector(settings.detector_model,
                                     settings.min_detection_confidence)
        self.class_names = self.classifier.class_names

    # ---------------------------------------------------------------- stage 1
    def _sample_and_crop(self, path: str, video: dict, target_fps: float,
                         progress=None):
        step = video["fps"] / target_fps
        wanted = sorted({min(int(round(i * step)), video["frame_count"] - 1)
                         for i in range(int(video["duration_s"] * target_fps) + 1)})
        wanted_set = set(wanted)

        cap = cv2.VideoCapture(path)
        frames, faces, index = [], [], 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index in wanted_set:
                    rec = self._process_frame(frame, index, video)
                    if rec["status"] == "ok":
                        faces.append(rec.pop("_face48"))
                    else:
                        rec.pop("_face48", None)
                    frames.append(rec)
                    if progress and len(frames) % 10 == 0:
                        progress(len(frames), len(wanted))
                index += 1
        finally:
            cap.release()
        return frames, faces, len(wanted)

    def _process_frame(self, frame, index: int, video: dict) -> dict:
        s = self.settings
        rec = {"sample_position": None, "frame_number": int(index),
               "timestamp_s": round(index / video["fps"], 3), "status": "no_face",
               "face_count": 0, "detector_score": None, "box": None,
               "face_area_fraction": None, "crop_shifted": False,
               "crop_padded": False, "_face48": None}

        detections = self.detector.detect(frame)
        rec["face_count"] = len(detections)
        if not detections:
            return rec

        best = self.detector.largest(detections)
        rec["detector_score"] = round(best.score, 4)
        rec["box"] = [int(best.x), int(best.y), int(best.w), int(best.h)]
        rec["face_area_fraction"] = round(
            best.area / (video["width"] * video["height"]), 6)

        if best.short_side < s.min_face_px:
            rec["status"] = "face_too_small"
            return rec

        crop = square_face_crop(frame, best.x, best.y, best.w, best.h,
                                s.crop_margin, s.crop_upward_offset)
        rec["status"] = crop.status
        rec["crop_shifted"] = crop.shifted
        rec["crop_padded"] = crop.padded
        if crop.status != "ok" or crop.image is None:
            return rec
        rec["_face48"] = to_fer48(crop.image)
        return rec

    # ------------------------------------------------------------ live frame
    def analyze_frame(self, frame_bgr: np.ndarray) -> dict:
        """One live camera frame -> one prediction.

        Deliberately the same detector, the same calibrated square crop, the
        same 48x48 FER bottleneck and the same classifier as the video path;
        only the frame source differs. No temporal state is kept here - the
        caller owns smoothing, so the endpoint stays stateless and cheap.
        """
        s = self.settings
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("empty frame")
        h, w = frame_bgr.shape[:2]

        rec = {
            "status": "no_face",
            "face_count": 0,
            "detector_score": None,
            "box": None,
            "frame_width": int(w),
            "frame_height": int(h),
            "face_area_fraction": None,
            "probabilities": None,
            "predicted_emotion": None,
            "confidence": None,
        }

        detections = self.detector.detect(frame_bgr)
        rec["face_count"] = len(detections)
        if not detections:
            return rec

        best = self.detector.largest(detections)          # largest face wins
        rec["detector_score"] = round(best.score, 4)
        rec["box"] = [int(best.x), int(best.y), int(best.w), int(best.h)]
        rec["face_area_fraction"] = round(best.area / float(w * h), 6)

        if best.short_side < s.min_face_px:
            rec["status"] = "face_too_small"
            return rec

        crop = square_face_crop(frame_bgr, best.x, best.y, best.w, best.h,
                                s.crop_margin, s.crop_upward_offset)
        rec["status"] = crop.status
        if crop.status != "ok" or crop.image is None:
            return rec

        p = self.classifier.predict([to_fer48(crop.image)])[0]
        rec["probabilities"] = {
            c: round(float(p[j]), 6) for j, c in enumerate(self.class_names)}
        rec["predicted_emotion"] = self.class_names[int(p.argmax())]
        rec["confidence"] = round(float(p.max()), 6)
        return rec

    # ---------------------------------------------------------------- analyse
    def analyze(self, video_path: str, target_fps: float | None = None,
                progress=None) -> dict:
        s = self.settings
        target_fps = float(target_fps or s.target_fps)
        started = time.time()

        video = probe_video(video_path)
        frames, faces, n_sampled = self._sample_and_crop(
            video_path, video, target_fps, progress)

        # ------------------------------------------------------- inference
        probs = self.classifier.predict(faces, batch_size=s.batch_size)

        ok_idx = [i for i, f in enumerate(frames) if f["status"] == "ok"]
        for slot, i in enumerate(ok_idx):
            p = probs[slot]
            frames[i]["sample_position"] = slot
            frames[i]["probabilities"] = {
                c: round(float(p[j]), 6) for j, c in enumerate(self.class_names)}
            frames[i]["predicted_emotion"] = self.class_names[int(p.argmax())]
            frames[i]["confidence"] = round(float(p.max()), 6)
        for i, f in enumerate(frames):
            if f["status"] != "ok":
                # Rule 4: never invent a prediction for a frame with no face.
                f["probabilities"] = None
                f["predicted_emotion"] = None
                f["confidence"] = None

        n_ok = len(ok_idx)
        quality = {
            "sampled_frames": n_sampled,
            "analyzed_frames": n_ok,
            "frames_with_face": sum(1 for f in frames if f["face_count"] > 0),
            "frames_no_face": sum(1 for f in frames if f["status"] == "no_face"),
            "frames_face_too_small": sum(
                1 for f in frames if f["status"] == "face_too_small"),
            "frames_rejected": sum(
                1 for f in frames if f["status"] not in ("ok", "no_face")),
            "frames_multi_face": sum(1 for f in frames if f["face_count"] > 1),
            "crops_shifted": sum(1 for f in frames if f["crop_shifted"]),
            "crops_padded": sum(1 for f in frames if f["crop_padded"]),
            "detection_rate": round(
                sum(1 for f in frames if f["face_count"] > 0) / n_sampled, 4)
            if n_sampled else 0.0,
            "usable_rate": round(n_ok / n_sampled, 4) if n_sampled else 0.0,
        }

        # No usable frame is a VALID result, not an error (rule 4).
        if n_ok == 0:
            return self._report(video, frames, quality, None, [], [], [],
                                compute_confidence(np.zeros((0, 7)), [], -1,
                                                   0, n_sampled),
                                target_fps, time.time() - started,
                                warning="No usable face was detected in any "
                                        "sampled frame; no emotion was inferred.")

        # ------------------------------------------------------- temporal
        positions = [frames[i]["frame_number"] for i in ok_idx]
        sample_slots = list(range(len(ok_idx)))
        gap_units = [round(p / (video["fps"] / target_fps)) for p in positions]

        smoothed = smooth_probabilities(probs, gap_units, s.smoothing_window,
                                        s.max_gap_samples)
        raw_labels = [int(v.argmax()) for v in smoothed]
        labels = remove_short_runs(raw_labels, s.min_run)
        conf_series = [float(smoothed[i][labels[i]]) for i in range(len(labels))]

        for slot, i in enumerate(ok_idx):
            frames[i]["smoothed_probabilities"] = {
                c: round(float(smoothed[slot][j]), 6)
                for j, c in enumerate(self.class_names)}
            frames[i]["smoothed_emotion"] = self.class_names[labels[slot]]
            frames[i]["smoothed_confidence"] = round(conf_series[slot], 6)

        mean_probs = smoothed.mean(axis=0)
        mean_probs = mean_probs / mean_probs.sum()
        dominant_index = int(mean_probs.argmax())
        dominant = self.class_names[dominant_index]

        distribution = {c: round(float(mean_probs[j]) * 100.0, 4)
                        for j, c in enumerate(self.class_names)}
        label_distribution = {
            c: round(100.0 * sum(1 for l in labels if l == j) / len(labels), 4)
            for j, c in enumerate(self.class_names)}

        timestamps = [frames[i]["timestamp_s"] for i in ok_idx]
        segments = build_segments(labels, timestamps, positions,
                                  self.class_names, conf_series)
        transitions = build_transitions(segments)
        confidence = compute_confidence(smoothed, labels, dominant_index,
                                        n_ok, n_sampled)

        warning = None
        if quality["usable_rate"] < 0.5:
            warning = (f"Only {quality['usable_rate'] * 100:.0f}% of sampled "
                       f"frames contained a usable face; treat the result with "
                       f"caution.")

        return self._report(video, frames, quality, dominant, segments,
                            transitions, list(sample_slots), confidence,
                            target_fps, time.time() - started,
                            distribution=distribution,
                            label_distribution=label_distribution,
                            warning=warning)

    # ---------------------------------------------------------------- report
    def _report(self, video, frames, quality, dominant, segments, transitions,
                slots, confidence, target_fps, elapsed, distribution=None,
                label_distribution=None, warning=None) -> dict:
        s = self.settings
        info = self.classifier.info
        empty = {c: 0.0 for c in self.class_names}
        report = {
            "status": "ok",
            "dominant_emotion": dominant,
            "emotion_distribution": distribution if distribution else empty,
            "temporal_label_distribution": (
                label_distribution if label_distribution else empty),
            "confidence": confidence,
            "frame_count": video["frame_count"],
            "sampled_frames": quality["sampled_frames"],
            "analyzed_frames": quality["analyzed_frames"],
            "quality": quality,
            "temporal_segments": segments,
            "segment_count": len(segments),
            "transitions": transitions,
            "transition_count": len(transitions),
            "video": {
                "width": video["width"], "height": video["height"],
                "fps": round(video["fps"], 6),
                "frame_count": video["frame_count"],
                "duration_s": round(video["duration_s"], 3),
            },
            "model": {
                "architecture": "ResNet18",
                "backbone": info.architecture,
                "checkpoint": os.path.relpath(info.checkpoint, s.root),
                "checkpoint_epoch": info.epoch,
                "validation_macro_f1": safe_number(info.val_macro_f1),
                "validation_accuracy": safe_number(info.val_accuracy),
                "validation_balanced_accuracy": safe_number(
                    info.val_balanced_accuracy),
                "class_names": self.class_names,
                "input_size": info.img_size,
                "norm_mean": info.norm_mean,
                "norm_std": info.norm_std,
                "device": info.device,
            },
            "analysis": {
                "sampling_fps": target_fps,
                "detector": "MediaPipe Tasks BlazeFace (short range)",
                "min_detection_confidence": s.min_detection_confidence,
                "min_face_px": s.min_face_px,
                "crop_margin": s.crop_margin,
                "crop_upward_offset": s.crop_upward_offset,
                "crop_calibration": "FER+ TRAIN framing (Phase 4A.1)",
                "preprocessing": "BT.601 grayscale -> 48x48 -> resize "
                                 f"{info.img_size} -> normalise",
                "smoothing_window_samples": s.smoothing_window,
                # samples / SAMPLING fps, not video fps
                "smoothing_window_seconds": round(
                    s.smoothing_window / target_fps, 4),
                "min_run_samples": s.min_run,
                "max_gap_samples": s.max_gap_samples,
                "aggregation": "mean of temporally smoothed probability vectors",
                "processing_time_s": round(elapsed, 3),
            },
            "warning": warning,
            "frames": frames,
        }
        return clean(report)
