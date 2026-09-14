"""
Phase 6D: can the SERVING code actually run the new checkpoint?

A checkpoint that scores well offline is worthless if the application cannot
load it, so this script exercises the real production classes rather than a
copy of them:

    ml.model.EmotionClassifier      - the loader the API builds at startup
    ml.pipeline.VideoEmotionPipeline.analyze        - the upload path
    ml.pipeline.VideoEmotionPipeline.analyze_frame  - the live camera path

Run it against any checkpoint:

    python verify_checkpoint.py --checkpoint runs/<run>/best_model.pt \\
                               --video frontend/public/__run_clip.mp4

Nothing here writes to the application's configuration. Switching the served
model is a separate, deliberate edit to configuration/settings.py.
"""

import argparse
import dataclasses
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configuration.settings import SETTINGS            # noqa: E402
from ml.model import EmotionClassifier                 # noqa: E402
from ml.pipeline import VideoEmotionPipeline           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video", default=os.path.join("frontend", "public",
                                                    "__run_clip.mp4"))
    ap.add_argument("--frames", type=int, default=6,
                    help="live-path frames to sample from the video")
    args = ap.parse_args()

    ckpt = os.path.join(ROOT, args.checkpoint) if not os.path.isabs(args.checkpoint) \
        else args.checkpoint

    # ---- 1. the production loader ----------------------------------------
    clf = EmotionClassifier(ckpt)
    info = clf.info
    print("loaded through ml.model.EmotionClassifier")
    print(f"  architecture   {info.architecture}")
    print(f"  classes        {info.class_names}")
    print(f"  img_size       {info.img_size}   norm {info.norm_mean:.4f}/{info.norm_std:.4f}")
    print(f"  epoch          {info.epoch}   val macro-F1 {info.val_macro_f1}")
    print(f"  device         {info.device}")

    expected = ["neutral", "happiness", "surprise", "sadness", "anger",
                "disgust", "fear"]
    assert info.class_names == expected, \
        f"class order changed - inference would silently mislabel: {info.class_names}"
    assert info.img_size == 112, f"unexpected img_size {info.img_size}"
    del clf

    # ---- 2. the full upload pipeline -------------------------------------
    settings = dataclasses.replace(SETTINGS, checkpoint=ckpt)
    pipeline = VideoEmotionPipeline(settings)
    video = os.path.join(ROOT, args.video) if not os.path.isabs(args.video) else args.video
    report = pipeline.analyze(video)
    print("\nupload path (VideoEmotionPipeline.analyze)")
    print(f"  status            {report['status']}")
    print(f"  dominant_emotion  {report['dominant_emotion']}")
    print(f"  frames analysed   {report['analyzed_frames']}/{report['sampled_frames']}")
    print(f"  segments          {report['segment_count']}")
    print("  distribution      " + ", ".join(
        f"{c}={v:.3f}" for c, v in report["emotion_distribution"].items() if v > 0))

    # ---- 3. the live camera path -----------------------------------------
    cap = cv2.VideoCapture(video)
    grabbed, i = [], 0
    while len(grabbed) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % 12 == 0:
            grabbed.append(frame)
        i += 1
    cap.release()

    print(f"\nlive path (VideoEmotionPipeline.analyze_frame) on {len(grabbed)} frames")
    for n, frame in enumerate(grabbed):
        scale = 480.0 / frame.shape[1]
        small = cv2.resize(frame, (480, int(round(frame.shape[0] * scale))))
        rec = pipeline.analyze_frame(small)
        if rec["status"] == "ok":
            probs = rec["probabilities"]
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            print(f"  frame {n}: {rec['predicted_emotion']:<10} "
                  f"conf {rec['confidence']:.3f}   " +
                  "  ".join(f"{c}={v:.2f}" for c, v in top))
        else:
            print(f"  frame {n}: status={rec['status']} (no prediction, as designed)")

    # a face-free frame must still refuse to predict
    blank = np.full((360, 480, 3), 40, np.uint8)
    rec = pipeline.analyze_frame(blank)
    assert rec["predicted_emotion"] is None and rec["probabilities"] is None, \
        "the pipeline invented a prediction for a frame with no face"
    print(f"  blank frame: status={rec['status']}, no prediction  [ok]")

    pipeline.detector.close()
    print("\nVERIFIED: this checkpoint is loadable and runnable by the live "
          "camera and upload paths unchanged.")


if __name__ == "__main__":
    main()
