"""Validate that this machine can run the analysis backend.

Checks imports, model files and device availability. Read-only: it loads
nothing into GPU memory and modifies no project file.

    .venv\\Scripts\\python.exe verify_setup.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

OK = "  [ OK ]"
BAD = "  [FAIL]"
WARN = "  [WARN]"

problems: list[str] = []
warnings: list[str] = []


def check_python() -> None:
    print("Python")
    v = sys.version_info
    print(f"    version           {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) < (3, 10):
        problems.append(f"Python {v.major}.{v.minor} is too old; use 3.10-3.12.")
        print(BAD, "need Python 3.10 or newer")
    else:
        print(OK)


def check_imports() -> None:
    print("\nPackages")
    for module, label in [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("multipart", "python-multipart"),
        ("cv2", "opencv-contrib-python"),
        ("numpy", "numpy"),
        ("mediapipe", "mediapipe"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
    ]:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installed")
            print(f"    {label:24s} {version}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"Cannot import {label}: {exc}")
            print(BAD, f"{label} missing ({exc})")


def check_device() -> None:
    print("\nCompute device")
    try:
        import torch
    except Exception:
        print(WARN, "torch not importable; skipping device check")
        return

    if torch.cuda.is_available():
        print(f"    CUDA available    yes ({torch.cuda.get_device_name(0)})")
        print(f"    torch build       {torch.__version__}")
        print(OK, "GPU will be used")
    else:
        print(f"    CUDA available    no")
        print(f"    torch build       {torch.__version__}")
        print(OK, "CPU will be used (slower, fully supported)")


def check_models() -> None:
    print("\nModel files")
    required = [
        (os.path.join(ROOT, "models", "blaze_face_short_range.tflite"),
         "face detector (MediaPipe BlazeFace)"),
        (os.path.join(ROOT, "runs", "baseline_resnet18", "best_model.pt"),
         "expression classifier checkpoint"),
    ]
    for path, label in required:
        rel = os.path.relpath(path, ROOT)
        if os.path.isfile(path):
            mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    {rel}")
            print(f"      {label} - {mb:.1f} MB")
            print(OK)
        else:
            problems.append(f"Missing model file: {rel} ({label})")
            print(BAD, f"missing {rel}")


def check_settings() -> None:
    print("\nProject paths")
    sys.path.insert(0, ROOT)
    try:
        from configuration.settings import SETTINGS
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Cannot load configuration.settings: {exc}")
        print(BAD, f"cannot import settings ({exc})")
        return

    print(f"    root              {SETTINGS.root}")
    if os.path.normcase(SETTINGS.root) != os.path.normcase(ROOT):
        warnings.append("Settings root does not match this folder.")
        print(WARN, "root mismatch")
    else:
        print(OK, "paths resolve relative to this folder")


def main() -> int:
    print("=" * 62)
    print(" Facial Expression Recognition - environment check")
    print("=" * 62)

    check_python()
    check_imports()
    check_device()
    check_models()
    check_settings()

    print("\n" + "=" * 62)
    if problems:
        print(f" {len(problems)} problem(s) found:\n")
        for p in problems:
            print(f"   - {p}")
        print("\n See SETUP_WINDOWS.md for fixes.")
        print("=" * 62)
        return 1

    for w in warnings:
        print(f" warning: {w}")
    print(" All checks passed. This machine can run the project.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
