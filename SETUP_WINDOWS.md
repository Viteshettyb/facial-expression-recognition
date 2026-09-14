# Setup on Windows

Video-based facial expression recognition and emotion analysis.
FastAPI + PyTorch backend, React + Vite frontend.

---

## Prerequisites

Install these two first. Both are free and only take a few minutes.
Everything else is handled by the setup script.

| Software | Version | Download | Important |
| --- | --- | --- | --- |
| **Python** | **3.12** (3.10 / 3.11 also fine) | https://www.python.org/downloads/ | Tick **"Add python.exe to PATH"** during install |
| **Node.js** | **20 LTS or newer** | https://nodejs.org/ | Take the LTS build, accept the defaults |

### Is an NVIDIA GPU required?

**No. A GPU is optional and CUDA is optional.**

The project runs fully on CPU with **identical accuracy** — only speed differs.
The setup script detects what you have and installs the right build
automatically. You do not need to install CUDA or any NVIDIA toolkit yourself.

| | NVIDIA GPU | CPU only |
| --- | --- | --- |
| Works | Yes | Yes |
| Accuracy | Identical | Identical |
| Speed | ~1s for a 12s clip | Roughly 5–15× slower |
| Download size | ~3 GB | ~250 MB |

Check your prerequisites are visible by opening a **new** Command Prompt:

```bat
python --version
node --version
```

If either says "not recognized", it is not on your PATH — reinstall with the
"Add to PATH" option ticked, then open a new window.

---

## First time

1. Copy / unzip the project folder anywhere (any drive, any folder name).
2. **Double-click `setup.bat`** in the project root.
3. Wait until it prints **SETUP COMPLETE**. Allow 5–15 minutes the first time,
   mostly PyTorch downloading.
4. Open the project folder in VS Code.

`setup.bat` does all of this for you:

- finds your Python installation
- creates the `.venv` virtual environment
- installs the correct PyTorch build (CUDA if you have an NVIDIA GPU, otherwise CPU)
- installs the backend packages from `requirements.txt`
- installs the frontend npm packages
- checks both ML model files are present
- validates the whole environment and reports any problem

It is **safe to run again** at any time. It reuses the existing `.venv`, skips
PyTorch if already installed, and skips `npm install` unless
`package-lock.json` has changed. A repeat run takes about 10 seconds.

To force the smaller CPU-only build, run it from a terminal instead of
double-clicking:

```bat
setup.bat cpu
```

---

## Running the project after setup

Open the project folder in VS Code and use **two terminals**.

### Terminal 1 — backend

From the **project root**:

```bat
.venv\Scripts\activate
python -m uvicorn backend.app.main:app --reload --port 8000
```

Wait for `Application startup complete.` — the model takes a few seconds to
load on first start.

> If you prefer not to activate the environment, this single line is equivalent:
> ```bat
> .venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
> ```
> In VS Code you can also press `Ctrl+Shift+P` → *Python: Select Interpreter* →
> choose the one inside `.venv`, and new terminals will use it automatically.

### Terminal 2 — frontend

```bat
cd frontend
npm run dev
```

### URLs

| What | URL |
| --- | --- |
| **Application** | **http://localhost:5173** |
| Backend API docs | http://127.0.0.1:8000/docs |

Start the backend first, then the frontend, then open
**http://localhost:5173**.

---

## What to copy to another PC

Copy the whole folder, but you can safely **skip** these — they are large and
are recreated by `setup.bat`:

| Skip | Why |
| --- | --- |
| `.venv/` | Recreated by setup |
| `frontend/node_modules/` | Recreated by setup |
| `__pycache__/` folders | Python cache |
| `artifacts/` | Created at runtime |
| `data/`, `processed/` | Training data only (~410 MB), unused at runtime |
| `runs/expB_*`, `runs/expC_*`, `runs/expD_*`, `runs/phase4*` | Training experiments (~175 MB) |

**These two model files must be kept**, at exactly these paths — the app will
not start without them:

```
models\blaze_face_short_range.tflite          (0.2 MB)  face detector
runs\baseline_resnet18\best_model.pt          (43 MB)   expression classifier
```

With those skips the copy is about **44 MB** instead of ~590 MB.
`.gitignore` already lists all of the above.

---

## Checking the installation

```bat
.venv\Scripts\python.exe verify_setup.py
```

Reports your Python version, every required package, whether GPU or CPU will be
used, and whether both model files are present.

---

## Common errors and fixes

### "python is not recognized" / "node is not recognized"

Not on your PATH. Reinstall with "Add to PATH" ticked, then open a **new**
terminal window.

### "Port 5173 is already in use"

The frontend is deliberately locked to port 5173, because the backend only
accepts API calls from that exact origin. Free it:

```bat
netstat -ano | findstr :5173
taskkill /PID <number from the last column> /F
```

### "Could not reach the analysis service" in the browser

The backend is not running, or not on port 8000. Check the backend terminal
shows `Application startup complete.`, and that
http://127.0.0.1:8000/docs opens.

Always use **http://localhost:5173** — on any other port the browser blocks
the request even when the backend is fine.

### "Missing model file" from setup

`models\blaze_face_short_range.tflite` or `runs\baseline_resnet18\best_model.pt`
was not copied. Copy them from the original PC, keeping those exact paths.

### PyTorch download fails or is very slow

It is a multi-GB download on GPU machines. Use the much smaller CPU build:

```bat
setup.bat cpu
```

### `npm install` fails

Delete `frontend\node_modules`, then run `setup.bat` again.

### Analysis says "No usable face was detected"

That is a valid result, not an error — no face was found large or clear enough.
Use a video with a single, well-lit face facing the camera. Faces under 48 px
are ignored by design.

---

## Configuration

No configuration is normally needed. The frontend defaults to
`http://127.0.0.1:8000`.

To point it elsewhere, create `frontend\.env`:

```
VITE_API_URL=http://127.0.0.1:8000
```

Backend constants (sampling rate, detector thresholds, smoothing, upload
limits) are in `configuration\settings.py`. All paths there are derived
relative to the project folder, so the project works from any location on any
drive.

There are no secrets, API keys or accounts. Everything runs locally.
