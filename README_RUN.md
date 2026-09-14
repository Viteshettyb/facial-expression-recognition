# Running the application

## Backend (FastAPI)
```
.venv/Scripts/python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- `GET  /api/health`     model + device status
- `POST /analyze-video`  synchronous analysis, returns the full report
- `POST /api/jobs`       asynchronous analysis (202 + job_id), for long videos
- `GET  /api/jobs/{id}`  progress
- `GET  /api/jobs/{id}/result`
- Docs: http://127.0.0.1:8000/docs

## Frontend (React + Vite + TypeScript)
```
cd frontend && npm run dev
```
Opens http://localhost:5173. Override the API base with `VITE_API_URL`.

## Layout
```
configuration/  settings (all production constants)
ml/             detector, preprocessing, model, temporal, pipeline
models/         blaze_face_short_range.tflite
utilities/      JSON safety helpers
backend/app/    FastAPI application
frontend/       React UI
runs/           frozen training + Phase 4 artifacts (unchanged)
```
