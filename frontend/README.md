# Emotia — Frontend

Frontend for **Video-Based Facial Expression Recognition and Emotion Analysis Using Deep Learning**.

React + TypeScript + Vite + Tailwind v4.

## Run

```
npm install
npm run dev
```

## Flow

`UPLOAD → PROCESSING → ANALYSIS → TEMPORAL RESULTS → FINAL REPORT`, one section per stage:

| Section | File |
| --- | --- |
| Hero | `src/components/flow/Hero.tsx` |
| Upload + status | `src/components/flow/UploadSection.tsx` |
| Processing progress | `src/components/flow/ProcessingSection.tsx` |
| Primary face detection | `src/components/flow/FaceSection.tsx` |
| Predictions, dominant emotion, distribution, confidence | `src/components/flow/ResultsSection.tsx` |
| Temporal timeline + segments | `src/components/flow/TimelineSection.tsx` |
| Frame-level analysis | `src/components/flow/FramesSection.tsx` |
| Final report | `src/components/flow/ReportSection.tsx` |

State for the whole flow lives in `src/hooks/useAnalysisFlow.ts`.

## Backend connection

The UI is wired to the real FastAPI pipeline. There is no mock data.

Start the backend first, from the project root:

```
.venv/Scripts/python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Endpoints used (see `backend/app/main.py`):

| Step | Call |
| --- | --- |
| Upload | `POST /api/jobs` → 202 `{ job_id }` |
| Progress | `GET /api/jobs/{id}` (polled) |
| Result | `GET /api/jobs/{id}/result` |

Override the base URL with `VITE_API_URL` (defaults to `http://127.0.0.1:8000`).

All network access is isolated in `src/api/client.ts`. `src/api/backendTypes.ts` is
the wire contract, and `src/api/mapReport.ts` is the only place the report is
translated into the UI model — the sole unit conversion applied to emotion values
is the backend's 0–100 percentages divided by 100.

Emotion classes come from the checkpoint and must not be renamed:

```
neutral, happiness, surprise, sadness, anger, disgust, fear
```

## Design system

Tokens live in `src/index.css` (`:root` + `@theme inline`): fluid `--text-*` scale,
`--width-content-width` container, glass `.card`, `.primary-button` / `.secondary-button`,
and the `--emotion-*` palette. Reusable primitives are in `src/components/ui/primitives.tsx`.
