import type { AnalysisResult, VideoMeta } from '../types/analysis';
import type {
  WireHealth,
  WireJobCreated,
  WireJobStatus,
  WireLiveFrame,
  WireReport,
} from './backendTypes';
import { mapReport } from './mapReport';

/**
 * ---------------------------------------------------------------------------
 * API BOUNDARY
 * ---------------------------------------------------------------------------
 * Every network call the UI makes lives here. Nothing else in the app performs
 * a request, and `mapReport.ts` is the only place the wire format is
 * translated into the UI model.
 *
 * Flow used (matches `backend/app/main.py`):
 *   POST /api/jobs             -> 202 { job_id }                (streamed upload)
 *   GET  /api/jobs/{id}        -> { status, progress, stage }   (polled)
 *   GET  /api/jobs/{id}/result -> the full analysis report
 *
 * The synchronous `POST /analyze-video` is also exposed below for callers that
 * want a single round trip with no progress reporting.
 * ---------------------------------------------------------------------------
 */

const API_BASE_URL = (
  import.meta.env.VITE_API_URL ??
  import.meta.env.VITE_API_BASE_URL ??
  'http://127.0.0.1:8000'
).replace(/\/$/, '');

export const apiConfig = { baseUrl: API_BASE_URL };

export interface UploadHandle {
  jobId: string;
}

export type ProgressCallback = (percent: number) => void;

const wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** FastAPI returns `{ detail: ... }` on error; surface that text when present. */
async function errorText(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === 'string') return detail;
    if (detail) return JSON.stringify(detail);
  } catch {
    /* not JSON — fall through */
  }
  return `${fallback} (HTTP ${res.status})`;
}

/** Model / device status. Used to show whether the real backend is reachable. */
export async function getHealth(signal?: AbortSignal): Promise<WireHealth> {
  const res = await fetch(`${API_BASE_URL}/api/health`, { signal });
  if (!res.ok) throw new Error(await errorText(res, 'Health check failed'));
  return (await res.json()) as WireHealth;
}

/**
 * Step 1 — stream the video to the analysis service and open a job.
 * Uses XMLHttpRequest rather than fetch because only XHR reports real upload
 * progress; the percentage shown is the actual number of bytes sent.
 */
export function uploadVideo(
  file: File,
  onProgress?: ProgressCallback,
  signal?: AbortSignal,
): Promise<UploadHandle> {
  return new Promise<UploadHandle>((resolve, reject) => {
    const body = new FormData();
    body.append('file', file);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}/api/jobs`);
    xhr.responseType = 'text';

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(Math.round((e.loaded / e.total) * 100));
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText) as WireJobCreated;
          onProgress?.(100);
          resolve({ jobId: data.job_id });
        } catch {
          reject(new Error('Malformed response from the analysis service.'));
        }
        return;
      }
      let detail = `Upload failed (HTTP ${xhr.status})`;
      try {
        const parsed = JSON.parse(xhr.responseText) as { detail?: string };
        if (parsed.detail) detail = parsed.detail;
      } catch {
        /* keep the status-based message */
      }
      reject(new Error(detail));
    };

    xhr.onerror = () =>
      reject(
        new Error(
          `Could not reach the analysis service at ${API_BASE_URL}. Is the FastAPI server running?`,
        ),
      );
    xhr.ontimeout = () => reject(new Error('The upload timed out.'));

    signal?.addEventListener('abort', () => xhr.abort(), { once: true });
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'));

    xhr.send(body);
  });
}

export interface AnalysisProgress {
  percent: number;
  stepId: string;
  message: string;
}

/**
 * Map the backend's single 0–1 progress value onto the pipeline stages the UI
 * lists. The percentage is the backend's own; only the stage label is inferred
 * from it, because the backend reports one scalar rather than a named stage.
 */
function describeProgress(fraction: number, stage: string | null): AnalysisProgress {
  const percent = Math.max(0, Math.min(100, Math.round(fraction * 100)));
  let stepId = 'decode';
  let message = 'Decoding video and sampling frames';

  if (percent >= 95) {
    stepId = 'report';
    message = 'Compiling the emotion report';
  } else if (percent >= 88) {
    stepId = 'temporal';
    message = 'Smoothing predictions over time';
  } else if (percent >= 30) {
    stepId = 'infer';
    message = 'Running expression inference across frames';
  } else if (percent >= 15) {
    stepId = 'align';
    message = 'Cropping and normalising face crops';
  } else if (percent >= 6) {
    stepId = 'detect';
    message = 'Detecting faces and selecting the largest';
  }

  if (stage === 'done') {
    stepId = 'report';
    message = 'Analysis complete';
  }
  return { percent, stepId, message };
}

const POLL_INTERVAL_MS = 700;

/** Step 2 — poll the job until it finishes, then fetch and map the real report. */
export async function runAnalysis(
  handle: UploadHandle,
  video: VideoMeta,
  onProgress?: (p: AnalysisProgress) => void,
  signal?: AbortSignal,
): Promise<AnalysisResult> {
  for (;;) {
    if (signal?.aborted) throw new DOMException('Analysis cancelled', 'AbortError');

    const res = await fetch(`${API_BASE_URL}/api/jobs/${handle.jobId}`, { signal });
    if (!res.ok) throw new Error(await errorText(res, 'Could not read job status'));
    const status = (await res.json()) as WireJobStatus;

    if (status.status === 'failed') {
      throw new Error(status.error || 'Analysis failed on the server.');
    }

    onProgress?.(describeProgress(status.progress, status.stage));

    if (status.status === 'completed') break;
    await wait(POLL_INTERVAL_MS);
  }

  const resultRes = await fetch(`${API_BASE_URL}/api/jobs/${handle.jobId}/result`, { signal });
  if (!resultRes.ok) throw new Error(await errorText(resultRes, 'Could not load the report'));

  const report = (await resultRes.json()) as WireReport;
  onProgress?.({ percent: 100, stepId: 'report', message: 'Analysis complete' });
  return mapReport(report, video);
}

/** Single round trip, no progress. Mirrors `POST /analyze-video`. */
export async function analyzeVideoSync(
  file: File,
  video: VideoMeta,
  signal?: AbortSignal,
): Promise<AnalysisResult> {
  const body = new FormData();
  body.append('file', file);

  const res = await fetch(`${API_BASE_URL}/analyze-video`, { method: 'POST', body, signal });
  if (!res.ok) throw new Error(await errorText(res, 'Analysis failed'));
  return mapReport((await res.json()) as WireReport, video);
}

/** Fetch a previously computed report by job id. */
export async function getResult(
  jobId: string,
  video: VideoMeta,
  signal?: AbortSignal,
): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE_URL}/api/jobs/${jobId}/result`, { signal });
  if (!res.ok) throw new Error(await errorText(res, 'Could not load the report'));
  return mapReport((await res.json()) as WireReport, video);
}

/**
 * Live camera — send ONE captured frame and get its raw per-class
 * probabilities back. Deliberately one frame per call with no batching: the
 * caller keeps a single request in flight, which is what keeps the preview
 * smooth and stops the browser queueing requests it can never catch up on.
 */
export async function analyzeLiveFrame(
  frame: Blob,
  signal?: AbortSignal,
): Promise<WireLiveFrame> {
  const body = new FormData();
  body.append('frame', frame, 'frame.jpg');

  const res = await fetch(`${API_BASE_URL}/api/live/frame`, {
    method: 'POST',
    body,
    signal,
  });
  if (!res.ok) throw new Error(await errorText(res, 'Live inference failed'));
  return (await res.json()) as WireLiveFrame;
}
