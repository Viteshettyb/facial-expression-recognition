import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import { analyzeLiveFrame } from '../api/client';
import { EMOTIONS, isEmotion, type Emotion } from '../types/analysis';

/**
 * ---------------------------------------------------------------------------
 * LIVE CAMERA — capture, inference loop and temporal stability
 * ---------------------------------------------------------------------------
 * The camera stream is local. Each captured frame is sent to
 * `POST /api/live/frame`, which runs the SAME BlazeFace detector, the same
 * calibrated crop and the same ResNet-18 checkpoint as the upload pipeline.
 *
 * Three rules shape everything below.
 *
 *  1. One request in flight, ever. The next frame is captured only after the
 *     previous reply lands, so a slow backend slows the sampling rate instead
 *     of queueing work the browser can never catch up on. The preview element
 *     renders independently and stays smooth regardless.
 *
 *  2. The displayed expression changes on EVIDENCE, not on a timer. Raw
 *     per-frame probabilities are averaged over a real time window, and the
 *     label only switches once a new winner has held the averaged argmax
 *     continuously for `STABILITY_MS` at or above `MIN_CONFIDENCE`. A single
 *     noisy frame can never flip it; a genuine change of expression takes
 *     roughly one to two seconds, set by the evidence, not by a fixed delay.
 *
 *  3. No face, no prediction. When the window holds no usable face the UI is
 *     told exactly that, and the last label is dropped rather than left to
 *     linger as a stale-looking answer.
 * ---------------------------------------------------------------------------
 */

/** Averaging window for the probability vectors. */
const WINDOW_MS = 1500;
/** How long a new winner must hold the window's argmax before it is shown. */
const STABILITY_MS = 900;
/** Smoothed probability a label needs before it is allowed on screen. */
const MIN_CONFIDENCE = 0.4;
/** Independent frames a winner must be backed by, so one frame can never
 *  decide anything. Counted since the candidate appeared rather than as a
 *  fixed window size: a slow machine samples less often, and the gate must be
 *  about evidence over time, not about achieving a frame rate. */
const MIN_SAMPLES = 2;
/** Face is declared lost when nothing usable arrives for this long. */
const FACE_LOST_MS = 900;
/** Floor between captures, so a fast backend cannot spin the CPU. */
const MIN_CAPTURE_INTERVAL_MS = 110;
/** Longest edge of the JPEG sent for inference. The model sees 48x48. */
const CAPTURE_MAX_WIDTH = 480;
const JPEG_QUALITY = 0.72;

export type LiveStatus = 'idle' | 'starting' | 'running' | 'error';

export interface LiveProbability {
  emotion: Emotion;
  value: number;
}

export interface LiveCamera {
  status: LiveStatus;
  error: string | null;
  /** Attach to the <video> element that shows the preview. */
  videoRef: RefObject<HTMLVideoElement>;
  /** Stable, smoothed expression. Null while unknown or no face. */
  emotion: Emotion | null;
  /** Smoothed probability of `emotion`. Real value, never synthesised. */
  confidence: number;
  /** Full smoothed distribution, descending. Empty when there is no face. */
  distribution: LiveProbability[];
  faceDetected: boolean;
  /** Number of faces the detector saw in the last analysed frame. */
  faceCount: number;
  /** Measured round trips per second, a real throughput reading. */
  samplesPerSecond: number;
  /** Frames analysed in this session. */
  framesAnalysed: number;
  start: () => Promise<void>;
  stop: () => void;
}

interface Sample {
  t: number;
  probs: number[];
}

const ZERO: number[] = EMOTIONS.map(() => 0);

/** Wire probabilities (checkpoint class names) -> fixed EMOTIONS ordering. */
function toVector(probs: Record<string, number>): number[] | null {
  const out = EMOTIONS.map((e) => probs[e]);
  return out.every((v) => typeof v === 'number' && Number.isFinite(v)) ? out : null;
}

function describeCameraError(err: unknown): string {
  if (typeof DOMException !== 'undefined' && err instanceof DOMException) {
    switch (err.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return 'Camera access was blocked. Allow camera permission for this site in your browser, then start again.';
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        return 'No camera was found on this device.';
      case 'NotReadableError':
      case 'TrackStartError':
        return 'The camera is already in use by another application. Close it and try again.';
      case 'OverconstrainedError':
        return 'No camera matched the requested settings.';
      case 'SecurityError':
        return 'The browser blocked camera access on an insecure connection. Use https:// or localhost.';
      default:
        return `Could not start the camera (${err.name}).`;
    }
  }
  return err instanceof Error ? err.message : 'Could not start the camera.';
}

/**
 * Open the real camera. The preferred request asks for the front/user-facing
 * camera at a sensible size — `facingMode` is what selects the selfie camera
 * on a phone and is simply ignored by a laptop webcam. Some laptop webcams and
 * virtual camera drivers reject any constraint set at all, so a rejection that
 * is about the CONSTRAINTS (rather than about permission) is retried with the
 * plainest possible request. A denied permission is never retried: that would
 * only prompt the user a second time for an answer they already gave.
 */
async function openCamera(): Promise<MediaStream> {
  const media = navigator.mediaDevices;
  try {
    return await media.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    const retryable =
      err instanceof DOMException &&
      (err.name === 'OverconstrainedError' ||
        err.name === 'ConstraintNotSatisfiedError' ||
        err.name === 'NotFoundError' ||
        err.name === 'NotReadableError' ||
        err.name === 'TrackStartError');
    if (!retryable) throw err;
    return await media.getUserMedia({ video: true, audio: false });
  }
}

export function useLiveCamera(): LiveCamera {
  const [status, setStatus] = useState<LiveStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [emotion, setEmotion] = useState<Emotion | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [distribution, setDistribution] = useState<LiveProbability[]>([]);
  const [faceDetected, setFaceDetected] = useState(false);
  const [faceCount, setFaceCount] = useState(0);
  const [samplesPerSecond, setSamplesPerSecond] = useState(0);
  const [framesAnalysed, setFramesAnalysed] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<number | null>(null);
  /** The loop reads this on every tick; false means "stop and release". */
  const runningRef = useRef(false);

  const samplesRef = useRef<Sample[]>([]);
  const candidateRef = useRef<{ emotion: Emotion; since: number } | null>(null);
  const shownRef = useRef<Emotion | null>(null);
  const lastFaceAtRef = useRef(0);
  const rttRef = useRef<number[]>([]);
  const lastTickAtRef = useRef(0);

  /* ------------------------------------------------------------------ */
  /* Teardown — must leave no timer, no request and no camera track live */
  /* ------------------------------------------------------------------ */
  const releaseHardware = useCallback(() => {
    runningRef.current = false;

    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    abortRef.current?.abort();
    abortRef.current = null;

    const stream = streamRef.current;
    if (stream) {
      for (const track of stream.getTracks()) track.stop();
      streamRef.current = null;
    }
    const video = videoRef.current;
    if (video) {
      video.srcObject = null;
      // Chrome keeps the capture indicator alive if the element still holds
      // a reference to the (now stopped) stream.
      video.removeAttribute('src');
      video.load();
    }
    canvasRef.current = null;
  }, []);

  const resetPrediction = useCallback(() => {
    samplesRef.current = [];
    candidateRef.current = null;
    shownRef.current = null;
    lastFaceAtRef.current = 0;
    rttRef.current = [];
    lastTickAtRef.current = 0;
    setEmotion(null);
    setConfidence(0);
    setDistribution([]);
    setFaceDetected(false);
    setFaceCount(0);
    setSamplesPerSecond(0);
  }, []);

  const stop = useCallback(() => {
    releaseHardware();
    resetPrediction();
    setFramesAnalysed(0);
    setStatus('idle');
    setError(null);
  }, [releaseHardware, resetPrediction]);

  /* ------------------------------------------------------------------ */
  /* Temporal stability                                                  */
  /* ------------------------------------------------------------------ */
  const applySample = useCallback((now: number, probs: number[] | null) => {
    const samples = samplesRef.current;
    if (probs) {
      samples.push({ t: now, probs });
      lastFaceAtRef.current = now;
    }
    // Drop everything older than the averaging window; this is also what makes
    // a lost face expire instead of being averaged in forever.
    while (samples.length && now - samples[0].t > WINDOW_MS) samples.shift();

    const faceLost = samples.length === 0 || now - lastFaceAtRef.current > FACE_LOST_MS;
    if (faceLost) {
      samplesRef.current = [];
      candidateRef.current = null;
      shownRef.current = null;
      setFaceDetected(false);
      setEmotion(null);
      setConfidence(0);
      setDistribution([]);
      return;
    }

    setFaceDetected(true);

    const mean = ZERO.slice();
    for (const s of samples) {
      for (let i = 0; i < mean.length; i += 1) mean[i] += s.probs[i];
    }
    const total = mean.reduce((a, b) => a + b, 0);
    for (let i = 0; i < mean.length; i += 1) mean[i] = total > 0 ? mean[i] / total : 0;

    let bestIdx = 0;
    for (let i = 1; i < mean.length; i += 1) if (mean[i] > mean[bestIdx]) bestIdx = i;
    const winner = EMOTIONS[bestIdx];
    const winnerScore = mean[bestIdx];

    // Track how long this winner has held the window's argmax.
    let candidate = candidateRef.current;
    if (!candidate || candidate.emotion !== winner) {
      candidate = { emotion: winner, since: now };
      candidateRef.current = candidate;
    }
    const heldFor = now - candidate.since;

    const backing = samples.reduce((n, s) => (s.t >= candidate.since ? n + 1 : n), 0);
    const shown = shownRef.current;
    const enoughEvidence =
      backing >= MIN_SAMPLES && winnerScore >= MIN_CONFIDENCE && heldFor >= STABILITY_MS;

    if (winner !== shown && enoughEvidence) {
      shownRef.current = winner;
      setEmotion(winner);
    }

    // Confidence always reflects the label actually on screen — the smoothed
    // probability of that class, not the winner's, and never a placeholder.
    const displayed = shownRef.current;
    setConfidence(displayed ? mean[EMOTIONS.indexOf(displayed)] : 0);

    setDistribution(
      EMOTIONS.map((e, i) => ({ emotion: e, value: mean[i] })).sort((a, b) => b.value - a.value),
    );
  }, []);

  /* ------------------------------------------------------------------ */
  /* Capture + inference loop                                            */
  /* ------------------------------------------------------------------ */
  const captureFrame = useCallback((): Promise<Blob | null> => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth) return Promise.resolve(null);

    const scale = Math.min(1, CAPTURE_MAX_WIDTH / video.videoWidth);
    const w = Math.max(1, Math.round(video.videoWidth * scale));
    const h = Math.max(1, Math.round(video.videoHeight * scale));

    let canvas = canvasRef.current;
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvasRef.current = canvas;
    }
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return Promise.resolve(null);
    // The preview is mirrored with CSS only; the pixels sent for inference are
    // the camera's own, so the crop geometry matches the training framing.
    ctx.drawImage(video, 0, 0, w, h);

    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', JPEG_QUALITY);
    });
  }, []);

  const scheduleNext = useCallback((delay: number, tick: () => void) => {
    if (!runningRef.current) return;
    timerRef.current = window.setTimeout(tick, Math.max(0, delay));
  }, []);

  const startLoop = useCallback(() => {
    const tick = async () => {
      if (!runningRef.current) return;
      const started = performance.now();

      let blob: Blob | null = null;
      try {
        blob = await captureFrame();
      } catch {
        blob = null;
      }
      if (!runningRef.current) return;
      if (!blob) {
        // The element is not ready yet (or the tab is hidden); try again soon.
        scheduleNext(MIN_CAPTURE_INTERVAL_MS, tick);
        return;
      }

      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const result = await analyzeLiveFrame(blob, controller.signal);
        if (!runningRef.current) return;

        const now = performance.now();
        setFaceCount(result.face_count);
        setFramesAnalysed((n) => n + 1);

        const usable =
          result.status === 'ok' && result.probabilities !== null && isEmotion(result.predicted_emotion ?? '');
        applySample(now, usable ? toVector(result.probabilities as Record<string, number>) : null);

        const gaps = rttRef.current;
        if (lastTickAtRef.current > 0) {
          gaps.push(now - lastTickAtRef.current);
          if (gaps.length > 12) gaps.shift();
          const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
          setSamplesPerSecond(mean > 0 ? Math.min(99, 1000 / mean) : 0);
        }
        lastTickAtRef.current = now;

        setError(null);
        scheduleNext(MIN_CAPTURE_INTERVAL_MS - (now - started), tick);
      } catch (err) {
        if (!runningRef.current) return;
        if (err instanceof DOMException && err.name === 'AbortError') return;
        // A transient backend hiccup must not kill the session or the camera;
        // surface it, back off, and keep trying.
        setError(err instanceof Error ? err.message : 'Live inference failed.');
        applySample(performance.now(), null);
        scheduleNext(700, tick);
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
      }
    };

    scheduleNext(0, tick);
  }, [applySample, captureFrame, scheduleNext]);

  const start = useCallback(async () => {
    if (runningRef.current) return;
    resetPrediction();
    setFramesAnalysed(0);
    setError(null);
    setStatus('starting');

    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      setStatus('error');
      setError(
        'This browser cannot open a camera here. Camera access needs a secure context — use https:// or localhost.',
      );
      return;
    }

    try {
      const stream = await openCamera();
      streamRef.current = stream;

      const video = videoRef.current;
      if (!video) {
        for (const track of stream.getTracks()) track.stop();
        streamRef.current = null;
        setStatus('error');
        setError('The camera preview is not ready. Try again.');
        return;
      }

      video.srcObject = stream;
      video.muted = true;
      try {
        await video.play();
      } catch {
        // Safari can reject play() while the element is still laying out; the
        // stream is attached either way and the loop waits for readyState.
      }

      runningRef.current = true;
      setStatus('running');
      startLoop();
    } catch (err) {
      releaseHardware();
      setStatus('error');
      setError(describeCameraError(err));
    }
  }, [releaseHardware, resetPrediction, startLoop]);

  /* Unmounting (including switching back to Upload) must release the camera. */
  useEffect(() => releaseHardware, [releaseHardware]);

  return {
    status,
    error,
    videoRef,
    emotion,
    confidence,
    distribution,
    faceDetected,
    faceCount,
    samplesPerSecond,
    framesAnalysed,
    start,
    stop,
  };
}
