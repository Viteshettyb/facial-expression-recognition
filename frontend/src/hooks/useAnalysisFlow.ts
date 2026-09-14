import { useCallback, useRef, useState } from 'react';
import { runAnalysis, uploadVideo } from '../api/client';
import type {
  AnalysisResult,
  FlowStage,
  ProcessingStep,
  ProcessingStepId,
  VideoMeta,
} from '../types/analysis';

const STEP_BLUEPRINT: Array<Omit<ProcessingStep, 'state'>> = [
  { id: 'upload', label: 'Upload', detail: 'Transferring the video to the analysis service' },
  { id: 'decode', label: 'Frame extraction', detail: 'Decoding the clip into sampled frames' },
  { id: 'detect', label: 'Face detection', detail: 'Locating faces and picking the largest track' },
  { id: 'align', label: 'Alignment', detail: 'Cropping and normalising each face' },
  { id: 'infer', label: 'Expression inference', detail: 'Classifying expressions frame by frame' },
  { id: 'temporal', label: 'Temporal analysis', detail: 'Smoothing predictions across the timeline' },
  { id: 'report', label: 'Report', detail: 'Aggregating results into the final report' },
];

const ORDER: ProcessingStepId[] = STEP_BLUEPRINT.map((s) => s.id);

function stepsFor(activeId: ProcessingStepId | null, complete: boolean): ProcessingStep[] {
  const activeIdx = activeId ? ORDER.indexOf(activeId) : -1;
  return STEP_BLUEPRINT.map((s, i) => ({
    ...s,
    state: complete || i < activeIdx ? 'done' : i === activeIdx ? 'active' : 'pending',
  }));
}

export interface AnalysisFlow {
  stage: FlowStage;
  video: VideoMeta | null;
  uploadPercent: number;
  analysisPercent: number;
  statusMessage: string;
  steps: ProcessingStep[];
  result: AnalysisResult | null;
  error: string | null;
  selectVideo: (file: File) => void;
  clearVideo: () => void;
  start: () => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

export function useAnalysisFlow(): AnalysisFlow {
  const [stage, setStage] = useState<FlowStage>('idle');
  const [video, setVideo] = useState<VideoMeta | null>(null);
  const [uploadPercent, setUploadPercent] = useState(0);
  const [analysisPercent, setAnalysisPercent] = useState(0);
  const [activeStep, setActiveStep] = useState<ProcessingStepId | null>(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileRef = useRef<File | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const revoke = useCallback(() => {
    setVideo((prev) => {
      if (prev) URL.revokeObjectURL(prev.previewUrl);
      return prev;
    });
  }, []);

  const selectVideo = useCallback(
    (file: File) => {
      revoke();
      fileRef.current = file;
      const previewUrl = URL.createObjectURL(file);

      const meta: VideoMeta = {
        name: file.name,
        sizeBytes: file.size,
        durationSec: 0,
        previewUrl,
        mimeType: file.type || 'video/mp4',
      };
      setVideo(meta);
      setResult(null);
      setError(null);
      setUploadPercent(0);
      setAnalysisPercent(0);
      setActiveStep(null);
      setStage('selected');

      // Read the real duration off a detached element.
      const probe = document.createElement('video');
      probe.preload = 'metadata';
      probe.onloadedmetadata = () => {
        setVideo((prev) =>
          prev && prev.previewUrl === previewUrl
            ? { ...prev, durationSec: Number.isFinite(probe.duration) ? probe.duration : 0 }
            : prev,
        );
      };
      probe.src = previewUrl;
    },
    [revoke],
  );

  const clearVideo = useCallback(() => {
    abortRef.current?.abort();
    revoke();
    fileRef.current = null;
    setVideo(null);
    setResult(null);
    setError(null);
    setUploadPercent(0);
    setAnalysisPercent(0);
    setActiveStep(null);
    setStage('idle');
  }, [revoke]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setStage('selected');
    setUploadPercent(0);
    setAnalysisPercent(0);
    setActiveStep(null);
    setStatusMessage('');
  }, []);

  const reset = useCallback(() => {
    clearVideo();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [clearVideo]);

  const start = useCallback(async () => {
    const file = fileRef.current;
    if (!file || !video) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);

    try {
      setStage('uploading');
      setActiveStep('upload');
      setStatusMessage('Uploading video…');
      const handle = await uploadVideo(file, setUploadPercent, controller.signal);

      setStage('processing');
      setStatusMessage('Starting analysis…');
      const analysis = await runAnalysis(
        handle,
        video,
        (p) => {
          setAnalysisPercent(p.percent);
          setActiveStep(p.stepId as ProcessingStepId);
          setStatusMessage(p.message);
        },
        controller.signal,
      );

      setResult(analysis);
      setStage('complete');
      setStatusMessage('Analysis complete');
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      setError(err instanceof Error ? err.message : 'Something went wrong during analysis.');
      setStage('error');
    }
  }, [video]);

  return {
    stage,
    video,
    uploadPercent,
    analysisPercent,
    statusMessage,
    steps: stepsFor(activeStep, stage === 'complete'),
    result,
    error,
    selectVideo,
    clearVideo,
    start,
    cancel,
    reset,
  };
}
