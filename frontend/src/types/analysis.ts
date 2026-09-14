/**
 * Domain types for the facial-expression / emotion analysis flow.
 *
 * These mirror the REAL FastAPI report produced by
 * `ml/pipeline.py :: VideoEmotionPipeline.analyze`. The class vocabulary below
 * is the checkpoint's own `class_names` (FER+ ordering) — do not rename it.
 */

export const EMOTIONS = [
  'neutral',
  'happiness',
  'surprise',
  'sadness',
  'anger',
  'disgust',
  'fear',
] as const;

export type Emotion = (typeof EMOTIONS)[number];

export function isEmotion(value: string): value is Emotion {
  return (EMOTIONS as readonly string[]).includes(value);
}

export const EMOTION_LABEL: Record<Emotion, string> = {
  neutral: 'Neutral',
  happiness: 'Happiness',
  surprise: 'Surprise',
  sadness: 'Sadness',
  anger: 'Anger',
  disgust: 'Disgust',
  fear: 'Fear',
};

export const EMOTION_COLOR: Record<Emotion, string> = {
  neutral: 'var(--emotion-neutral)',
  happiness: 'var(--emotion-happy)',
  surprise: 'var(--emotion-surprise)',
  sadness: 'var(--emotion-sad)',
  anger: 'var(--emotion-angry)',
  disgust: 'var(--emotion-disgust)',
  fear: 'var(--emotion-fear)',
};

/**
 * Null-safe accessors. `dominant_emotion` is null when no usable face was
 * found — a valid pipeline result, so the UI must render it rather than crash.
 */
export function emotionLabel(emotion: Emotion | null): string {
  return emotion ? EMOTION_LABEL[emotion] : 'No face detected';
}

export function emotionColor(emotion: Emotion | null): string {
  return emotion ? EMOTION_COLOR[emotion] : 'var(--emotion-neutral)';
}

/** Where the user currently is in UPLOAD → PROCESSING → RESULTS. */
export type FlowStage = 'idle' | 'selected' | 'uploading' | 'processing' | 'complete' | 'error';

export interface VideoMeta {
  name: string;
  sizeBytes: number;
  durationSec: number;
  /** Object URL for local preview. Never sent anywhere. */
  previewUrl: string;
  mimeType: string;
}

export type ProcessingStepId =
  | 'upload'
  | 'decode'
  | 'detect'
  | 'align'
  | 'infer'
  | 'temporal'
  | 'report';

export type ProcessingStepState = 'pending' | 'active' | 'done';

export interface ProcessingStep {
  id: ProcessingStepId;
  label: string;
  detail: string;
  state: ProcessingStepState;
}

/** Aggregate of the largest-face detections across the sampled frames. */
export interface PrimaryFace {
  /** Mean normalised box (0–1) of the selected face, for the preview overlay. */
  box: { x: number; y: number; w: number; h: number };
  /** Mean MediaPipe detector score over frames that had a face. */
  detectionConfidence: number;
  /** quality.detection_rate — share of sampled frames containing a face. */
  presenceRatio: number;
  /** quality.usable_rate — share of sampled frames that produced a prediction. */
  usableRatio: number;
  /** Largest face_count seen in any sampled frame. */
  maxFacesInFrame: number;
  /** Number of sampled frames where more than one face was present. */
  multiFaceFrames: number;
  /** Mean short side of the selected face box, in source pixels. */
  averageFaceSizePx: number;
  /** Mean face_area_fraction, as a share of the frame area. */
  averageFaceAreaFraction: number;
}

export interface EmotionScore {
  emotion: Emotion;
  /** 0–1 (converted from the backend's 0–100 percentages). */
  value: number;
}

export interface TimelinePoint {
  /** Seconds from the start of the clip. */
  t: number;
  frameNumber: number;
  /** Smoothed label for this sample. */
  dominant: Emotion;
  /** Smoothed probability of the assigned label. */
  confidence: number;
  scores: Record<Emotion, number>;
}

export interface EmotionSegment {
  startSec: number;
  endSec: number;
  durationSec: number;
  startFrame: number;
  endFrame: number;
  sampleCount: number;
  emotion: Emotion;
  averageConfidence: number;
}

export interface EmotionTransition {
  from: Emotion;
  to: Emotion;
  timeSec: number;
  frame: number;
}

export interface FrameSample {
  id: string;
  timeSec: number;
  frameIndex: number;
  dominant: Emotion;
  /** Smoothed confidence for this frame. */
  confidence: number;
  /** Raw (unsmoothed) top-1 confidence for this frame. */
  rawConfidence: number;
  /** MediaPipe detector score for the selected face. */
  detectorScore: number;
  /** Share of the frame occupied by the face box. */
  faceAreaFraction: number;
}

/** Per-frame quality counters straight from `quality` in the report. */
export interface QualityStats {
  sampledFrames: number;
  analyzedFrames: number;
  framesWithFace: number;
  framesNoFace: number;
  framesFaceTooSmall: number;
  framesRejected: number;
  framesMultiFace: number;
  cropsShifted: number;
  cropsPadded: number;
  detectionRate: number;
  usableRate: number;
}

/** The pipeline's composite result-confidence — NOT model accuracy. */
export interface ConfidenceReport {
  level: string;
  /** 0–1, or null when no usable frame existed. */
  score: number | null;
  components: {
    margin: number | null;
    stability: number | null;
    coverage: number;
    meanEntropyNormalised: number | null;
  };
  note?: string;
}

export interface AnalysisResult {
  jobId: string | null;
  video: VideoMeta;
  source: {
    width: number;
    height: number;
    fps: number;
    frameCount: number;
    durationSec: number;
  };
  model: {
    architecture: string;
    backbone: string;
    checkpoint: string;
    checkpointEpoch: number | null;
    validationMacroF1: number | null;
    inputSize: number | string;
    device: string;
    classNames: string[];
    detector: string;
    samplingFps: number;
    preprocessing: string;
    aggregation: string;
    smoothingWindowSamples: number;
    smoothingWindowSeconds: number;
    minRunSamples: number;
  };
  stats: {
    framesTotal: number;
    sampledFrames: number;
    framesAnalysed: number;
    fps: number;
    processingSec: number;
  };
  quality: QualityStats;
  face: PrimaryFace;
  /** null when no usable face was found — a valid result, not an error. */
  dominantEmotion: Emotion | null;
  /** Mean smoothed confidence over frames labelled with the dominant emotion. */
  dominantConfidence: number;
  /** confidence.score — the pipeline's composite result confidence. */
  overallConfidence: number;
  confidence: ConfidenceReport;
  /** Mean smoothed probability mass per class (0–1). */
  distribution: EmotionScore[];
  /** Share of smoothed frame LABELS per class (0–1). */
  labelDistribution: EmotionScore[];
  timeline: TimelinePoint[];
  segments: EmotionSegment[];
  transitions: EmotionTransition[];
  frames: FrameSample[];
  report: {
    summary: string;
    /** confidence.components.stability */
    stability: number;
    transitions: number;
    /** Presentation band derived from `stability`. */
    volatility: 'Low' | 'Moderate' | 'High';
    /** confidence.components.coverage */
    coverage: number;
    notes: string[];
  };
  /** Backend-issued caution (e.g. low usable-frame rate). */
  warning: string | null;
}
