/**
 * Wire types — the exact JSON shape returned by the FastAPI pipeline
 * (`backend/app/schemas.py :: AnalysisResponse` plus the nested payloads built
 * in `ml/pipeline.py :: _report`). Snake_case is preserved deliberately: this
 * file is the wire contract, `mapReport.ts` is the only thing that translates.
 */

export interface WireFrame {
  sample_position: number | null;
  frame_number: number;
  timestamp_s: number;
  status: string; // ok | no_face | face_too_small | ...
  face_count: number;
  detector_score: number | null;
  box: [number, number, number, number] | null;
  face_area_fraction: number | null;
  crop_shifted: boolean;
  crop_padded: boolean;
  probabilities: Record<string, number> | null;
  predicted_emotion: string | null;
  confidence: number | null;
  smoothed_probabilities?: Record<string, number>;
  smoothed_emotion?: string;
  smoothed_confidence?: number;
}

export interface WireSegment {
  emotion: string;
  start_time_s: number;
  end_time_s: number;
  duration_s: number;
  start_frame: number;
  end_frame: number;
  sample_count: number;
  mean_confidence: number;
}

export interface WireTransition {
  from: string;
  to: string;
  time_s: number;
  frame: number;
}

export interface WireConfidence {
  level: string;
  score: number | null;
  components: {
    margin: number | null;
    stability: number | null;
    coverage: number;
    mean_entropy_normalised: number | null;
  };
  note?: string;
}

export interface WireQuality {
  sampled_frames: number;
  analyzed_frames: number;
  frames_with_face: number;
  frames_no_face: number;
  frames_face_too_small: number;
  frames_rejected: number;
  frames_multi_face: number;
  crops_shifted: number;
  crops_padded: number;
  detection_rate: number;
  usable_rate: number;
}

export interface WireReport {
  status: string;
  job_id: string | null;
  dominant_emotion: string | null;
  /** Percentages, 0–100. */
  emotion_distribution: Record<string, number>;
  /** Percentages, 0–100. */
  temporal_label_distribution: Record<string, number>;
  confidence: WireConfidence;
  frame_count: number;
  sampled_frames: number;
  analyzed_frames: number;
  quality: WireQuality;
  temporal_segments: WireSegment[];
  segment_count: number;
  transitions: WireTransition[];
  transition_count: number;
  video: {
    width: number;
    height: number;
    fps: number;
    frame_count: number;
    duration_s: number;
  };
  model: {
    architecture: string;
    backbone: string;
    checkpoint: string;
    checkpoint_epoch: number | null;
    validation_macro_f1: number | null;
    validation_accuracy: number | null;
    validation_balanced_accuracy: number | null;
    class_names: string[];
    input_size: number | string;
    norm_mean: unknown;
    norm_std: unknown;
    device: string;
  };
  analysis: {
    sampling_fps: number;
    detector: string;
    min_detection_confidence: number;
    min_face_px: number;
    crop_margin: number;
    crop_upward_offset: number;
    crop_calibration: string;
    preprocessing: string;
    smoothing_window_samples: number;
    smoothing_window_seconds: number;
    min_run_samples: number;
    max_gap_samples: number;
    aggregation: string;
    processing_time_s: number;
    upload_bytes?: number;
    source_filename?: string;
  };
  warning: string | null;
  frames: WireFrame[];
}

export interface WireJobCreated {
  job_id: string;
  status: string;
  filename: string;
}

export interface WireJobStatus {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  stage: string | null;
  filename: string | null;
  error: string | null;
  created_at: number | null;
  finished_at: number | null;
}

export interface WireHealth {
  status: string;
  device: string;
  gpu: string | null;
  model_loaded: boolean;
  detector_loaded: boolean;
  architecture: string;
  checkpoint_epoch: number | null;
  validation_macro_f1: number | null;
  class_names: string[];
}

/** `POST /api/live/frame` — one camera frame, one stateless prediction. */
export interface WireLiveFrame {
  status: string; // ok | no_face | face_too_small | ...
  face_count: number;
  detector_score: number | null;
  box: [number, number, number, number] | null;
  frame_width: number;
  frame_height: number;
  face_area_fraction: number | null;
  probabilities: Record<string, number> | null;
  predicted_emotion: string | null;
  confidence: number | null;
}
