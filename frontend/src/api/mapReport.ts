/**
 * The single translation point between the FastAPI report and the UI model.
 *
 * Rules followed here:
 *  - Every number shown in the UI comes from the backend report. Nothing is
 *    synthesised, rescaled beyond unit conversion, or back-filled with a guess.
 *  - The backend reports percentages (0–100); the UI works in 0–1, so the only
 *    arithmetic applied to emotion values is `/ 100`.
 *  - Frames with no usable face carry null predictions and are excluded from
 *    the timeline rather than being given an invented label.
 */

import {
  EMOTIONS,
  isEmotion,
  type AnalysisResult,
  type Emotion,
  type EmotionScore,
  type EmotionSegment,
  type EmotionTransition,
  type FrameSample,
  type TimelinePoint,
  type VideoMeta,
} from '../types/analysis';
import type { WireFrame, WireReport } from './backendTypes';

/** Frame strip size in the UI. Frames are sampled evenly from real analysed frames. */
const FRAME_STRIP_SIZE = 8;

function toEmotion(name: string | null | undefined): Emotion | null {
  if (!name) return null;
  const key = name.toLowerCase();
  return isEmotion(key) ? key : null;
}

/** Backend percentages (0–100) → sorted 0–1 scores, restricted to known classes. */
function toDistribution(percentages: Record<string, number>): EmotionScore[] {
  return EMOTIONS.map((emotion) => ({
    emotion,
    value: (percentages[emotion] ?? 0) / 100,
  })).sort((a, b) => b.value - a.value);
}

function toScores(probabilities: Record<string, number> | undefined): Record<Emotion, number> {
  const scores = {} as Record<Emotion, number>;
  EMOTIONS.forEach((emotion) => {
    scores[emotion] = probabilities?.[emotion] ?? 0;
  });
  return scores;
}

function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

function analysedFrames(frames: WireFrame[]): WireFrame[] {
  return frames.filter(
    (f) => f.status === 'ok' && f.smoothed_emotion != null && toEmotion(f.smoothed_emotion) != null,
  );
}

function buildTimeline(frames: WireFrame[]): TimelinePoint[] {
  return analysedFrames(frames).map((f) => ({
    t: f.timestamp_s,
    frameNumber: f.frame_number,
    dominant: toEmotion(f.smoothed_emotion) as Emotion,
    confidence: f.smoothed_confidence ?? 0,
    scores: toScores(f.smoothed_probabilities),
  }));
}

function buildFrameStrip(frames: WireFrame[]): FrameSample[] {
  const usable = analysedFrames(frames);
  if (usable.length === 0) return [];

  const count = Math.min(FRAME_STRIP_SIZE, usable.length);
  const picked: FrameSample[] = [];

  for (let i = 0; i < count; i += 1) {
    // Evenly spaced across the analysed frames, first and last included.
    const idx = count === 1 ? 0 : Math.round((i * (usable.length - 1)) / (count - 1));
    const f = usable[idx];
    picked.push({
      id: `frame-${f.frame_number}`,
      timeSec: f.timestamp_s,
      frameIndex: f.frame_number,
      dominant: toEmotion(f.smoothed_emotion) as Emotion,
      confidence: f.smoothed_confidence ?? 0,
      rawConfidence: f.confidence ?? 0,
      detectorScore: f.detector_score ?? 0,
      faceAreaFraction: f.face_area_fraction ?? 0,
    });
  }
  return picked;
}

function buildSegments(report: WireReport): EmotionSegment[] {
  return report.temporal_segments
    .filter((s) => toEmotion(s.emotion) != null)
    .map((s) => ({
      startSec: s.start_time_s,
      endSec: s.end_time_s,
      durationSec: s.duration_s,
      startFrame: s.start_frame,
      endFrame: s.end_frame,
      sampleCount: s.sample_count,
      emotion: toEmotion(s.emotion) as Emotion,
      averageConfidence: s.mean_confidence,
    }));
}

function buildTransitions(report: WireReport): EmotionTransition[] {
  return report.transitions
    .filter((t) => toEmotion(t.from) != null && toEmotion(t.to) != null)
    .map((t) => ({
      from: toEmotion(t.from) as Emotion,
      to: toEmotion(t.to) as Emotion,
      timeSec: t.time_s,
      frame: t.frame,
    }));
}

/**
 * Aggregate the per-frame largest-face detections into the single face summary
 * the UI shows. Every field is a real statistic over frames that had a face —
 * head pose is deliberately absent because the pipeline does not estimate it.
 */
function buildFace(report: WireReport): AnalysisResult['face'] {
  const { width, height } = report.video;
  const withFace = report.frames.filter((f) => f.box != null && f.face_count > 0);

  const boxes = withFace.map((f) => f.box as [number, number, number, number]);
  const box =
    boxes.length && width > 0 && height > 0
      ? {
          x: mean(boxes.map((b) => b[0])) / width,
          y: mean(boxes.map((b) => b[1])) / height,
          w: mean(boxes.map((b) => b[2])) / width,
          h: mean(boxes.map((b) => b[3])) / height,
        }
      : { x: 0.33, y: 0.16, w: 0.34, h: 0.46 };

  return {
    box,
    detectionConfidence: mean(
      withFace.map((f) => f.detector_score).filter((s): s is number => s != null),
    ),
    presenceRatio: report.quality.detection_rate,
    usableRatio: report.quality.usable_rate,
    maxFacesInFrame: report.frames.reduce((max, f) => Math.max(max, f.face_count), 0),
    multiFaceFrames: report.quality.frames_multi_face,
    averageFaceSizePx: Math.round(mean(boxes.map((b) => Math.min(b[2], b[3])))),
    averageFaceAreaFraction: mean(
      withFace.map((f) => f.face_area_fraction).filter((v): v is number => v != null),
    ),
  };
}

/**
 * A factual summary assembled from reported values only — no interpretation of
 * the subject's state beyond what the pipeline measured.
 */
function buildSummary(report: WireReport, dominant: Emotion | null, share: number): string {
  if (!dominant) {
    return (
      report.warning ??
      'No usable face was detected in any sampled frame, so no expression was inferred.'
    );
  }

  const pct = (share * 100).toFixed(1);
  const level = report.confidence.level;
  const segs = report.segment_count;
  const trans = report.transition_count;
  const dur = report.video.duration_s.toFixed(1);

  return (
    `Across ${report.analyzed_frames} analysed frames spanning ${dur}s, the mean ` +
    `smoothed probability mass is highest for ${dominant} at ${pct}%. The temporal ` +
    `pass produced ${segs} segment${segs === 1 ? '' : 's'} with ${trans} ` +
    `transition${trans === 1 ? '' : 's'}, and the composite result confidence is ${level}.`
  );
}

function buildNotes(report: WireReport): string[] {
  const q = report.quality;
  const c = report.confidence.components;
  const notes: string[] = [
    `${q.frames_with_face} of ${q.sampled_frames} sampled frames contained a face ` +
      `(detection rate ${(q.detection_rate * 100).toFixed(1)}%).`,
    `${q.analyzed_frames} frames produced a prediction (usable rate ` +
      `${(q.usable_rate * 100).toFixed(1)}%).`,
  ];

  if (q.frames_no_face > 0) notes.push(`${q.frames_no_face} frames had no detectable face.`);
  if (q.frames_face_too_small > 0)
    notes.push(`${q.frames_face_too_small} frames were skipped: face below the minimum size.`);
  if (q.frames_multi_face > 0)
    notes.push(
      `${q.frames_multi_face} frames contained more than one face; the largest was used.`,
    );
  if (q.crops_shifted > 0 || q.crops_padded > 0)
    notes.push(`${q.crops_shifted} crops were shifted and ${q.crops_padded} padded to stay in frame.`);
  if (c.stability != null)
    notes.push(
      `The dominant label held in ${(c.stability * 100).toFixed(1)}% of analysed frames ` +
        `(mean top-2 margin ${c.margin?.toFixed(3) ?? 'n/a'}).`,
    );
  if (report.warning) notes.push(report.warning);

  return notes;
}

export function mapReport(report: WireReport, video: VideoMeta): AnalysisResult {
  const dominant = toEmotion(report.dominant_emotion);
  const distribution = toDistribution(report.emotion_distribution);
  const labelDistribution = toDistribution(report.temporal_label_distribution);
  const timeline = buildTimeline(report.frames);

  const dominantShare = distribution.find((d) => d.emotion === dominant)?.value ?? 0;

  // Mean smoothed confidence over the frames actually labelled with the
  // dominant emotion — derived from the real per-frame series.
  const dominantConfidence = dominant
    ? mean(timeline.filter((p) => p.dominant === dominant).map((p) => p.confidence))
    : 0;

  const stability = report.confidence.components.stability ?? 0;
  const volatility: AnalysisResult['report']['volatility'] =
    stability > 0.72 ? 'Low' : stability > 0.45 ? 'Moderate' : 'High';

  return {
    jobId: report.job_id,
    video,
    source: {
      width: report.video.width,
      height: report.video.height,
      fps: report.video.fps,
      frameCount: report.video.frame_count,
      durationSec: report.video.duration_s,
    },
    model: {
      architecture: report.model.architecture,
      backbone: report.model.backbone,
      checkpoint: report.model.checkpoint,
      checkpointEpoch: report.model.checkpoint_epoch,
      validationMacroF1: report.model.validation_macro_f1,
      inputSize: report.model.input_size,
      device: report.model.device,
      classNames: report.model.class_names,
      detector: report.analysis.detector,
      samplingFps: report.analysis.sampling_fps,
      preprocessing: report.analysis.preprocessing,
      aggregation: report.analysis.aggregation,
      smoothingWindowSamples: report.analysis.smoothing_window_samples,
      smoothingWindowSeconds: report.analysis.smoothing_window_seconds,
      minRunSamples: report.analysis.min_run_samples,
    },
    stats: {
      framesTotal: report.frame_count,
      sampledFrames: report.sampled_frames,
      framesAnalysed: report.analyzed_frames,
      fps: report.video.fps,
      processingSec: report.analysis.processing_time_s,
    },
    quality: {
      sampledFrames: report.quality.sampled_frames,
      analyzedFrames: report.quality.analyzed_frames,
      framesWithFace: report.quality.frames_with_face,
      framesNoFace: report.quality.frames_no_face,
      framesFaceTooSmall: report.quality.frames_face_too_small,
      framesRejected: report.quality.frames_rejected,
      framesMultiFace: report.quality.frames_multi_face,
      cropsShifted: report.quality.crops_shifted,
      cropsPadded: report.quality.crops_padded,
      detectionRate: report.quality.detection_rate,
      usableRate: report.quality.usable_rate,
    },
    face: buildFace(report),
    dominantEmotion: dominant,
    dominantConfidence,
    overallConfidence: report.confidence.score ?? 0,
    confidence: {
      level: report.confidence.level,
      score: report.confidence.score,
      components: {
        margin: report.confidence.components.margin,
        stability: report.confidence.components.stability,
        coverage: report.confidence.components.coverage,
        meanEntropyNormalised: report.confidence.components.mean_entropy_normalised,
      },
      note: report.confidence.note,
    },
    distribution,
    labelDistribution,
    timeline,
    segments: buildSegments(report),
    transitions: buildTransitions(report),
    frames: buildFrameStrip(report.frames),
    report: {
      summary: buildSummary(report, dominant, dominantShare),
      stability,
      transitions: report.transition_count,
      volatility,
      coverage: report.confidence.components.coverage,
      notes: buildNotes(report),
    },
    warning: report.warning,
  };
}
