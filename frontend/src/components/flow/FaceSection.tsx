import type { AnalysisResult } from '../../types/analysis';
import { percent } from '../../lib/format';
import { Badge, Card, Reveal, Section, SectionHeader } from '../ui/primitives';
import { FaceMesh } from '../ui/FaceMesh';

export function FaceSection({ result }: { result: AnalysisResult }) {
  const { face, model, quality } = result;

  return (
    <Section id="face">
      <div className="w-content-width mx-auto flex flex-col gap-8 md:gap-10">
        <SectionHeader
          eyebrow="Step 03 — Detection"
          title="Primary face located and locked"
          subtitle="Across all sampled frames the detector selects the largest, most consistently visible face and tracks it as the analysis subject."
        />

        <Reveal>
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.15fr] gap-4 md:gap-6">
            {/* Crop preview */}
            <Card className="p-2">
              <div className="relative rounded-[calc(var(--radius)-0.5rem)] overflow-hidden aspect-square bg-foreground/5 grid place-items-center">
                <video
                  src={result.video.previewUrl}
                  muted
                  loop
                  autoPlay
                  playsInline
                  className="absolute inset-0 w-full h-full object-cover"
                  style={{
                    transform: `scale(${1 / Math.max(face.box.w, face.box.h) / 1.35})`,
                    transformOrigin: `${(face.box.x + face.box.w / 2) * 100}% ${
                      (face.box.y + face.box.h / 2) * 100
                    }%`,
                  }}
                />
                <FaceMesh className="relative w-4/5 opacity-70 mix-blend-screen" />
                <div className="absolute bottom-3 left-3 right-3 flex flex-wrap gap-2">
                  <Badge className="text-xs">Largest face</Badge>
                  <Badge className="text-xs">
                    {percent(face.detectionConfidence)} detection
                  </Badge>
                </div>
              </div>
            </Card>

            {/* Metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 md:gap-6 content-start">
              <Metric
                label="Max faces in frame"
                value={String(face.maxFacesInFrame)}
                hint={
                  face.multiFaceFrames > 0
                    ? `${face.multiFaceFrames} frames had more than one face`
                    : 'Single subject throughout'
                }
              />
              <Metric
                label="Frame presence"
                value={percent(face.presenceRatio)}
                hint={`${quality.framesWithFace} of ${quality.sampledFrames} sampled frames`}
              />
              <Metric
                label="Average face size"
                value={`${face.averageFaceSizePx} px`}
                hint={`Model input ${model.inputSize}`}
              />
              <Metric
                label="Usable rate"
                value={percent(face.usableRatio)}
                hint={`${quality.analyzedFrames} frames produced a prediction`}
              />

              <Card className="p-6 sm:col-span-2 flex flex-col gap-3">
                <p className="text-xs uppercase tracking-widest text-foreground/55">
                  Detection pipeline
                </p>
                <div className="flex flex-wrap gap-x-8 gap-y-3">
                  <Pair k="Detector" v={model.detector} />
                  <Pair k="Classifier" v={model.architecture} />
                  <Pair k="Backbone" v={model.backbone} />
                </div>
              </Card>
            </div>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <Card hover className="p-6 flex flex-col gap-1.5">
      <span className="text-xs uppercase tracking-widest text-foreground/55">{label}</span>
      <span className="text-3xl md:text-4xl font-semibold leading-none tabular-nums">{value}</span>
      <span className="text-sm text-foreground/50 leading-snug mt-0.5">{hint}</span>
    </Card>
  );
}

function Pair({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <p className="text-xs text-foreground/55">{k}</p>
      <p className="text-base font-medium mt-0.5">{v}</p>
    </div>
  );
}
