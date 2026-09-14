import { useRef, useState } from 'react';
import { EMOTION_COLOR, EMOTION_LABEL, type AnalysisResult } from '../../types/analysis';
import { cx, formatSeconds, percent } from '../../lib/format';
import { Badge, Card, Reveal, Section, SectionHeader } from '../ui/primitives';
import { Sparkline } from '../ui/charts';

export function FramesSection({ result }: { result: AnalysisResult }) {
  const [selected, setSelected] = useState(0);
  const frame = result.frames[selected];
  const videoRef = useRef<HTMLVideoElement>(null);

  const pick = (i: number) => {
    setSelected(i);
    const v = videoRef.current;
    if (v) v.currentTime = result.frames[i].timeSec;
  };

  return (
    <Section id="frames">
      <div className="w-content-width mx-auto flex flex-col gap-8 md:gap-10">
        <SectionHeader
          eyebrow="Step 06 — Frames"
          title="Frame-level visual analysis"
          subtitle="Key sampled frames with their classification, face-crop quality and blur score. Select one to jump to that moment in the clip."
        />

        <Reveal>
          <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_1fr] gap-4 md:gap-6">
            {/* Selected frame viewer */}
            <Card className="p-2">
              <div className="relative rounded-[calc(var(--radius)-0.5rem)] overflow-hidden aspect-video bg-black/85">
                <video
                  ref={videoRef}
                  src={result.video.previewUrl}
                  muted
                  playsInline
                  preload="metadata"
                  className="absolute inset-0 w-full h-full object-contain"
                />
                <div
                  className="absolute rounded-lg border-2 pointer-events-none transition-all duration-500"
                  style={{
                    left: `${result.face.box.x * 100}%`,
                    top: `${result.face.box.y * 100}%`,
                    width: `${result.face.box.w * 100}%`,
                    height: `${result.face.box.h * 100}%`,
                    borderColor: EMOTION_COLOR[frame.dominant],
                    boxShadow: `0 0 0 9999px color-mix(in srgb, #000 22%, transparent)`,
                  }}
                >
                  <span
                    className="absolute -top-7 left-0 px-2 py-0.5 rounded-md text-[10px] font-medium text-white whitespace-nowrap"
                    style={{ background: EMOTION_COLOR[frame.dominant] }}
                  >
                    {EMOTION_LABEL[frame.dominant]} · {percent(frame.confidence, 0)}
                  </span>
                </div>

                <div className="absolute bottom-3 left-3 flex flex-wrap gap-2">
                  <Badge className="text-xs">Frame #{frame.frameIndex}</Badge>
                  <Badge className="text-xs">{formatSeconds(frame.timeSec)}</Badge>
                </div>
              </div>
            </Card>

            {/* Frame metrics */}
            <Card className="p-6 md:p-8 flex flex-col gap-6">
              <div>
                <p className="text-xs uppercase tracking-widest text-foreground/55">
                  Selected frame
                </p>
                <p
                  className="text-3xl md:text-4xl font-semibold mt-1"
                  style={{ color: EMOTION_COLOR[frame.dominant] }}
                >
                  {EMOTION_LABEL[frame.dominant]}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-5">
                <Meter label="Smoothed confidence" value={frame.confidence} />
                <Meter label="Raw confidence" value={frame.rawConfidence} />
                <Meter label="Detector score" value={frame.detectorScore} />
                <Meter label="Face area of frame" value={frame.faceAreaFraction} />
              </div>

              <div className="pt-4 border-t border-foreground/8">
                <p className="text-xs uppercase tracking-widest text-foreground/55 mb-2">
                  Confidence across sampled frames
                </p>
                <Sparkline values={result.frames.map((f) => f.confidence)} />
              </div>
            </Card>
          </div>
        </Reveal>

        {/* Frame strip */}
        <Reveal delay={100}>
          <div className="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-8 gap-3">
            {result.frames.map((f, i) => (
              <button
                key={f.id}
                onClick={() => pick(i)}
                className={cx(
                  'group relative rounded-token overflow-hidden aspect-4/5 cursor-pointer transition-all duration-300 text-left',
                  selected === i
                    ? 'ring-2 ring-offset-2 ring-offset-background scale-[1.02]'
                    : 'ring-1 ring-foreground/8 hover:ring-foreground/20 hover:-translate-y-1',
                )}
                style={
                  selected === i
                    ? ({ '--tw-ring-color': EMOTION_COLOR[f.dominant] } as React.CSSProperties)
                    : undefined
                }
              >
                <span
                  className="absolute inset-0"
                  style={{
                    background: `linear-gradient(160deg, color-mix(in srgb, ${
                      EMOTION_COLOR[f.dominant]
                    } 30%, transparent), color-mix(in srgb, ${
                      EMOTION_COLOR[f.dominant]
                    } 8%, transparent))`,
                  }}
                />
                <span className="absolute inset-0 flex flex-col justify-between p-3">
                  <span className="text-[10px] tabular-nums text-foreground/55">
                    {formatSeconds(f.timeSec)}
                  </span>
                  <span className="flex flex-col">
                    <span className="text-sm font-semibold leading-tight">
                      {EMOTION_LABEL[f.dominant]}
                    </span>
                    <span className="text-[11px] tabular-nums text-foreground/55">
                      {percent(f.confidence, 0)}
                    </span>
                  </span>
                </span>
              </button>
            ))}
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

function Meter({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs text-foreground/55">{label}</span>
      <span className="text-2xl font-semibold tabular-nums leading-none">
        {percent(value, 0)}
      </span>
      <span className="h-1.5 w-full rounded-full bg-foreground/8 overflow-hidden">
        <span
          className="block h-full rounded-full bg-accent transition-[width] duration-500"
          style={{ width: `${value * 100}%` }}
        />
      </span>
    </div>
  );
}
