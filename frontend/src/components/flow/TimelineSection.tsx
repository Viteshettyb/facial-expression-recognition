import { useState } from 'react';
import {
  EMOTION_COLOR,
  EMOTION_LABEL,
  type AnalysisResult,
  type Emotion,
} from '../../types/analysis';
import { cx, formatSeconds, percent } from '../../lib/format';
import { Badge, Card, Reveal, Section, SectionHeader } from '../ui/primitives';
import { SegmentRibbon, TemporalChart } from '../ui/charts';

export function TimelineSection({ result }: { result: AnalysisResult }) {
  const topEmotions = result.distribution.slice(0, 4).map((d) => d.emotion);
  const [visible, setVisible] = useState<Emotion[]>(topEmotions.slice(0, 3));
  const duration = result.timeline[result.timeline.length - 1].t;

  const toggle = (e: Emotion) =>
    setVisible((prev) =>
      prev.includes(e) ? (prev.length > 1 ? prev.filter((x) => x !== e) : prev) : [...prev, e],
    );

  return (
    <Section id="timeline">
      <div className="w-content-width mx-auto flex flex-col gap-8 md:gap-10">
        <SectionHeader
          eyebrow="Step 05 — Temporal"
          title="How the emotion changed over time"
          subtitle="Per-frame probabilities smoothed across the clip, with the dominant expression segmented into contiguous windows."
        />

        <Reveal>
          <Card className="p-6 md:p-8 flex flex-col gap-6">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div>
                <p className="text-xs uppercase tracking-widest text-foreground/55">
                  Probability over time
                </p>
                <p className="text-2xl md:text-3xl font-semibold mt-1">Emotion trajectory</p>
              </div>

              <div className="flex flex-wrap gap-2">
                {result.distribution.map((d) => {
                  const on = visible.includes(d.emotion);
                  return (
                    <button
                      key={d.emotion}
                      onClick={() => toggle(d.emotion)}
                      className={cx(
                        'inline-flex items-center gap-2 px-3 py-1.5 rounded-token text-sm cursor-pointer border transition-all duration-300',
                        on
                          ? 'border-transparent card-solid text-foreground'
                          : 'border-foreground/10 text-foreground/55 hover:text-foreground/75 hover:border-foreground/20',
                      )}
                    >
                      <span
                        className="size-2.5 rounded-full transition-opacity"
                        style={{
                          background: EMOTION_COLOR[d.emotion],
                          opacity: on ? 1 : 0.35,
                        }}
                      />
                      {EMOTION_LABEL[d.emotion]}
                    </button>
                  );
                })}
              </div>
            </div>

            <TemporalChart timeline={result.timeline} visible={visible} />
          </Card>
        </Reveal>

        <Reveal delay={100}>
          <div className="grid grid-cols-1 lg:grid-cols-[1.4fr_1fr] gap-4 md:gap-6">
            <Card className="p-6 md:p-8 flex flex-col gap-5">
              <div>
                <p className="text-xs uppercase tracking-widest text-foreground/55">
                  Dominant emotion map
                </p>
                <p className="text-2xl md:text-3xl font-semibold mt-1">
                  {result.segments.length} segments · {result.report.transitions} transitions
                </p>
              </div>

              <SegmentRibbon segments={result.segments} duration={duration} />

              <div className="flex flex-wrap gap-x-5 gap-y-2 pt-1">
                {[...new Set(result.segments.map((s) => s.emotion))].map((e) => (
                  <span
                    key={e}
                    className="inline-flex items-center gap-2 text-sm text-foreground/60"
                  >
                    <span
                      className="size-2.5 rounded-full"
                      style={{ background: EMOTION_COLOR[e] }}
                    />
                    {EMOTION_LABEL[e]}
                  </span>
                ))}
              </div>
            </Card>

            <Card className="p-6 md:p-8 flex flex-col gap-4">
              <p className="text-xs uppercase tracking-widest text-foreground/55">
                Segment breakdown
              </p>
              <ul className="flex flex-col overflow-y-auto max-h-[300px] pr-1">
                {result.segments.map((s, i) => (
                  <li
                    key={`${s.emotion}-${i}`}
                    className={cx(
                      'flex items-center gap-3 py-3',
                      i > 0 && 'border-t border-foreground/6',
                    )}
                  >
                    <span
                      className="size-2.5 rounded-full shrink-0"
                      style={{ background: EMOTION_COLOR[s.emotion] }}
                    />
                    <span className="text-base font-medium flex-1">
                      {EMOTION_LABEL[s.emotion]}
                    </span>
                    <span className="text-sm tabular-nums text-foreground/50">
                      {formatSeconds(s.startSec)} – {formatSeconds(s.endSec)}
                    </span>
                    <Badge className="text-xs">{percent(s.averageConfidence, 0)}</Badge>
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}
