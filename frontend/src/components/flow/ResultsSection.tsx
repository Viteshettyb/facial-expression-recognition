import {
  EMOTION_COLOR,
  EMOTION_LABEL,
  emotionColor,
  emotionLabel,
  type AnalysisResult,
} from '../../types/analysis';
import { percent } from '../../lib/format';
import { useCountUp } from '../../hooks/useCountUp';
import { useInView } from '../../hooks/useInView';
import { Badge, Card, Reveal, Section, SectionHeader } from '../ui/primitives';
import { ConfidenceGauge, DonutChart, EmotionBars } from '../ui/charts';

export function ResultsSection({ result }: { result: AnalysisResult }) {
  const { ref, inView } = useInView<HTMLDivElement>(0.3);
  const dominantShare =
    result.distribution.find((d) => d.emotion === result.dominantEmotion)?.value ?? 0;
  const dominantPct = useCountUp(dominantShare * 100, inView);

  return (
    <Section id="results">
      <div className="w-content-width mx-auto flex flex-col gap-8 md:gap-10">
        <SectionHeader
          eyebrow="Step 04 — Predictions"
          title="Expression predictions and dominant emotion"
          subtitle="Every sampled frame produces a probability across seven expression classes. Aggregated over the clip, they give the dominant emotion and its distribution."
        />

        {/* -------- Dominant emotion hero card -------- */}
        <Reveal>
          <div
            ref={ref}
            className="relative overflow-hidden rounded-token card p-8 md:p-12 grid grid-cols-1 lg:grid-cols-[1.1fr_auto] gap-8 items-center"
          >
            <div
              className="absolute inset-0 -z-10 opacity-70"
              style={{
                background: `radial-gradient(80% 120% at 12% 0%, color-mix(in srgb, ${
                  emotionColor(result.dominantEmotion)
                } 24%, transparent), transparent 65%)`,
              }}
            />

            <div className="flex flex-col gap-4">
              <Badge>Dominant emotion</Badge>
              <div className="flex items-baseline gap-4 flex-wrap">
                <h3
                  className="text-6xl md:text-7xl 2xl:text-8xl font-semibold leading-none"
                  style={{ color: emotionColor(result.dominantEmotion) }}
                >
                  {emotionLabel(result.dominantEmotion)}
                </h3>
                <span className="text-3xl md:text-4xl font-semibold tabular-nums text-foreground/55">
                  {dominantPct.toFixed(1)}%
                </span>
              </div>
              <p className="text-lg md:text-xl text-foreground/60 leading-snug max-w-[54ch]">
                The strongest expression across the clip, held with an average class confidence of{' '}
                <span className="font-medium text-foreground">
                  {percent(result.dominantConfidence)}
                </span>{' '}
                over {result.stats.framesAnalysed} analysed frames.
              </p>

              <div className="flex flex-wrap gap-2 mt-2">
                {result.distribution.slice(0, 4).map((d) => (
                  <span
                    key={d.emotion}
                    className="inline-flex items-center gap-2 px-3 py-1.5 rounded-token text-sm card-solid"
                  >
                    <span
                      className="size-2.5 rounded-full"
                      style={{ background: EMOTION_COLOR[d.emotion] }}
                    />
                    {EMOTION_LABEL[d.emotion]}
                    <span className="tabular-nums text-foreground/50">{percent(d.value, 0)}</span>
                  </span>
                ))}
              </div>
            </div>

            <ConfidenceGauge value={result.overallConfidence} size={210} />
          </div>
        </Reveal>

        {/* -------- Distribution -------- */}
        <Reveal delay={100}>
          <div className="grid grid-cols-1 lg:grid-cols-[auto_1fr] gap-4 md:gap-6">
            <Card className="p-8 flex flex-col items-center justify-center gap-6">
              <DonutChart
                data={result.distribution}
                centerLabel="Dominant"
                centerValue={emotionLabel(result.dominantEmotion)}
              />
              <p className="text-sm text-foreground/55 text-center max-w-[30ch]">
                Percentage of total expression probability mass, hover a segment for detail.
              </p>
            </Card>

            <Card className="p-6 md:p-8 flex flex-col gap-6">
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div>
                  <p className="text-xs uppercase tracking-widest text-foreground/55">
                    Emotion distribution
                  </p>
                  <p className="text-2xl md:text-3xl font-semibold mt-1">
                    Across the full clip
                  </p>
                </div>
                <Badge>{result.stats.framesAnalysed} samples</Badge>
              </div>

              <EmotionBars data={result.distribution} />

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 border-t border-foreground/8">
                <Mini
                  label="Result confidence"
                  value={`${percent(result.overallConfidence)} · ${result.confidence.level}`}
                />
                <Mini label="Dominant conf." value={percent(result.dominantConfidence)} />
                <Mini label="Classes" value={String(result.distribution.length)} />
                <Mini label="Processing" value={`${result.stats.processingSec}s`} />
              </div>
            </Card>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-foreground/55">{label}</p>
      <p className="text-xl font-semibold tabular-nums mt-0.5">{value}</p>
    </div>
  );
}
