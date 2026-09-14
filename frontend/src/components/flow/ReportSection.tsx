import {
  EMOTION_COLOR,
  EMOTION_LABEL,
  emotionColor,
  emotionLabel,
  type AnalysisResult,
} from '../../types/analysis';
import { formatBytes, formatDuration, formatSeconds, percent } from '../../lib/format';
import { Badge, Button, Card, Divider, Reveal, Section, SectionHeader } from '../ui/primitives';
import { IconCheck, IconDownload, IconSparkle } from '../ui/icons';

export function ReportSection({
  result,
  onReset,
}: {
  result: AnalysisResult;
  onReset: () => void;
}) {
  const { report } = result;

  const downloadJson = () => {
    const { video, ...rest } = result;
    const payload = {
      ...rest,
      video: { name: video.name, sizeBytes: video.sizeBytes, durationSec: video.durationSec },
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `emotion-report-${result.jobId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Section id="report">
      <div className="w-content-width mx-auto flex flex-col gap-8 md:gap-10">
        <SectionHeader
          eyebrow="Step 07 — Report"
          title="Final emotion analysis report"
          subtitle="Everything the pipeline produced, consolidated into a single summary you can export."
        />

        <Reveal>
          <Card className="overflow-hidden">
            {/* Header band */}
            <div
              className="relative px-6 md:px-10 py-8 md:py-10 flex flex-wrap items-start justify-between gap-6"
              style={{
                background: `linear-gradient(120deg, color-mix(in srgb, ${
                  emotionColor(result.dominantEmotion)
                } 16%, transparent), transparent 70%)`,
              }}
            >
              <div className="flex flex-col gap-2 min-w-0">
                <Badge className="inline-flex items-center gap-1.5">
                  <IconCheck className="size-3" />
                  Analysis complete
                </Badge>
                <h3 className="text-3xl md:text-4xl font-semibold mt-1 break-all">
                  {result.video.name}
                </h3>
                <p className="text-sm text-foreground/50">
                  Job {result.jobId} · {formatDuration(result.source.durationSec)} ·{' '}
                  {formatBytes(result.video.sizeBytes)} · {result.source.width}×
                  {result.source.height}
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <Button variant="secondary" onClick={downloadJson}>
                  <IconDownload className="size-4" />
                  Export JSON
                </Button>
                <Button onClick={onReset}>
                  <IconSparkle className="size-4" />
                  Analyse another
                </Button>
              </div>
            </div>

            <Divider />

            {/* Headline metrics */}
            <div className="grid grid-cols-2 lg:grid-cols-4 divide-x divide-y lg:divide-y-0 divide-foreground/8">
              <Cell
                label="Dominant emotion"
                value={emotionLabel(result.dominantEmotion)}
                accent={emotionColor(result.dominantEmotion)}
              />
              <Cell label="Overall confidence" value={percent(result.overallConfidence)} />
              <Cell label="Emotional stability" value={percent(report.stability, 0)} />
              <Cell label="Volatility" value={report.volatility} />
            </div>

            <Divider />

            {/* Body */}
            <div className="grid grid-cols-1 lg:grid-cols-[1.4fr_1fr] divide-y lg:divide-y-0 lg:divide-x divide-foreground/8">
              <div className="p-6 md:p-10 flex flex-col gap-6">
                <div>
                  <p className="text-xs uppercase tracking-widest text-foreground/55 mb-3">
                    Summary
                  </p>
                  <p className="text-lg md:text-xl leading-relaxed text-foreground/75 text-balance">
                    {report.summary}
                  </p>
                </div>

                <div>
                  <p className="text-xs uppercase tracking-widest text-foreground/55 mb-3">
                    Observations
                  </p>
                  <ul className="flex flex-col gap-3">
                    {report.notes.map((note) => (
                      <li key={note} className="flex items-start gap-3">
                        <span className="grid place-items-center size-5 shrink-0 rounded-full bg-primary-cta/10 text-primary-cta mt-0.5">
                          <IconCheck className="size-3" />
                        </span>
                        <span className="text-base text-foreground/65 leading-snug">{note}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="p-6 md:p-10 flex flex-col gap-6">
                <div>
                  <p className="text-xs uppercase tracking-widest text-foreground/55 mb-3">
                    Distribution
                  </p>
                  <ul className="flex flex-col gap-2.5">
                    {result.distribution.map((d) => (
                      <li key={d.emotion} className="flex items-center gap-3 text-base">
                        <span
                          className="size-2.5 rounded-full shrink-0"
                          style={{ background: EMOTION_COLOR[d.emotion] }}
                        />
                        <span className="flex-1 text-foreground/70">
                          {EMOTION_LABEL[d.emotion]}
                        </span>
                        <span className="tabular-nums font-medium">{percent(d.value)}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <Divider />

                <div>
                  <p className="text-xs uppercase tracking-widest text-foreground/55 mb-3">
                    Run details
                  </p>
                  <dl className="flex flex-col gap-2.5 text-base">
                    <Row k="Frames analysed" v={`${result.stats.framesAnalysed} / ${result.stats.framesTotal}`} />
                    <Row k="Sampling rate" v={`${result.model.samplingFps} fps`} />
                    <Row k="Source fps" v={result.source.fps.toFixed(2)} />
                    <Row k="Processing time" v={formatSeconds(result.stats.processingSec)} />
                    <Row k="Transitions" v={String(report.transitions)} />
                    <Row k="Coverage" v={percent(report.coverage, 0)} />
                    <Row k="Model" v={result.model.architecture} />
                  </dl>
                </div>
              </div>
            </div>
          </Card>
        </Reveal>
      </div>
    </Section>
  );
}

function Cell({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="px-6 md:px-8 py-7 flex flex-col gap-1.5">
      <span className="text-xs uppercase tracking-widest text-foreground/55">{label}</span>
      <span
        className="text-3xl md:text-4xl font-semibold leading-none"
        style={accent ? { color: accent } : undefined}
      >
        {value}
      </span>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <dt className="text-foreground/55">{k}</dt>
      <dd className="font-medium tabular-nums text-right">{v}</dd>
    </div>
  );
}
