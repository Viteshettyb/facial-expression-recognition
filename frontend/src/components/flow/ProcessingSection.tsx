import type { AnalysisFlow } from '../../hooks/useAnalysisFlow';
import { cx } from '../../lib/format';
import { Card, Reveal, Section, SectionHeader, Spinner } from '../ui/primitives';
import { IconCheck } from '../ui/icons';
import { FaceMesh } from '../ui/FaceMesh';

export function ProcessingSection({ flow }: { flow: AnalysisFlow }) {
  const running = flow.stage === 'uploading' || flow.stage === 'processing';
  const done = flow.stage === 'complete';
  const percent =
    flow.stage === 'uploading'
      ? Math.round(flow.uploadPercent * 0.15)
      : flow.stage === 'processing'
        ? 15 + Math.round(flow.analysisPercent * 0.85)
        : done
          ? 100
          : 0;

  return (
    <Section id="processing">
      <div className="w-content-width mx-auto flex flex-col gap-8 md:gap-10">
        <SectionHeader
          eyebrow="Step 02 — Processing"
          title="Watch the pipeline run"
          subtitle="Frames are decoded, the primary face is isolated and aligned, then every crop is passed through the expression classifier."
        />

        <Reveal>
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.1fr] gap-4 md:gap-6">
            {/* -------- Live preview with detection overlay -------- */}
            <Card className="p-2 overflow-hidden">
              <div className="relative rounded-[calc(var(--radius)-0.5rem)] overflow-hidden bg-foreground/5 aspect-video grid place-items-center">
                {flow.video ? (
                  <video
                    key={`proc-${flow.video.previewUrl}`}
                    src={flow.video.previewUrl}
                    muted
                    loop
                    autoPlay
                    playsInline
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                ) : (
                  <FaceMesh className="w-3/5 opacity-40" />
                )}

                {/* Detection reticle */}
                <div className="absolute inset-0 pointer-events-none">
                  <div
                    className={cx(
                      'absolute rounded-xl border-2 transition-all duration-700',
                      running || done ? 'border-[var(--background-accent)]' : 'border-foreground/20',
                    )}
                    style={{ left: '33%', top: '14%', width: '34%', height: '58%' }}
                  >
                    <Corner className="-top-px -left-px border-t-2 border-l-2 rounded-tl-xl" />
                    <Corner className="-top-px -right-px border-t-2 border-r-2 rounded-tr-xl" />
                    <Corner className="-bottom-px -left-px border-b-2 border-l-2 rounded-bl-xl" />
                    <Corner className="-bottom-px -right-px border-b-2 border-r-2 rounded-br-xl" />
                    <span className="absolute -top-7 left-0 px-2 py-0.5 rounded-md text-[10px] font-medium tracking-wide bg-[var(--primary-cta)] text-primary-cta-text whitespace-nowrap">
                      face_01 · primary
                    </span>
                  </div>

                  {running && (
                    <div className="absolute inset-x-0 top-0 h-[6%] overflow-visible">
                      <div className="scanline h-full w-full bg-gradient-to-b from-transparent via-[var(--background-accent)]/45 to-transparent" />
                    </div>
                  )}
                </div>

                {!running && !done && (
                  <div className="absolute inset-0 grid place-items-center bg-background/55 backdrop-blur-[2px]">
                    <p className="text-base text-foreground/55 text-center px-6">
                      Upload a video and press <span className="font-medium text-foreground">Start</span> to
                      begin.
                    </p>
                  </div>
                )}
              </div>
            </Card>

            {/* -------- Progress + steps -------- */}
            <Card className="p-6 md:p-8 flex flex-col gap-6">
              <div className="flex flex-col gap-3">
                <div className="flex items-end justify-between gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-widest text-foreground/55">
                      Overall progress
                    </p>
                    <p className="text-base text-foreground/65 mt-1 inline-flex items-center gap-2">
                      {running && <Spinner className="size-3.5 text-accent" />}
                      {done ? 'Analysis complete' : flow.statusMessage || 'Idle — awaiting input'}
                    </p>
                  </div>
                  <span className="text-4xl md:text-5xl font-semibold tabular-nums leading-none">
                    {percent}
                    <span className="text-xl text-foreground/40">%</span>
                  </span>
                </div>

                <div className="relative h-2 w-full rounded-full bg-foreground/8 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-[var(--primary-cta)] to-[var(--background-accent)] transition-[width] duration-300 ease-out"
                    style={{ width: `${percent}%` }}
                  />
                  {running && (
                    <div className="absolute inset-y-0 left-0 w-1/4 shimmer bg-gradient-to-r from-transparent via-white/45 to-transparent" />
                  )}
                </div>
              </div>

              <ol className="flex flex-col">
                {flow.steps.map((step, i) => (
                  <li
                    key={step.id}
                    className={cx(
                      'flex items-start gap-4 py-3.5 transition-opacity duration-500',
                      i > 0 && 'border-t border-foreground/6',
                      step.state === 'pending' && 'opacity-45',
                    )}
                  >
                    <span
                      className={cx(
                        'grid place-items-center size-7 shrink-0 rounded-full text-[11px] font-semibold transition-all duration-500 mt-0.5',
                        step.state === 'done'
                          ? 'bg-primary-cta text-primary-cta-text'
                          : step.state === 'active'
                            ? 'bg-accent/15 text-accent ring-4 ring-accent/12'
                            : 'bg-foreground/8 text-foreground/50',
                      )}
                    >
                      {step.state === 'done' ? (
                        <IconCheck className="size-3.5" />
                      ) : step.state === 'active' ? (
                        <Spinner className="size-3.5" />
                      ) : (
                        i + 1
                      )}
                    </span>
                    <div className="min-w-0">
                      <p className="text-base font-medium leading-tight">{step.label}</p>
                      <p className="text-sm text-foreground/50 leading-snug mt-0.5">
                        {step.detail}
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
            </Card>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

function Corner({ className }: { className?: string }) {
  return (
    <span
      className={cx('absolute size-5 border-[var(--background-accent)]', className)}
      aria-hidden="true"
    />
  );
}
