import { useCallback, useRef, useState } from 'react';
import type { AnalysisFlow } from '../../hooks/useAnalysisFlow';
import { cx, formatBytes, formatDuration } from '../../lib/format';
import { Badge, Button, Card, Reveal, Section, SectionHeader, Spinner } from '../ui/primitives';
import { IconCheck, IconClose, IconUpload, IconVideo } from '../ui/icons';

const ACCEPT = 'video/mp4,video/quicktime,video/webm,video/x-matroska,video/avi';
const MAX_BYTES = 200 * 1024 * 1024;

export function UploadSection({ flow }: { flow: AnalysisFlow }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const busy = flow.stage === 'uploading' || flow.stage === 'processing';

  const accept = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!file.type.startsWith('video/')) {
        setLocalError('That file is not a video. Please choose an MP4, MOV, WEBM or MKV file.');
        return;
      }
      if (file.size > MAX_BYTES) {
        setLocalError(`Video is too large (${formatBytes(file.size)}). The limit is 200 MB.`);
        return;
      }
      setLocalError(null);
      flow.selectVideo(file);
    },
    [flow],
  );

  return (
    <Section id="upload">
      <div className="w-content-width mx-auto flex flex-col gap-8 md:gap-10">
        <SectionHeader
          eyebrow="Step 01 — Upload"
          title="Start with a video of a person speaking"
          subtitle="Best results come from a single, well-lit subject facing the camera. The file never leaves your machine until you press Start."
        />

        <Reveal>
          <div className="grid grid-cols-1 lg:grid-cols-[1.15fr_1fr] gap-4 md:gap-6">
            {/* ---------------- Dropzone / preview ---------------- */}
            <Card className="p-2">
              {!flow.video ? (
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragging(false);
                    accept(e.dataTransfer.files?.[0]);
                  }}
                  onClick={() => inputRef.current?.click()}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
                  }}
                  className={cx(
                    'relative flex flex-col items-center justify-center text-center gap-4',
                    'rounded-[calc(var(--radius)-0.5rem)] border-2 border-dashed cursor-pointer',
                    'px-6 py-16 md:py-24 transition-all duration-300',
                    dragging
                      ? 'border-accent bg-accent/6 scale-[0.995]'
                      : 'border-foreground/12 hover:border-accent/45 hover:bg-accent/4',
                  )}
                >
                  <div className="relative grid place-items-center size-16 rounded-full bg-primary-cta/8">
                    <span className="absolute inset-0 rounded-full bg-accent/20 soft-pulse" />
                    <IconUpload className="relative size-7 text-primary-cta" />
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <p className="text-xl md:text-2xl font-semibold">
                      Drop your video here
                    </p>
                    <p className="text-base text-foreground/55 max-w-[42ch]">
                      or click to browse. MP4, MOV, WEBM or MKV — up to 200 MB.
                    </p>
                  </div>

                  <Button variant="secondary" size="md" type="button" tabIndex={-1}>
                    Choose file
                  </Button>

                  <input
                    ref={inputRef}
                    type="file"
                    accept={ACCEPT}
                    className="hidden"
                    onChange={(e) => accept(e.target.files?.[0] ?? undefined)}
                  />
                </div>
              ) : (
                <div className="rounded-[calc(var(--radius)-0.5rem)] overflow-hidden bg-foreground/4">
                  <video
                    key={flow.video.previewUrl}
                    src={flow.video.previewUrl}
                    controls
                    playsInline
                    className="w-full aspect-video object-contain bg-black/85"
                  />
                </div>
              )}
            </Card>

            {/* ---------------- Status panel ---------------- */}
            <Card className="p-6 md:p-8 flex flex-col gap-6">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-xs uppercase tracking-widest text-foreground/55">
                    Selected video
                  </p>
                  <p className="text-xl md:text-2xl font-semibold mt-1.5 break-all">
                    {flow.video ? flow.video.name : 'No file selected'}
                  </p>
                </div>
                {flow.video && !busy && (
                  <button
                    onClick={() => flow.clearVideo()}
                    aria-label="Remove video"
                    className="shrink-0 grid place-items-center size-9 rounded-full border border-foreground/10 text-foreground/50 hover:text-foreground hover:border-foreground/25 transition-colors cursor-pointer"
                  >
                    <IconClose className="size-4" />
                  </button>
                )}
              </div>

              <dl className="grid grid-cols-2 gap-x-4 gap-y-5">
                <Field label="Size" value={flow.video ? formatBytes(flow.video.sizeBytes) : '—'} />
                <Field
                  label="Duration"
                  value={flow.video ? formatDuration(flow.video.durationSec) : '—'}
                />
                <Field label="Format" value={flow.video?.mimeType.split('/')[1]?.toUpperCase() ?? '—'} />
                <Field label="Frame rate" value={flow.video ? '30 fps (target)' : '—'} />
              </dl>

              {/* Upload progress */}
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-foreground/60 inline-flex items-center gap-2">
                    {flow.stage === 'uploading' && <Spinner className="size-3.5 text-accent" />}
                    {uploadLabel(flow)}
                  </span>
                  <span className="tabular-nums text-foreground/50">
                    {flow.stage === 'uploading' ? `${flow.uploadPercent}%` : ''}
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-foreground/8 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-accent transition-[width] duration-300 ease-out"
                    style={{
                      width:
                        flow.stage === 'idle'
                          ? '0%'
                          : flow.stage === 'selected'
                            ? '8%'
                            : flow.stage === 'uploading'
                              ? `${flow.uploadPercent}%`
                              : '100%',
                    }}
                  />
                </div>
              </div>

              {(localError || flow.error) && (
                <p className="text-sm text-[var(--emotion-angry)]">{localError ?? flow.error}</p>
              )}

              <div className="flex flex-wrap gap-3 mt-auto pt-2">
                <Button
                  onClick={() => void flow.start()}
                  disabled={!flow.video || busy}
                  size="lg"
                  className="flex-1 min-w-[190px]"
                >
                  {busy ? (
                    <>
                      <Spinner className="size-4" />
                      Analysing…
                    </>
                  ) : flow.stage === 'complete' ? (
                    <>
                      <IconCheck className="size-4" />
                      Re-run Analysis
                    </>
                  ) : (
                    <>
                      <IconVideo className="size-4" />
                      Start Facial Analysis
                    </>
                  )}
                </Button>
                {busy && (
                  <Button variant="ghost" size="lg" onClick={flow.cancel}>
                    Cancel
                  </Button>
                )}
              </div>
            </Card>
          </div>
        </Reveal>

        {/* Guidance row */}
        <Reveal delay={120}>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 md:gap-6">
            {[
              {
                t: 'One subject',
                d: 'The largest detected face is tracked. Additional faces are ignored.',
              },
              {
                t: 'Even lighting',
                d: 'Front-lit faces produce noticeably higher classification confidence.',
              },
              {
                t: '10–60 seconds',
                d: 'Long enough for a meaningful temporal profile, short enough to stay fast.',
              },
            ].map((g) => (
              <Card key={g.t} hover className="p-6 flex flex-col gap-2">
                <Badge className="mb-1">{g.t}</Badge>
                <p className="text-base text-foreground/60 leading-snug">{g.d}</p>
              </Card>
            ))}
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-widest text-foreground/55">{label}</dt>
      <dd className="text-base font-medium mt-1">{value}</dd>
    </div>
  );
}

function uploadLabel(flow: AnalysisFlow): string {
  switch (flow.stage) {
    case 'idle':
      return 'Waiting for a file';
    case 'selected':
      return 'Ready to upload';
    case 'uploading':
      return 'Uploading video';
    case 'processing':
      return 'Uploaded — analysis running';
    case 'complete':
      return 'Uploaded and analysed';
    case 'error':
      return 'Upload failed';
  }
}
