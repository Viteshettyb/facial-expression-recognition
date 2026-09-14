import { useEffect, useRef, useState } from 'react';
import { useLiveCamera } from '../../hooks/useLiveCamera';
import { percent } from '../../lib/format';
import { EMOTION_LABEL, emotionColor, type Emotion } from '../../types/analysis';
import { Badge, Button, Card, Reveal, Section, SectionHeader, Spinner } from '../ui/primitives';
import { IconClose, IconFace, IconVideo } from '../ui/icons';

/**
 * Live camera analysis. The preview is local and continuous; the label under
 * it is driven by `useLiveCamera`, which averages real per-frame probabilities
 * over a time window and only switches once a new expression has genuinely
 * held. Nothing here invents a prediction — when there is no face, it says so.
 */
export function LiveCameraSection() {
  const live = useLiveCamera();
  const running = live.status === 'running';
  const starting = live.status === 'starting';

  return (
    <Section id="live">
      <div className="w-content-width mx-auto flex flex-col gap-8 md:gap-10">
        <SectionHeader
          eyebrow="Live — Camera"
          title="Read expressions straight from your camera"
          subtitle="The same detector, crop and classifier used for uploaded video, running on your camera in real time. Frames are analysed one at a time and never stored."
        />

        <Reveal>
          <div className="grid grid-cols-1 lg:grid-cols-[1.15fr_1fr] gap-4 md:gap-6">
            {/* ---------------- Preview ---------------- */}
            <Card className="p-2">
              <div className="relative rounded-[calc(var(--radius)-0.5rem)] overflow-hidden aspect-video bg-black/85">
                <video
                  ref={live.videoRef}
                  muted
                  playsInline
                  autoPlay
                  className="camera-mirror absolute inset-0 w-full h-full object-cover"
                />

                {!running && !starting && (
                  <div className="absolute inset-0 grid place-items-center text-center px-6">
                    <div className="flex flex-col items-center gap-4">
                      <div className="relative grid place-items-center size-16 rounded-full bg-primary-cta/10">
                        <span className="absolute inset-0 rounded-full bg-accent/20 soft-pulse" />
                        <IconFace className="relative size-7 text-primary-cta" />
                      </div>
                      <p className="text-lg text-foreground/65 max-w-[36ch]">
                        Your camera is off. Start the session and allow camera access when your
                        browser asks.
                      </p>
                    </div>
                  </div>
                )}

                {starting && (
                  <div className="absolute inset-0 grid place-items-center bg-black/50">
                    <span className="inline-flex items-center gap-3 text-base text-foreground/80">
                      <Spinner className="size-4 text-accent" />
                      Waiting for camera permission…
                    </span>
                  </div>
                )}

                {running && (
                  <div className="absolute top-3 left-3 flex flex-wrap gap-2">
                    <Badge className="text-xs inline-flex items-center gap-2">
                      <span className="size-2 rounded-full bg-[var(--emotion-angry)] soft-pulse" />
                      Live
                    </Badge>
                    <Badge className="text-xs">
                      {live.samplesPerSecond > 0
                        ? `${live.samplesPerSecond.toFixed(1)} analyses/s`
                        : 'measuring…'}
                    </Badge>
                    {live.faceCount > 1 && (
                      <Badge className="text-xs">
                        {live.faceCount} faces — largest tracked
                      </Badge>
                    )}
                  </div>
                )}
              </div>

              {/* ------------- The expression, prominently below the camera ------------- */}
              <ExpressionReadout
                running={running || starting}
                faceDetected={live.faceDetected}
                emotion={live.emotion}
                confidence={live.confidence}
              />
            </Card>

            {/* ---------------- Controls + live distribution ---------------- */}
            <Card className="p-6 md:p-8 flex flex-col gap-6">
              <div>
                <p className="text-xs uppercase tracking-widest text-foreground/55">Session</p>
                <p className="text-xl md:text-2xl font-semibold mt-1.5">
                  {running ? 'Camera analysis running' : starting ? 'Starting camera' : 'Camera idle'}
                </p>
              </div>

              <dl className="grid grid-cols-2 gap-x-4 gap-y-5">
                <Field
                  label="Faces in frame"
                  value={running ? String(live.faceCount) : '—'}
                />
                <Field
                  label="Frames analysed"
                  value={live.framesAnalysed > 0 ? String(live.framesAnalysed) : '—'}
                />
                <Field label="Camera" value={running ? 'Front / user facing' : 'Not open'} />
                <Field label="Model input" value="112 × 112 (ResNet-18)" />
              </dl>

              <div className="flex flex-col gap-3">
                <p className="text-xs uppercase tracking-widest text-foreground/55">
                  Smoothed distribution
                </p>
                <DistributionBars
                  rows={live.distribution}
                  active={live.emotion}
                  empty={!running || !live.faceDetected}
                />
              </div>

              {live.error && (
                <p className="text-sm text-[var(--emotion-angry)]">{live.error}</p>
              )}

              <div className="flex flex-wrap gap-3 mt-auto pt-2">
                {!running ? (
                  <Button
                    onClick={() => void live.start()}
                    disabled={starting}
                    size="lg"
                    className="flex-1 min-w-[190px]"
                  >
                    {starting ? (
                      <>
                        <Spinner className="size-4" />
                        Starting…
                      </>
                    ) : live.status === 'error' ? (
                      <>
                        <IconVideo className="size-4" />
                        Try Camera Again
                      </>
                    ) : (
                      <>
                        <IconVideo className="size-4" />
                        Start Live Camera
                      </>
                    )}
                  </Button>
                ) : (
                  <Button
                    onClick={live.stop}
                    variant="secondary"
                    size="lg"
                    className="flex-1 min-w-[190px]"
                  >
                    <IconClose className="size-4" />
                    End Camera Analysis
                  </Button>
                )}
              </div>
            </Card>
          </div>
        </Reveal>

        <Reveal delay={120}>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 md:gap-6">
            {[
              {
                t: 'Stable, not twitchy',
                d: 'Predictions are averaged over roughly a second; a new expression must hold before the label changes.',
              },
              {
                t: 'Largest face only',
                d: 'If several people are in frame, the largest detected face is the one analysed.',
              },
              {
                t: 'Nothing is kept',
                d: 'Each frame is analysed and discarded. No video is recorded or written to disk.',
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

/* ------------------------------------------------------------------ */
/* The headline readout                                                */
/* ------------------------------------------------------------------ */

function ExpressionReadout({
  running,
  faceDetected,
  emotion,
  confidence,
}: {
  running: boolean;
  faceDetected: boolean;
  emotion: Emotion | null;
  confidence: number;
}) {
  // Re-key the label on every change so the swap animation replays; without
  // this React would reuse the node and the text would simply pop.
  const [key, setKey] = useState(0);
  const previous = useRef<Emotion | null>(null);
  useEffect(() => {
    if (previous.current !== emotion) {
      previous.current = emotion;
      setKey((k) => k + 1);
    }
  }, [emotion]);

  const color = emotion ? emotionColor(emotion) : 'var(--emotion-neutral)';

  let headline: string;
  let sub: string;
  if (!running) {
    headline = 'Camera off';
    sub = 'Start the session to begin live analysis.';
  } else if (!faceDetected) {
    headline = 'Face not detected';
    sub = 'Move into frame and face the camera.';
  } else if (!emotion) {
    headline = 'Reading expression…';
    sub = 'Collecting enough stable evidence before committing to a label.';
  } else {
    headline = EMOTION_LABEL[emotion].toUpperCase();
    sub = 'Smoothed over the last second of frames.';
  }

  return (
    <div className="px-4 md:px-6 pt-5 pb-4 flex flex-col items-center text-center gap-2">
      <p className="text-xs uppercase tracking-widest text-foreground/55">
        Current expression
      </p>

      <p
        key={key}
        className="emotion-swap text-4xl md:text-6xl font-semibold leading-[1.05] tracking-tight"
        style={{ color: emotion ? color : undefined }}
      >
        {headline}
      </p>

      <div className="flex items-center gap-3 min-h-6">
        {emotion ? (
          <>
            <span className="text-base md:text-lg tabular-nums font-medium">
              {percent(confidence, 0)} confidence
            </span>
            <span
              className="h-1.5 w-28 md:w-40 rounded-full bg-foreground/10 overflow-hidden"
              aria-hidden="true"
            >
              <span
                className="block h-full rounded-full transition-[width] duration-500 ease-out"
                style={{ width: `${Math.round(confidence * 100)}%`, backgroundColor: color }}
              />
            </span>
          </>
        ) : (
          <span className="text-base text-foreground/50">{sub}</span>
        )}
      </div>

      {emotion && <p className="text-sm text-foreground/45">{sub}</p>}

      <span className="sr-only" role="status" aria-live="polite">
        {running ? headline : ''}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function DistributionBars({
  rows,
  active,
  empty,
}: {
  rows: { emotion: Emotion; value: number }[];
  active: Emotion | null;
  empty: boolean;
}) {
  if (empty || rows.length === 0) {
    return (
      <p className="text-base text-foreground/45">
        No usable face in frame — nothing is being predicted.
      </p>
    );
  }

  return (
    <ul className="flex flex-col gap-2.5">
      {rows.map((row) => (
        <li key={row.emotion} className="flex items-center gap-3">
          <span className="w-20 md:w-24 shrink-0 text-sm text-foreground/60">
            {EMOTION_LABEL[row.emotion]}
          </span>
          <span className="flex-1 h-1.5 rounded-full bg-foreground/8 overflow-hidden">
            <span
              className="block h-full rounded-full transition-[width] duration-300 ease-out"
              style={{
                width: `${Math.max(1, Math.round(row.value * 100))}%`,
                backgroundColor: emotionColor(row.emotion),
                opacity: row.emotion === active ? 1 : 0.45,
              }}
            />
          </span>
          <span className="w-12 shrink-0 text-right text-sm tabular-nums text-foreground/55">
            {percent(row.value, 0)}
          </span>
        </li>
      ))}
    </ul>
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
