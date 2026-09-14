import { useMemo, useState } from 'react';
import {
  EMOTION_COLOR,
  EMOTION_LABEL,
  type Emotion,
  type EmotionScore,
  type EmotionSegment,
  type TimelinePoint,
} from '../../types/analysis';
import { useInView } from '../../hooks/useInView';
import { useMediaQuery } from '../../hooks/useMediaQuery';
import { percent } from '../../lib/format';

/* ------------------------------------------------------------------ */
/* Donut — emotion distribution                                        */
/* ------------------------------------------------------------------ */

export function DonutChart({
  data,
  size = 220,
  thickness = 26,
  centerLabel,
  centerValue,
}: {
  data: EmotionScore[];
  size?: number;
  thickness?: number;
  centerLabel: string;
  centerValue: string;
}) {
  const { ref, inView } = useInView<HTMLDivElement>(0.3);
  const [active, setActive] = useState<Emotion | null>(null);

  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;

  let offset = 0;
  const arcs = data.map((d) => {
    const len = d.value * c;
    const arc = { ...d, len, offset };
    offset += len;
    return arc;
  });

  return (
    <div ref={ref} className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.06}
          strokeWidth={thickness}
        />
        {arcs.map((a) => (
          <circle
            key={a.emotion}
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={EMOTION_COLOR[a.emotion]}
            strokeWidth={active === a.emotion ? thickness + 6 : thickness}
            strokeLinecap="butt"
            strokeDasharray={`${inView ? a.len : 0} ${c}`}
            strokeDashoffset={-a.offset}
            opacity={active && active !== a.emotion ? 0.28 : 1}
            style={{
              transition:
                'stroke-dasharray 1s cubic-bezier(0.4,0,0.2,1), opacity 0.3s ease, stroke-width 0.3s ease',
              transitionDelay: 'stroke-dasharray 0.1s',
            }}
            onMouseEnter={() => setActive(a.emotion)}
            onMouseLeave={() => setActive(null)}
          />
        ))}
      </svg>

      <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
        <span className="text-2xs uppercase tracking-widest text-foreground/55">
          {active ? EMOTION_LABEL[active] : centerLabel}
        </span>
        <span className="text-3xl md:text-4xl font-semibold leading-tight">
          {active
            ? percent(data.find((d) => d.emotion === active)?.value ?? 0)
            : centerValue}
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Horizontal bars — distribution list                                 */
/* ------------------------------------------------------------------ */

export function EmotionBars({ data }: { data: EmotionScore[] }) {
  const { ref, inView } = useInView<HTMLDivElement>(0.2);
  const max = Math.max(...data.map((d) => d.value), 0.0001);

  return (
    <div ref={ref} className="flex flex-col gap-3 w-full">
      {data.map((d, i) => (
        <div key={d.emotion} className="flex items-center gap-3">
          <span className="text-sm text-foreground/70 w-20 shrink-0">
            {EMOTION_LABEL[d.emotion]}
          </span>
          <div className="relative flex-1 h-2.5 rounded-full bg-foreground/6 overflow-hidden">
            <div
              className="absolute inset-y-0 left-0 rounded-full"
              style={{
                width: inView ? `${(d.value / max) * 100}%` : '0%',
                background: EMOTION_COLOR[d.emotion],
                transition: 'width 0.9s cubic-bezier(0.4,0,0.2,1)',
                transitionDelay: `${i * 70}ms`,
              }}
            />
          </div>
          <span className="text-sm tabular-nums w-14 text-right text-foreground/80">
            {percent(d.value)}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Temporal area chart — emotion over time                             */
/* ------------------------------------------------------------------ */

export function TemporalChart({
  timeline,
  visible,
}: {
  timeline: TimelinePoint[];
  visible: Emotion[];
}) {
  const { ref, inView } = useInView<HTMLDivElement>(0.2);
  const [hover, setHover] = useState<number | null>(null);
  const narrow = useMediaQuery('(max-width: 768px)');

  const W = 1000;
  // A taller viewBox on narrow screens keeps the plot from collapsing to a strip.
  const H = narrow ? 560 : 260;
  const padT = 16;
  const padB = narrow ? 44 : 28;
  const padL = narrow ? 56 : 34;
  const plotH = H - padT - padB;
  const plotW = W - padL - 8;

  const duration = timeline[timeline.length - 1]?.t || 1;

  const paths = useMemo(
    () =>
      visible.map((emotion) => {
        const pts = timeline.map((p, i) => {
          const x = padL + (i / (timeline.length - 1)) * plotW;
          const y = padT + (1 - p.scores[emotion]) * plotH;
          return [x, y] as const;
        });

        // Smooth with a simple cardinal-ish curve.
        let d = `M ${pts[0][0]} ${pts[0][1]}`;
        for (let i = 0; i < pts.length - 1; i += 1) {
          const [x0, y0] = pts[i];
          const [x1, y1] = pts[i + 1];
          const mx = (x0 + x1) / 2;
          d += ` C ${mx} ${y0}, ${mx} ${y1}, ${x1} ${y1}`;
        }
        const area = `${d} L ${pts[pts.length - 1][0]} ${padT + plotH} L ${pts[0][0]} ${
          padT + plotH
        } Z`;
        return { emotion, d, area };
      }),
    [timeline, visible, plotH, plotW],
  );

  const hoverPoint = hover !== null ? timeline[hover] : null;

  return (
    <div ref={ref} className="w-full">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ height: 'auto' }}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = ((e.clientX - rect.left) / rect.width) * W;
          const i = Math.round(((x - padL) / plotW) * (timeline.length - 1));
          setHover(Math.max(0, Math.min(timeline.length - 1, i)));
        }}
      >
        <defs>
          {visible.map((emotion) => (
            <linearGradient
              key={emotion}
              id={`grad-${emotion}`}
              x1="0"
              y1="0"
              x2="0"
              y2="1"
            >
              <stop offset="0%" stopColor={EMOTION_COLOR[emotion]} stopOpacity="0.22" />
              <stop offset="100%" stopColor={EMOTION_COLOR[emotion]} stopOpacity="0" />
            </linearGradient>
          ))}
        </defs>

        {/* gridlines */}
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line
              x1={padL}
              x2={W - 8}
              y1={padT + g * plotH}
              y2={padT + g * plotH}
              stroke="currentColor"
              strokeOpacity={0.07}
              strokeWidth={1}
            />
            <text
              x={padL - 8}
              y={padT + g * plotH + 4}
              textAnchor="end"
              fontSize={narrow ? 22 : 11}
              fill="currentColor"
              fillOpacity={0.35}
            >
              {Math.round((1 - g) * 100)}
            </text>
          </g>
        ))}

        {paths.map(({ emotion, d, area }, idx) => (
          <g key={emotion} opacity={inView ? 1 : 0} style={{ transition: 'opacity .6s ease' }}>
            <path d={area} fill={`url(#grad-${emotion})`} />
            <path
              d={d}
              fill="none"
              stroke={EMOTION_COLOR[emotion]}
              strokeWidth={narrow ? 3.5 : 2}
              strokeLinecap="round"
              strokeDasharray={4000}
              strokeDashoffset={inView ? 0 : 4000}
              style={{
                transition: 'stroke-dashoffset 1.6s cubic-bezier(0.4,0,0.2,1)',
                transitionDelay: `${idx * 90}ms`,
              }}
            />
          </g>
        ))}

        {/* x axis ticks */}
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <text
            key={g}
            x={padL + g * plotW}
            y={H - 8}
            textAnchor={g === 0 ? 'start' : g === 1 ? 'end' : 'middle'}
            fontSize={narrow ? 22 : 11}
            fill="currentColor"
            fillOpacity={0.35}
          >
            {(g * duration).toFixed(1)}s
          </text>
        ))}

        {hoverPoint && hover !== null && (
          <g>
            <line
              x1={padL + (hover / (timeline.length - 1)) * plotW}
              x2={padL + (hover / (timeline.length - 1)) * plotW}
              y1={padT}
              y2={padT + plotH}
              stroke="currentColor"
              strokeOpacity={0.25}
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            {visible.map((emotion) => (
              <circle
                key={emotion}
                cx={padL + (hover / (timeline.length - 1)) * plotW}
                cy={padT + (1 - hoverPoint.scores[emotion]) * plotH}
                r={narrow ? 7 : 4}
                fill={EMOTION_COLOR[emotion]}
                stroke="var(--card)"
                strokeWidth={2}
              />
            ))}
          </g>
        )}
      </svg>

      <div className="h-14 mt-2">
        {hoverPoint ? (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
            <span className="tabular-nums text-foreground/50">
              {hoverPoint.t.toFixed(1)}s
            </span>
            {visible.map((emotion) => (
              <span key={emotion} className="inline-flex items-center gap-1.5 text-foreground/70">
                <span
                  className="size-2 rounded-full"
                  style={{ background: EMOTION_COLOR[emotion] }}
                />
                {EMOTION_LABEL[emotion]}
                <span className="tabular-nums font-medium text-foreground">
                  {percent(hoverPoint.scores[emotion], 0)}
                </span>
              </span>
            ))}
          </div>
        ) : (
          <p className="text-sm text-foreground/40">
            Hover the chart to inspect per-frame probabilities.
          </p>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Segment ribbon — dominant emotion across the clip                   */
/* ------------------------------------------------------------------ */

export function SegmentRibbon({
  segments,
  duration,
}: {
  segments: EmotionSegment[];
  duration: number;
}) {
  const { ref, inView } = useInView<HTMLDivElement>(0.3);

  return (
    <div ref={ref} className="w-full">
      <div className="flex h-11 w-full rounded-token overflow-hidden border border-foreground/8">
        {segments.map((s, i) => {
          const w = ((s.endSec - s.startSec) / duration) * 100;
          return (
            <div
              key={`${s.emotion}-${i}`}
              title={`${EMOTION_LABEL[s.emotion]} · ${s.startSec.toFixed(1)}s – ${s.endSec.toFixed(
                1,
              )}s`}
              className="group relative h-full transition-[flex-basis] duration-700 ease-out"
              style={{
                flexBasis: inView ? `${w}%` : '0%',
                background: EMOTION_COLOR[s.emotion],
                transitionDelay: `${i * 80}ms`,
              }}
            >
              <span className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity bg-black/12" />
            </div>
          );
        })}
      </div>
      <div className="flex justify-between mt-2 text-xs text-foreground/40 tabular-nums">
        <span>0.0s</span>
        <span>{duration.toFixed(1)}s</span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Confidence gauge                                                    */
/* ------------------------------------------------------------------ */

export function ConfidenceGauge({ value, size = 190 }: { value: number; size?: number }) {
  const { ref, inView } = useInView<HTMLDivElement>(0.3);
  const thickness = 14;
  const r = (size - thickness) / 2;
  const sweep = 0.72; // fraction of circle used
  const c = 2 * Math.PI * r;
  const track = c * sweep;

  return (
    <div ref={ref} className="relative" style={{ width: size, height: size * 0.78 }}>
      <svg width={size} height={size} style={{ transform: 'rotate(140deg)' }}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.08}
          strokeWidth={thickness}
          strokeLinecap="round"
          strokeDasharray={`${track} ${c}`}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={thickness}
          strokeLinecap="round"
          strokeDasharray={`${inView ? track * value : 0} ${c}`}
          style={{ transition: 'stroke-dasharray 1.2s cubic-bezier(0.4,0,0.2,1)' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center pt-2">
        <span className="text-4xl md:text-5xl font-semibold leading-none tabular-nums">
          {percent(value, 1)}
        </span>
        <span className="text-2xs uppercase tracking-widest text-foreground/55 mt-2">
          Confidence
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Sparkline — per-frame confidence                                    */
/* ------------------------------------------------------------------ */

export function Sparkline({
  values,
  color = 'var(--accent)',
  height = 40,
}: {
  values: number[];
  color?: string;
  height?: number;
}) {
  const W = 200;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * W;
    const y = (1 - v) * height;
    return `${x},${y}`;
  });
  return (
    <svg viewBox={`0 0 ${W} ${height}`} className="w-full" style={{ height }}>
      <polyline
        points={pts.join(' ')}
        fill="none"
        stroke={color}
        strokeWidth={1.75}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
