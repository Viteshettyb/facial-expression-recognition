/**
 * Decorative facial-landmark mesh. Purely visual — a stylised stand-in for the
 * detector's landmark output, used as hero and panel artwork.
 */

const LANDMARKS: Array<[number, number]> = [
  // jaw
  [86, 138], [90, 165], [97, 192], [107, 217], [122, 239], [141, 257], [163, 271],
  [186, 279], [200, 281], [214, 279], [237, 271], [259, 257], [278, 239], [293, 217],
  [303, 192], [310, 165], [314, 138],
  // brows
  [110, 116], [127, 104], [148, 101], [169, 105], [188, 113],
  [212, 113], [231, 105], [252, 101], [273, 104], [290, 116],
  // nose
  [200, 133], [200, 152], [200, 171], [200, 190],
  [178, 203], [189, 207], [200, 210], [211, 207], [222, 203],
  // eyes
  [131, 137], [144, 130], [158, 130], [170, 138], [157, 143], [143, 143],
  [230, 138], [242, 130], [256, 130], [269, 137], [257, 143], [243, 143],
  // mouth
  [162, 231], [176, 222], [190, 217], [200, 220], [210, 217], [224, 222], [238, 231],
  [225, 244], [211, 250], [200, 251], [189, 250], [175, 244],
];

const EDGES: Array<[number, number]> = [];
for (let i = 0; i < LANDMARKS.length; i += 1) {
  // Connect each point to its 3 nearest neighbours for a triangulated look.
  const dists = LANDMARKS.map((p, j) => ({
    j,
    d: (p[0] - LANDMARKS[i][0]) ** 2 + (p[1] - LANDMARKS[i][1]) ** 2,
  }))
    .filter((x) => x.j !== i)
    .sort((a, b) => a.d - b.d)
    .slice(0, 3);
  dists.forEach((x) => {
    if (i < x.j) EDGES.push([i, x.j]);
  });
}

export function FaceMesh({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 400 320" className={className} aria-hidden="true">
      <defs>
        <linearGradient id="mesh-stroke" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.85" />
          <stop offset="100%" stopColor="var(--background-accent)" stopOpacity="0.25" />
        </linearGradient>
        <radialGradient id="mesh-glow" cx="50%" cy="45%" r="55%">
          <stop offset="0%" stopColor="var(--background-accent)" stopOpacity="0.22" />
          <stop offset="100%" stopColor="var(--background-accent)" stopOpacity="0" />
        </radialGradient>
      </defs>

      <ellipse cx="200" cy="180" rx="150" ry="150" fill="url(#mesh-glow)" />

      <g stroke="url(#mesh-stroke)" strokeWidth="0.85" fill="none">
        {EDGES.map(([a, b], i) => (
          <line
            key={i}
            x1={LANDMARKS[a][0]}
            y1={LANDMARKS[a][1]}
            x2={LANDMARKS[b][0]}
            y2={LANDMARKS[b][1]}
          />
        ))}
      </g>

      <g fill="var(--accent)">
        {LANDMARKS.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={i % 7 === 0 ? 2.4 : 1.4} opacity={i % 7 === 0 ? 0.9 : 0.5}>
            {i % 11 === 0 && (
              <animate
                attributeName="opacity"
                values="0.25;0.95;0.25"
                dur={`${3 + (i % 5) * 0.6}s`}
                repeatCount="indefinite"
              />
            )}
          </circle>
        ))}
      </g>
    </svg>
  );
}
