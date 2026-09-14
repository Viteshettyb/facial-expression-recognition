import { Card, Reveal, Section, SectionHeader } from '../ui/primitives';
import { IconChart, IconClock, IconFace, IconReport } from '../ui/icons';

const PREVIEW = [
  {
    icon: IconFace,
    title: 'Primary face detection',
    body: 'The largest face in the frame is isolated, aligned and tracked across the whole clip.',
  },
  {
    icon: IconChart,
    title: 'Expression distribution',
    body: 'Seven-class probabilities aggregated into a dominant emotion and a percentage breakdown.',
  },
  {
    icon: IconClock,
    title: 'Temporal trajectory',
    body: 'Frame-by-frame probabilities charted over time, segmented into contiguous emotion windows.',
  },
  {
    icon: IconReport,
    title: 'Final report',
    body: 'Confidence, stability, volatility and per-frame observations in one exportable summary.',
  },
];

export function EmptyResults() {
  return (
    <Section id="results">
      <div className="w-content-width mx-auto flex flex-col gap-8 md:gap-10">
        <SectionHeader
          eyebrow="Results"
          title="What you get back"
          subtitle="Upload a clip above to populate these panels with your own analysis."
        />

        <Reveal>
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 md:gap-6">
            {PREVIEW.map((p, i) => (
              <Card key={p.title} hover className="p-6 md:p-8 flex flex-col gap-4">
                <div className="grid place-items-center size-11 rounded-token bg-primary-cta/8 text-primary-cta">
                  <p.icon className="size-5" />
                </div>
                <div className="flex flex-col gap-1.5">
                  <span className="text-xs tabular-nums text-foreground/35">
                    0{i + 1}
                  </span>
                  <h3 className="text-xl md:text-2xl font-semibold leading-tight">{p.title}</h3>
                  <p className="text-base text-foreground/55 leading-snug">{p.body}</p>
                </div>
              </Card>
            ))}
          </div>
        </Reveal>
      </div>
    </Section>
  );
}
