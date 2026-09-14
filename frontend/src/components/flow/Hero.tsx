import { Badge, RevealText, Reveal } from '../ui/primitives';
import { IconArrowDown } from '../ui/icons';
import { FaceMesh } from '../ui/FaceMesh';

const HIGHLIGHTS = [
  { value: '7', label: 'Expression classes' },
  { value: '30 fps', label: 'Frame sampling rate' },
  { value: '±0.4s', label: 'Temporal resolution' },
];

export function Hero() {
  return (
    <section
      id="top"
      className="relative isolate w-full min-h-svh overflow-hidden flex flex-col justify-center mb-8 md:mb-16"
    >
      {/* Backdrop */}
      <div className="absolute inset-0 -z-10">
        <div
          className="absolute inset-0"
          style={{
            background:
              'radial-gradient(90% 62% at 62% 8%, color-mix(in srgb, var(--background-accent) 20%, transparent), transparent 68%), radial-gradient(70% 55% at 8% 78%, color-mix(in srgb, var(--accent) 10%, transparent), transparent 70%)',
          }}
        />
        <FaceMesh className="absolute right-[-8%] top-1/2 -translate-y-1/2 w-[62vw] max-w-[880px] opacity-[0.55] mix-blend-screen hidden md:block" />
        <FaceMesh className="absolute left-1/2 -translate-x-1/2 top-[58%] w-[105vw] opacity-30 mix-blend-screen md:hidden" />
      </div>

      <div className="w-content-width mx-auto flex flex-col items-start gap-5 pt-24">
        <Reveal>
          <Badge>Deep-learning facial expression recognition</Badge>
        </Reveal>

        <RevealText
          as="h1"
          text="Read the emotion behind every frame."
          className="text-6xl md:text-7xl 2xl:text-8xl leading-[1.1] font-semibold text-balance md:max-w-[16ch]"
        />

        <RevealText
          as="p"
          text="Upload a video of a person speaking to camera. Emotion Recognition detects the primary face, classifies expression frame by frame, and returns a temporal emotion profile with a full analysis report."
          stagger={14}
          className="text-lg md:text-xl leading-snug text-foreground/65 text-balance max-w-[62ch]"
        />

        <Reveal delay={200} className="flex flex-wrap items-center gap-3 mt-2">
          <a
            href="#upload"
            className="inline-flex items-center justify-center gap-2 h-12 md:h-13 px-8 text-base rounded-token cursor-pointer primary-button text-primary-cta-text font-medium"
          >
            Start Analysis
            <IconArrowDown className="size-4" />
          </a>
          <a
            href="#results"
            className="inline-flex items-center justify-center h-12 md:h-13 px-8 text-base rounded-token cursor-pointer secondary-button font-medium"
          >
            See Sample Output
          </a>
        </Reveal>

        <Reveal delay={320} className="w-full">
          <div className="flex flex-wrap gap-3 mt-6 md:mt-10">
            {HIGHLIGHTS.map((h) => (
              <div key={h.label} className="card rounded-token px-5 py-4 min-w-[150px]">
                <p className="text-2xl md:text-3xl font-semibold leading-none">{h.value}</p>
                <p className="text-xs text-foreground/50 mt-1.5">{h.label}</p>
              </div>
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  );
}
