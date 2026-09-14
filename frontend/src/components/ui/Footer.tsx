import { apiConfig } from '../../api/client';
import { Badge } from './primitives';

export function Footer() {
  return (
    <footer className="relative mt-8 pb-10 pt-16 overflow-hidden ambient">
      <div className="w-content-width mx-auto flex flex-col gap-10">
        <div className="flex flex-wrap items-end justify-between gap-8">
          <div className="flex flex-col gap-3 max-w-[46ch]">
            <div className="flex items-center gap-2 text-xl font-medium whitespace-nowrap">
              <span className="grid place-items-center size-8 rounded-full bg-primary-cta text-primary-cta-text text-xs font-semibold tracking-[0.02em]">
                ER
              </span>
              Emotion Recognition
            </div>
            <p className="text-base text-foreground/55 leading-snug">
              Video-based facial expression recognition and emotion analysis using deep learning.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <Badge>{apiConfig.baseUrl}</Badge>
            <Badge>Live FastAPI</Badge>
          </div>
        </div>

        <div className="h-px w-full bg-foreground/8" />

        <div className="flex flex-wrap items-center justify-between gap-4 text-sm text-foreground/55">
          <p>Videos are previewed locally in the browser and are never stored by this interface.</p>
          <p>© {new Date().getFullYear()} Emotion Recognition</p>
        </div>
      </div>
    </footer>
  );
}
