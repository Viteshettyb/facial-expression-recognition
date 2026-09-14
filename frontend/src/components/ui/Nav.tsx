import { useState } from 'react';
import { useScrolled } from '../../hooks/useScrolled';
import { cx } from '../../lib/format';
import { IconClose, IconPlus } from './icons';

const LINKS = [
  { href: '#upload', label: 'Upload' },
  { href: '#processing', label: 'Processing' },
  { href: '#results', label: 'Results' },
  { href: '#timeline', label: 'Timeline' },
  { href: '#report', label: 'Report' },
];

export function Nav() {
  const scrolled = useScrolled();
  const [open, setOpen] = useState(false);

  return (
    <nav
      className={cx(
        'fixed z-[1000] top-0 left-0 w-full transition-all duration-500 ease-in-out h-20',
        scrolled
          ? 'bg-background/70 backdrop-blur-md border-b border-foreground/8'
          : 'bg-background/0 backdrop-blur-0 border-b border-transparent',
      )}
    >
      <div className="relative mx-auto flex items-center justify-between h-full w-content-width">
        <a
          href="#top"
          className="flex items-center gap-2 text-xl font-medium text-foreground whitespace-nowrap"
        >
          <span className="grid place-items-center size-8 rounded-full bg-primary-cta text-primary-cta-text text-xs font-semibold tracking-[0.02em]">
            ER
          </span>
          Emotion Recognition
        </a>

        <div className="hidden md:flex absolute left-1/2 -translate-x-1/2 items-center gap-6">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="text-base text-foreground hover:opacity-70 transition-opacity"
            >
              {l.label}
            </a>
          ))}
        </div>

        <div className="flex items-center gap-2 xl:gap-3">
          <a
            href="#upload"
            className="hidden sm:flex items-center justify-center h-10 px-6 text-sm rounded-token cursor-pointer primary-button text-primary-cta-text font-medium"
          >
            Analyse a Video
          </a>
          <button
            aria-label={open ? 'Close menu' : 'Open menu'}
            onClick={() => setOpen((v) => !v)}
            className="flex md:hidden items-center justify-center shrink-0 size-9 rounded-token cursor-pointer primary-button"
          >
            {open ? (
              <IconClose className="w-1/2 h-1/2 text-primary-cta-text" />
            ) : (
              <IconPlus className="w-1/2 h-1/2 text-primary-cta-text" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile sheet */}
      <div
        className={cx(
          'md:hidden overflow-hidden transition-all duration-500 ease-in-out card border-x-0 border-b-0',
          open ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0',
        )}
      >
        <div className="w-content-width mx-auto flex flex-col py-4">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              onClick={() => setOpen(false)}
              className="py-3 text-base border-b border-foreground/6 last:border-0"
            >
              {l.label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  );
}
