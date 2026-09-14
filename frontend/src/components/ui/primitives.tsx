import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { useInView } from '../../hooks/useInView';
import { cx } from '../../lib/format';

/* ------------------------------------------------------------------ */
/* Layout                                                              */
/* ------------------------------------------------------------------ */

export function Container({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={cx('w-content-width mx-auto', className)}>{children}</div>;
}

export function Section({
  id,
  children,
  className,
}: {
  id?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={cx('py-16 md:py-20 scroll-mt-24', className)}>
      {children}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Reveal-on-scroll                                                    */
/* ------------------------------------------------------------------ */

export function Reveal({
  children,
  delay = 0,
  className,
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const { ref, inView } = useInView<HTMLDivElement>();
  return (
    <div
      ref={ref}
      className={cx('reveal', className)}
      data-shown={inView}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

/** Word-by-word staggered reveal, matching the reference's headline motion. */
export function RevealText({
  text,
  as: Tag = 'h2',
  className,
  stagger = 40,
}: {
  text: string;
  as?: 'h1' | 'h2' | 'h3' | 'p';
  className?: string;
  stagger?: number;
}) {
  const { ref, inView } = useInView<HTMLHeadingElement>();
  const words = text.split(' ');

  return (
    <Tag className={className} ref={ref}>
      {words.map((word, i) => (
        <span key={`${word}-${i}`}>
          {i > 0 ? ' ' : ''}
          <span
            className="reveal-word"
            data-shown={inView}
            style={{ transitionDelay: `${i * stagger}ms` }}
          >
            {word}
          </span>
        </span>
      ))}
    </Tag>
  );
}

/* ------------------------------------------------------------------ */
/* Surfaces                                                            */
/* ------------------------------------------------------------------ */

export function Badge({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cx('w-fit px-3 py-1 text-sm card rounded-token whitespace-nowrap', className)}
    >
      {children}
    </div>
  );
}

export function Card({
  children,
  className,
  solid = false,
  hover = false,
}: {
  children: ReactNode;
  className?: string;
  solid?: boolean;
  hover?: boolean;
}) {
  return (
    <div
      className={cx(
        solid ? 'card-solid' : 'card',
        hover && 'card-hover',
        'rounded-token',
        className,
      )}
    >
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Buttons                                                             */
/* ------------------------------------------------------------------ */

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
};

export function Button({
  variant = 'primary',
  size = 'md',
  className,
  children,
  ...rest
}: ButtonProps) {
  const sizes = {
    sm: 'h-9 px-4 text-xs',
    md: 'h-10 md:h-11 px-6 text-sm',
    lg: 'h-12 md:h-13 px-8 text-base',
  };
  const variants = {
    primary: 'primary-button text-primary-cta-text',
    secondary: 'secondary-button',
    ghost:
      'border border-foreground/12 text-foreground/70 hover:text-foreground hover:border-foreground/25 transition-colors',
  };

  return (
    <button
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-token cursor-pointer font-medium select-none',
        sizes[size],
        variants[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* Section header (badge + headline + subtitle)                        */
/* ------------------------------------------------------------------ */

export function SectionHeader({
  eyebrow,
  title,
  subtitle,
  align = 'center',
}: {
  eyebrow: string;
  title: string;
  subtitle?: string;
  align?: 'center' | 'left';
}) {
  const centered = align === 'center';
  return (
    <div
      className={cx(
        'flex flex-col gap-2',
        centered ? 'items-center text-center' : 'items-start text-left',
      )}
    >
      <Reveal>
        <Badge>{eyebrow}</Badge>
      </Reveal>
      <RevealText
        as="h2"
        text={title}
        className={cx(
          'text-5xl md:text-6xl 2xl:text-7xl leading-[1.15] font-semibold text-balance mt-1',
          centered && 'md:max-w-8/10 mx-auto',
        )}
      />
      {subtitle && (
        <RevealText
          as="p"
          text={subtitle}
          stagger={16}
          className={cx(
            'text-lg md:text-xl leading-snug text-balance text-foreground/65',
            centered && 'md:max-w-7/10 mx-auto',
          )}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Misc                                                                */
/* ------------------------------------------------------------------ */

export function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs uppercase tracking-widest text-foreground/55">{label}</span>
      <span className="text-3xl md:text-4xl font-semibold leading-none">{value}</span>
      {hint && <span className="text-xs text-foreground/50">{hint}</span>}
    </div>
  );
}

export function Divider({ className }: { className?: string }) {
  return <div className={cx('h-px w-full bg-foreground/8', className)} />;
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={cx('spin size-4', className)} aria-hidden="true">
      <circle
        cx="12"
        cy="12"
        r="9"
        fill="none"
        stroke="currentColor"
        strokeOpacity="0.2"
        strokeWidth="2.5"
      />
      <path
        d="M21 12a9 9 0 0 0-9-9"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
    </svg>
  );
}
