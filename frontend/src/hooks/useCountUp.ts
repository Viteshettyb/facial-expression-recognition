import { useEffect, useRef, useState } from 'react';

/** Eases a number from 0 to `target` once `active` becomes true. */
export function useCountUp(target: number, active: boolean, durationMs = 1100) {
  const [value, setValue] = useState(0);
  const frame = useRef(0);

  useEffect(() => {
    if (!active) return;

    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setValue(target);
      return;
    }

    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - p, 3);
      setValue(target * eased);
      if (p < 1) frame.current = requestAnimationFrame(tick);
    };

    frame.current = requestAnimationFrame(tick);

    // rAF is suspended on hidden pages; without this the figure would stay
    // stuck at 0 instead of showing the real value.
    const settle = window.setTimeout(() => setValue(target), durationMs + 250);

    return () => {
      cancelAnimationFrame(frame.current);
      window.clearTimeout(settle);
    };
  }, [target, active, durationMs]);

  return value;
}
