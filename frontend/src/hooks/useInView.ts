import { useEffect, useRef, useState } from 'react';

/**
 * Fires once when the element enters the viewport — drives the reveal
 * animations. Deliberately rect-based rather than IntersectionObserver so
 * content can never get stuck invisible if observer callbacks are throttled
 * (background tabs, embedded webviews, some automation contexts).
 */
export function useInView<T extends HTMLElement>(threshold = 0.15) {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    let done = false;

    const check = () => {
      const node = ref.current;
      if (done || !node) return;

      const rect = node.getBoundingClientRect();
      const vh = window.innerHeight || document.documentElement.clientHeight;
      // Visible height of the element, allowing for elements taller than the viewport.
      const visible = Math.min(rect.bottom, vh) - Math.max(rect.top, 0);
      const needed = Math.min(rect.height * threshold, vh * 0.25);

      if (visible >= needed && rect.bottom > 0 && rect.top < vh) {
        done = true;
        setInView(true);
      }
    };

    // Checked synchronously rather than inside requestAnimationFrame: rAF is
    // suspended on hidden pages, which would leave sections stuck invisible.
    check();
    window.addEventListener('scroll', check, { passive: true });
    window.addEventListener('resize', check);

    // Re-check while layout settles — fonts, media metadata, and sections that
    // mount after the initial paint (the analysis results).
    const settle = window.setInterval(() => {
      if (done) window.clearInterval(settle);
      else check();
    }, 200);
    const stopSettling = window.setTimeout(() => window.clearInterval(settle), 4000);

    return () => {
      window.clearInterval(settle);
      window.clearTimeout(stopSettling);
      window.removeEventListener('scroll', check);
      window.removeEventListener('resize', check);
    };
  }, [threshold]);

  return { ref, inView };
}
