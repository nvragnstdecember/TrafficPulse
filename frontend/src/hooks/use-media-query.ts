import { useEffect, useState } from 'react';

/**
 * Subscribe to a CSS media query, SSR/JSDOM-safe.
 *
 * Used for responsive behaviour (e.g. collapsing the sidebar into a drawer on
 * small screens) without hardcoding breakpoint widths in components.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, [query]);

  return matches;
}
