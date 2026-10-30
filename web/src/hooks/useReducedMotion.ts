import { useEffect, useState } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

/** True when the viewer asked for less motion: animations then show their final state. */
export function useReducedMotion(): boolean {
  // False until mounted: the pre-rendered landing page and the first client render must agree.
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const media = window.matchMedia?.(QUERY);
    if (!media) return;
    setReduced(media.matches);
    const onChange = () => setReduced(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);
  return reduced;
}
