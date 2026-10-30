import { useEffect, useRef } from "react";

/** Focus the page heading once it exists, as a page load would announce it. `ready`: the heading is rendered. */
export function useFocusHeading<T extends HTMLElement = HTMLHeadingElement>(ready = true) {
  const ref = useRef<T>(null);
  useEffect(() => {
    if (ready) ref.current?.focus({ preventScroll: true });
  }, [ready]);
  return ref;
}
