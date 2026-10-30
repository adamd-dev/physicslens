import { useEffect, useState } from "react";

/** Hash routes: no server rewrites, and the local app never exposes paths a crawler could index. */
export type Route =
  | { page: "home" }
  | { page: "analyze" }
  | { page: "history"; query: URLSearchParams }
  | { page: "run"; id: string };

export function parseRoute(hash: string): Route {
  const [path, search = ""] = hash.replace(/^#/, "").split("?");
  const parts = path.split("/").filter(Boolean);
  if (parts[0] === "analyze") return { page: "analyze" };
  if (parts[0] === "history" && parts[1] && /^[0-9a-f]{32}$/.test(parts[1])) return { page: "run", id: parts[1] };
  if (parts[0] === "history") return { page: "history", query: new URLSearchParams(search) };
  return { page: "home" };
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

/** Go to a route; `replace` updates the URL without a new history entry (filters, for example). */
export function navigate(path: string, replace = false): void {
  if (!replace) {
    window.location.hash = path;
    return;
  }
  window.history.replaceState(null, "", `#${path}`);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

export const href = (path: string) => `#${path}`;
