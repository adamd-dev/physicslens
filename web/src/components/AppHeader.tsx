import type { Route } from "../router";
import { href } from "../router";

const TABS = [
  { page: "home", label: "Home", path: "/" },
  { page: "analyze", label: "Analyze", path: "/analyze" },
  { page: "history", label: "History", path: "/history" },
] as const;

/** The same header on every page: where you are, and whether a run is still going on Analyze. */
export default function AppHeader({ route, running, runCount }: { route: Route; running: boolean; runCount: number | null }) {
  const current = route.page === "run" ? "history" : route.page;
  return (
    <header className="sticky top-0 z-20 -mx-5 mb-6 border-b border-slate-800 bg-slate-950/80 px-5 py-3 backdrop-blur">
      <a href="#main" className="sr-only rounded-lg bg-cyan-400 px-3 py-2 font-semibold text-slate-950 focus:not-sr-only focus:absolute focus:left-5 focus:top-3">
        Skip to content
      </a>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <a href={href("/")} className="flex items-center gap-3 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400">
          <span aria-hidden="true" className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-cyan-400 to-violet-500 text-lg font-bold text-slate-950">ϕ</span>
          <span className="text-lg font-semibold leading-tight">PhysicsLens</span>
        </a>
        <nav aria-label="Main">
          <ul className="flex items-center gap-1 text-sm">
            {TABS.map((tab) => {
              const active = current === tab.page;
              return (
                <li key={tab.page}>
                  <a
                    href={href(tab.path)}
                    aria-current={active ? "page" : undefined}
                    className={`relative inline-flex min-h-10 items-center gap-1.5 rounded-lg px-3 font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 ${
                      active ? "text-white underline decoration-cyan-400 decoration-2 underline-offset-[10px]" : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {tab.label}
                    {tab.page === "analyze" && running && !active && (
                      <>
                        <span aria-hidden="true" className="h-2 w-2 animate-pulse rounded-full bg-cyan-400" />
                        <span className="sr-only">(analysis running)</span>
                      </>
                    )}
                    {tab.page === "history" && runCount != null && runCount > 0 && (
                      <span className="rounded-full bg-slate-800 px-1.5 font-mono text-xs text-slate-300">
                        {runCount}
                        <span className="sr-only"> runs</span>
                      </span>
                    )}
                  </a>
                </li>
              );
            })}
          </ul>
        </nav>
      </div>
    </header>
  );
}
