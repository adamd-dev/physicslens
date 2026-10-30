import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useFocusHeading } from "../hooks/useFocusHeading";
import type { RunStatus, RunSummary } from "../api";
import { deleteRun, listRuns } from "../api";
import { MODEL_LABEL, runTitle, TRACKER_LABEL } from "../history/format";
import RunRow from "../history/RunRow";
import { href, navigate } from "../router";

const STATUSES: { value: RunStatus | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "ok", label: "OK" },
  { value: "flagged", label: "Flagged" },
  { value: "none", label: "No measurement" },
];
const SORTS = { new: "Newest", old: "Oldest", dev: "Closest to 9.81", cost: "Cost" } as const;
type Sort = keyof typeof SORTS;

/** Every saved analysis. Filters live in the URL, so a filtered view survives a reload. */
export default function HistoryPage({ query, onChange }: { query: URLSearchParams; onChange?: () => void }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null); // announced after a delete, or why it failed
  const heading = useFocusHeading();
  const request = useRef<AbortController | null>(null);
  const status = (query.get("status") as RunStatus | null) ?? "all";
  const model = query.get("model") ?? "";
  const tracker = query.get("tracker") ?? "";
  const sort = (query.get("sort") as Sort | null) ?? "new";

  const load = useCallback(() => {
    request.current?.abort(); // a second Retry wins over the first
    const controller = (request.current = new AbortController());
    setError(null);
    listRuns(controller.signal)
      .then(setRuns)
      .catch((e: Error) => e.name !== "AbortError" && setError(e.message));
  }, []);
  useEffect(() => {
    load();
    return () => request.current?.abort();
  }, [load]);

  const setQuery = (key: string, value: string) => {
    const next = new URLSearchParams(query);
    if (!value || value === "all" || (key === "sort" && value === "new")) next.delete(key);
    else next.set(key, value);
    const qs = next.toString();
    navigate(`/history${qs ? `?${qs}` : ""}`, true);
  };

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: runs?.length ?? 0, ok: 0, flagged: 0, none: 0 };
    runs?.forEach((r) => (c[r.status] += 1));
    return c;
  }, [runs]);

  const shown = useMemo(() => {
    const list = (runs ?? []).filter(
      (r) => (status === "all" || r.status === status) && (!model || r.model === model) && (!tracker || r.tracker === tracker),
    );
    const dev = (r: RunSummary) => (r.dev_pct == null ? Infinity : Math.abs(r.dev_pct));
    const by: Record<Sort, (a: RunSummary, b: RunSummary) => number> = {
      new: (a, b) => b.created_at.localeCompare(a.created_at),
      old: (a, b) => a.created_at.localeCompare(b.created_at),
      dev: (a, b) => dev(a) - dev(b),
      cost: (a, b) => (b.cost_usd ?? 0) - (a.cost_usd ?? 0),
    };
    return [...list].sort(by[sort] ?? by.new);
  }, [runs, status, model, tracker, sort]);

  const remove = async (id: string) => {
    const run = runs?.find((x) => x.id === id);
    setRuns((r) => r?.filter((x) => x.id !== id) ?? null); // optimistic: back if the server says no
    heading.current?.focus(); // the focused button is gone; keep keyboard users on the page
    try {
      await deleteRun(id);
      setNotice(`Deleted ${run ? runTitle(run) : "the run"}.`);
      onChange?.();
    } catch (e) {
      if (run) setRuns((r) => (r && !r.some((x) => x.id === id) ? [...r, run] : r));
      setNotice(`Could not delete ${run ? runTitle(run) : "the run"}: ${(e as Error).message}`);
    }
  };

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 ref={heading} tabIndex={-1} className="text-2xl font-semibold tracking-tight focus:outline-none">History</h1>
          <p className="text-sm text-slate-400">
            {runs ? `${runs.length} saved ${runs.length === 1 ? "run" : "runs"}` : "Saved runs"} · stored on this machine only
          </p>
        </div>
        <a href={href("/analyze")} className="btn bg-gradient-to-r from-cyan-400 to-violet-500 font-semibold text-slate-950 hover:brightness-110">
          Analyze a video
        </a>
      </div>

      <p role="status" className={notice ? "mb-3 text-sm text-slate-300" : "sr-only"}>
        {notice ?? (runs ? `${shown.length} of ${runs.length} runs shown` : "")}
      </p>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div role="group" aria-label="Status" className="flex flex-wrap gap-1 rounded-xl border border-slate-800 p-1">
          {STATUSES.map((s) => (
            <button
              key={s.value}
              aria-pressed={status === s.value}
              className={`btn min-h-8 px-2.5 text-xs ${status === s.value ? "bg-slate-800 text-white" : "text-slate-400 hover:text-white"}`}
              onClick={() => setQuery("status", s.value)}
            >
              {s.label} <span className="readout text-slate-400">{counts[s.value]}</span>
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-sm text-slate-400">
          Model
          <select className="input w-auto" value={model} onChange={(e) => setQuery("model", e.target.value)}>
            <option value="">Any</option>
            {Object.entries(MODEL_LABEL).map(([v, l]) => (
              <option key={v} value={v}>{l}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm text-slate-400">
          Tracker
          <select className="input w-auto" value={tracker} onChange={(e) => setQuery("tracker", e.target.value)}>
            <option value="">Any</option>
            {Object.entries(TRACKER_LABEL).map(([v, l]) => (
              <option key={v} value={v}>{l}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm text-slate-400 sm:ml-auto">
          Sort
          <select className="input w-auto" value={sort} onChange={(e) => setQuery("sort", e.target.value)}>
            {Object.entries(SORTS).map(([v, l]) => (
              <option key={v} value={v}>{l}</option>
            ))}
          </select>
        </label>
      </div>

      {error ? (
        <div role="alert" className="flex flex-wrap items-center gap-3 rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
          Couldn't load runs: {error}
          <button className="btn-ghost" onClick={load}>Retry</button>
        </div>
      ) : runs === null ? (
        <ul aria-busy="true" aria-label="Loading runs" className="space-y-2">
          {Array.from({ length: 6 }, (_, i) => (
            <li key={i} className="h-[76px] animate-pulse rounded-xl border border-slate-800 bg-slate-900/40" />
          ))}
        </ul>
      ) : runs.length === 0 ? (
        <EmptyRuns />
      ) : shown.length === 0 ? (
        <div className="panel p-6 text-sm text-slate-300">
          No runs match these filters.{" "}
          <button className="text-cyan-300 underline underline-offset-4" onClick={() => navigate("/history", true)}>
            Clear filters
          </button>
        </div>
      ) : (
        <ul className="space-y-2" aria-label="Runs">
          {shown.map((run) => (
            <RunRow key={run.id} run={run} onDelete={remove} />
          ))}
        </ul>
      )}
    </div>
  );
}

/** A dotted throw on graph paper: what a run will look like. */
export function EmptyRuns() {
  return (
    <div className="instrument graph-paper flex flex-col items-center gap-3 px-6 py-12 text-center">
      <svg aria-hidden="true" width="160" height="70" viewBox="0 0 160 70">
        {Array.from({ length: 11 }, (_, i) => {
          const x = 8 + i * 14.4;
          return <circle key={i} cx={x} cy={12 + 0.0078 * (x - 80) ** 2} r="2.5" fill="var(--data)" />;
        })}
      </svg>
      <p className="max-w-md text-sm text-slate-300">No runs yet. Every analysis is saved here with its fit, the agent's steps and the video overlay.</p>
      <a href={href("/analyze")} className="btn bg-gradient-to-r from-cyan-400 to-violet-500 font-semibold text-slate-950 hover:brightness-110">
        Analyze a video
      </a>
    </div>
  );
}
