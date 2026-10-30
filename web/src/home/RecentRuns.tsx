import { useEffect, useState } from "react";
import type { RunSummary } from "../api";
import { listRuns } from "../api";
import RunRow from "../history/RunRow";
import { href } from "../router";

/** The three newest runs. On error the section hides: Home must never look broken. */
export default function RecentRuns() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    listRuns(controller.signal)
      .then((r) => setRuns(r.slice(0, 3)))
      .catch((e: Error) => {
        if (e.name !== "AbortError") {
          console.warn("recent runs:", e.message);
          setFailed(true);
        }
      });
    return () => controller.abort();
  }, []);
  if (failed) return null;
  return (
    <section aria-labelledby="recent-runs" className="py-12">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <h2 id="recent-runs" className="text-2xl font-semibold tracking-tight">Recent runs</h2>
        <a href={href("/history")} className="text-sm text-cyan-300 underline-offset-4 hover:underline">All runs →</a>
      </div>
      {runs === null ? (
        <ul aria-busy="true" className="space-y-2">
          {[0, 1, 2].map((i) => <li key={i} className="h-[76px] animate-pulse rounded-xl border border-slate-800 bg-slate-900/40" />)}
        </ul>
      ) : runs.length === 0 ? (
        <p className="text-sm text-slate-400">
          No runs yet. Your analyses are saved here.{" "}
          <a className="text-cyan-300 underline underline-offset-4" href={href("/analyze")}>Analyze a video</a>
        </p>
      ) : (
        <ul className="space-y-2">{runs.map((r) => <RunRow key={r.id} run={r} />)}</ul>
      )}
    </section>
  );
}
