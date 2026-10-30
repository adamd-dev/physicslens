import { useEffect, useRef, useState } from "react";
import type { RunSummary } from "../api";
import StatusBadge from "../components/StatusBadge";
import { href } from "../router";
import { costText, keyText, relativeTime, runTitle, TRACKER_LABEL } from "./format";

/** One saved run: thumbnail, what was measured, status. The whole row opens the run; delete sits outside the link. */
export default function RunRow({ run, onDelete }: { run: RunSummary; onDelete?: (id: string) => void }) {
  const [thumbOk, setThumbOk] = useState(true);
  const [confirming, setConfirming] = useState(false);
  const trash = useRef<HTMLButtonElement>(null);
  const asked = useRef(false);
  useEffect(() => {
    if (!confirming && asked.current) trash.current?.focus(); // back to the button after Cancel
    asked.current = confirming;
  }, [confirming]);
  const measured = run.key && run.key.value != null;
  return (
    <li className="group relative flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/40 p-2 pr-3 transition hover:border-slate-600 focus-within:border-slate-600">
      <div className="relative aspect-video w-24 shrink-0 overflow-hidden rounded-lg bg-slate-800 sm:w-28">
        {thumbOk ? (
          <img
            src={`/api/videos/${run.video_id}/frame?t=${run.thumb_t}`}
            alt=""
            loading="lazy"
            width={112}
            height={63}
            className="h-full w-full object-cover"
            onError={() => setThumbOk(false)}
          />
        ) : (
          <span className="grid h-full place-items-center text-xs text-slate-400">video gone</span>
        )}
      </div>
      <div className="min-w-0 flex-1">
        <a href={href(`/history/${run.id}`)} title={runTitle(run)} className="block truncate font-medium text-slate-100 after:absolute after:inset-0 after:rounded-xl focus-visible:outline-none focus-visible:after:ring-2 focus-visible:after:ring-cyan-400">
          {runTitle(run)}
        </a>
        <div className={`readout truncate text-sm ${measured ? "text-cyan-200" : "text-amber-200"}`}>{keyText(run)}</div>
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-400 sm:hidden">
          <StatusBadge status={run.status} />
          <span>{TRACKER_LABEL[run.tracker] ?? run.tracker}</span>
          <span className="readout">{costText(run.cost_usd)}</span>
        </div>
      </div>
      <div className="hidden shrink-0 items-center gap-3 text-xs text-slate-400 sm:flex">
        <StatusBadge status={run.status} />
        <span className="w-14 text-right">{TRACKER_LABEL[run.tracker] ?? run.tracker}{run.agent ? " · agent" : ""}</span>
        <time dateTime={run.created_at} title={new Date(run.created_at).toLocaleString()} className="w-24 text-right">
          {relativeTime(run.created_at)}
        </time>
        <span className="readout w-20 text-right">{costText(run.cost_usd)}</span>
      </div>
      {onDelete && (
        <div className="relative z-10 flex shrink-0 items-center gap-1">
          {confirming ? (
            <>
              <button
                className="btn min-h-8 px-2 text-xs text-rose-300 hover:bg-rose-500/10"
                aria-label={`Confirm delete ${runTitle(run)}`}
                onClick={() => onDelete(run.id)}
              >
                Delete
              </button>
              {/* Focus goes to Cancel: the safe choice is the default. */}
              <button autoFocus className="btn min-h-8 px-2 text-xs text-slate-300" aria-label={`Cancel delete ${runTitle(run)}`} onClick={() => setConfirming(false)}>
                Cancel
              </button>
            </>
          ) : (
            <button ref={trash} className="btn min-h-8 px-2 text-slate-400 hover:text-rose-300" aria-label={`Delete run ${runTitle(run)}`} onClick={() => setConfirming(true)}>
              <svg aria-hidden="true" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M2.5 4h11M6 4V2.5h4V4M4 4l.7 9.5h6.6L12 4" /></svg>
            </button>
          )}
        </div>
      )}
    </li>
  );
}
