import { lazy, Suspense, useEffect, useState } from "react";
import type { RunRecord, VideoInfo } from "../api";
import { deleteRun, getRun, videoInfo } from "../api";
import AgentTimeline from "../components/AgentTimeline";
import ResultCard from "../components/ResultCard";
import StatusBadge from "../components/StatusBadge";
import VideoStage from "../components/VideoStage";
import { MODEL_LABEL, TRACKER_LABEL, runTitle } from "../history/format";
import { href, navigate } from "../router";
import { applyEvent, EMPTY_RUN, type RunState } from "../runState";

const Charts = lazy(() => import("../components/Charts"));
const noop = () => {};

/** A saved run, rebuilt by replaying its events through the same handler the live page uses. */
export default function RunDetailPage({ id, onDeleted }: { id: string; onDeleted?: () => void }) {
  const [record, setRecord] = useState<RunRecord | null>(null);
  const [state, setState] = useState<RunState>(EMPTY_RUN);
  const [info, setInfo] = useState<VideoInfo | null | undefined>(undefined); // null: the video is gone
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setRecord(null);
    setError(null);
    getRun(id, controller.signal)
      .then((r) => {
        setRecord(r);
        setState(r.events.reduce(applyEvent, EMPTY_RUN));
        return videoInfo(r.video_id, controller.signal).then(setInfo);
      })
      .catch((e: Error) => e.name !== "AbortError" && setError(e.message));
    return () => controller.abort();
  }, [id]);

  if (error) {
    return (
      <div className="panel p-6">
        <h1 tabIndex={-1} className="text-lg font-semibold focus:outline-none">Run not found</h1>
        <p className="mt-1 text-sm text-slate-400">It may have been deleted. ({error})</p>
        <a className="mt-3 inline-block text-cyan-300 underline underline-offset-4" href={href("/history")}>← All runs</a>
      </div>
    );
  }
  if (!record) return <div aria-busy="true" className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_400px]"><div className="panel h-96 animate-pulse" /><div className="panel h-96 animate-pulse" /></div>;

  const { summary, request } = record;
  const { track, result, messages, cost, warnings } = state;
  const points = [request.scale.p1, request.scale.p2];
  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <a href={href("/history")} className="text-sm text-slate-400 hover:text-white">← All runs</a>
          <h1 tabIndex={-1} className="mt-1 flex flex-wrap items-center gap-3 text-xl font-semibold focus:outline-none">
            <span className="truncate">{runTitle(summary)}</span>
            <StatusBadge status={summary.status} />
          </h1>
          <p className="text-sm text-slate-400">
            <time dateTime={summary.created_at}>{new Date(summary.created_at).toLocaleString()}</time> · tracked by{" "}
            {TRACKER_LABEL[summary.tracker] ?? summary.tracker} · {request.agent ? `Muse agent${summary.steps ? `, ${summary.steps} steps` : ""}` : "deterministic fit"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {confirming ? (
            <>
              <button
                className="btn border border-rose-500/50 text-rose-300 hover:bg-rose-500/10"
                onClick={() =>
                  deleteRun(id)
                    .then(() => {
                      onDeleted?.();
                      navigate("/history");
                    })
                    .catch((e: Error) => setError(e.message))
                }
              >
                Delete run
              </button>
              <button autoFocus className="btn-ghost" onClick={() => setConfirming(false)}>Cancel</button>
            </>
          ) : (
            <button className="btn-ghost" onClick={() => setConfirming(true)}>Delete</button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px] xl:grid-cols-[minmax(0,1fr)_400px]">
        <section aria-label="Video and charts" className="space-y-4">
          <div className="panel p-4">
            {info === undefined ? (
              <div className="aspect-video animate-pulse rounded-lg bg-slate-800" />
            ) : info === null ? (
              <div className="graph-paper grid aspect-video place-items-center rounded-lg text-sm text-slate-400">The video of this run is no longer on this machine.</div>
            ) : (
              <VideoStage
                info={info} mode="view" points={points} lengthM={request.scale.length_m} roi={request.settings.roi} ballBox={request.settings.sam_box}
                startS={request.start} track={track} result={result} onPoints={noop} onRoi={noop} onBallBox={noop}
              />
            )}
          </div>
          {track && result && (
            <Suspense fallback={<div className="panel h-[262px]" />}>
              <Charts track={track} result={result} />
            </Suspense>
          )}
          <details className="panel p-4 text-sm">
            <summary className="cursor-pointer font-medium text-slate-200">Request</summary>
            <dl className="readout mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-slate-300">
              <dt className="text-slate-400">tracker</dt><dd>{TRACKER_LABEL[request.settings.tracker] ?? request.settings.tracker}</dd>
              <dt className="text-slate-400">scale</dt><dd>{request.scale.length_m} m between ({points.map((p) => p.map(Math.round).join(", ")).join(") and (")})</dd>
              <dt className="text-slate-400">window</dt><dd>{request.start} s to {request.end ?? "end"}{request.capture_fps ? ` · captured at ${request.capture_fps} fps` : ""}</dd>
              <dt className="text-slate-400">model</dt><dd>{request.model === "auto" ? "auto" : MODEL_LABEL[request.model] ?? request.model} · axis {request.axis}</dd>
              {request.settings.roi && (<><dt className="text-slate-400">search area</dt><dd>{request.settings.roi.join(", ")}</dd></>)}
              {request.settings.sam_box && (<><dt className="text-slate-400">ball box</dt><dd>{request.settings.sam_box.join(", ")}</dd></>)}
            </dl>
          </details>
        </section>

        <aside aria-label="Result" className="space-y-4">
          {state.error && (
            <div role="alert" className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">{state.error}</div>
          )}
          {warnings.map((w, i) => (
            <div key={`${i}-${w}`} className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-200">{w}</div>
          ))}
          {result && <ResultCard result={result} cost={cost} />}
          {messages.length > 0 && (
            <div className="panel p-4">
              <h2 className="label mb-4">Agent reasoning · saved</h2>
              <AgentTimeline messages={messages} running={false} />
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
