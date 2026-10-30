import { lazy, Suspense, useEffect, useRef, useState } from "react";
import type { AgentMessage, Detection, Point, Result, Settings, StreamEvent, TrackEvent, VideoInfo } from "./api";
import { analyze, uploadVideo } from "./api";
import AgentTimeline from "./components/AgentTimeline";
import ResultCard from "./components/ResultCard";
import SettingsPanel, { type Options } from "./components/SettingsPanel";
import VideoStage, { type Mode, type Roi } from "./components/VideoStage";

// Recharts is heavy and only needed after a result, so keep it out of the first load.
const Charts = lazy(() => import("./components/Charts"));

const DEFAULT_SETTINGS: Settings = {
  tracker: "motion",
  hsv_lower: [0, 0, 200],
  hsv_upper: [180, 60, 255],
  fields: "none",
  roi: null,
  min_area: 10,
  max_area: null,
  max_jump: null,
  sam_box: null,
};
const DEFAULT_OPTIONS: Options = { start: 0, end: null, capture_fps: null, model: "auto", axis: "x", agent: true, height_m: null, detect_release: true };
const MOTION_WORD: Record<string, string> = { free_fall: "falling", projectile: "thrown", damped_oscillation: "swinging" };
const TRACKERS: Record<string, string> = { motion: "motion", color: "color", sam: "SAM 3.1" };
const STAGES: Record<string, string> = { loading: "Loading frames", tracking: "Tracking the ball", fitting: "Fitting models", agent: "Muse is reasoning" };

export default function App() {
  const [info, setInfo] = useState<VideoInfo | null>(null);
  const [uploading, setUploading] = useState(false);
  const [mode, setMode] = useState<Mode>("calibrate");
  const [points, setPoints] = useState<Point[]>([]);
  const [lengthM, setLengthM] = useState(1);
  const [roi, setRoi] = useState<Roi | null>(null);
  const [ballBox, setBallBox] = useState<Roi | null>(null);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [options, setOptions] = useState(DEFAULT_OPTIONS);

  const [running, setRunning] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [track, setTrack] = useState<TrackEvent | null>(null);
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [cost, setCost] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [detection, setDetection] = useState<Detection | null>(null);
  const abort = useRef<AbortController | null>(null);

  const reset = () => {
    setTrack(null);
    setMessages([]);
    setResult(null);
    setCost(null);
    setError(null);
    setWarnings([]);
    setDetection(null);
  };

  const onFile = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const uploaded = await uploadVideo(file);
      reset();
      setPoints([]);
      setRoi(null);
      setBallBox(null);
      setMode("calibrate");
      setInfo(uploaded);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const onEvent = (e: StreamEvent) => {
    switch (e.event) {
      case "status":
        setStage(e.data.stage);
        break;
      case "video":
        setWarnings(e.data.warnings);
        break;
      case "detection":
        setDetection(e.data);
        break;
      case "track":
        setTrack(e.data);
        // The auto-detect note is shown as information below, not as a warning.
        setWarnings((w) => [...w, ...(e.data.notes ?? []).filter((n) => !n.startsWith("ball found automatically"))]);
        break;
      case "agent_message":
        setMessages((m) => [...m, e.data]);
        break;
      case "result":
        setResult(e.data);
        break;
      case "cost":
        setCost(e.data.run_usd);
        break;
      case "error":
        setError(`${e.data.stage}: ${e.data.message}`);
        break;
    }
  };

  const run = async () => {
    if (!info || points.length < 2) return;
    reset();
    setRunning(true);
    setMode("view");
    abort.current = new AbortController();
    try {
      await analyze(
        { video_id: info.video_id, scale: { p1: points[0], p2: points[1], length_m: lengthM }, settings: { ...settings, roi, sam_box: sam ? ballBox : null }, ...options },
        onEvent,
        abort.current.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
    } finally {
      setRunning(false);
      setStage(null);
    }
  };

  const calibrated = points.length === 2;
  const sam = settings.tracker === "sam";
  useEffect(() => {
    if (!sam && mode === "ball") setMode("view");
  }, [sam, mode]);
  // The box belongs to the first analysed frame: a new start time makes it point at the wrong place.
  useEffect(() => setBallBox(null), [options.start]);
  const pxPerM = calibrated ? Math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]) / lengthM : null;

  return (
    <div className="mx-auto max-w-[1400px] px-5 py-4">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div aria-hidden="true" className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-cyan-400 to-violet-500 text-lg font-bold text-slate-950">ϕ</div>
          <div>
            <h1 className="text-lg font-semibold leading-tight">PhysicsLens</h1>
            <p className="text-sm text-slate-400">Film an experiment. Measure the physics. Checked by an agent, not guessed.</p>
          </div>
        </div>
        {info && (
          <label className="btn-ghost cursor-pointer focus-within:ring-2 focus-within:ring-cyan-400">
            {uploading ? "Uploading…" : "New video"}
            <input type="file" accept="video/*" className="sr-only" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
          </label>
        )}
      </header>

      <main>
      {!info ? (
        <Upload onFile={onFile} uploading={uploading} error={error} />
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px] xl:grid-cols-[minmax(0,1fr)_400px]">
          <section aria-label="Video and charts" className="space-y-4">
            <div className="panel p-4">
              <VideoStage
                info={info} mode={mode} points={points} lengthM={lengthM} roi={roi} ballBox={sam ? ballBox : null} startS={options.start}
                track={track} result={result} onPoints={setPoints} onRoi={setRoi} onBallBox={setBallBox}
              />
              <div className="mt-4 flex flex-wrap items-center gap-2">
                <button className={mode === "calibrate" ? "btn-active" : "btn-ghost"} aria-pressed={mode === "calibrate"} onClick={() => setMode("calibrate")}>
                  <span aria-hidden="true">📏</span> Scale
                </button>
                <button className={mode === "roi" ? "btn-active" : "btn-ghost"} aria-pressed={mode === "roi"} onClick={() => setMode("roi")}>
                  <span aria-hidden="true">⬚</span> Search area
                </button>
                {sam && (
                  <button className={mode === "ball" ? "btn-active" : "btn-ghost"} aria-pressed={mode === "ball"} onClick={() => setMode("ball")}>
                    <span aria-hidden="true">◎</span> Ball
                  </button>
                )}
                <button className={mode === "view" ? "btn-active" : "btn-ghost"} aria-pressed={mode === "view"} onClick={() => setMode("view")}>
                  <span aria-hidden="true">▶</span> Play
                </button>
                <div className="ml-1 flex items-center gap-2 text-sm text-slate-400">
                  reference =
                  <input className="input w-24 font-mono" aria-label="Reference length in metres" type="number" step={0.01} min={0.01} value={lengthM} onChange={(e) => setLengthM(Number(e.target.value) || 1)} />
                  m
                </div>
                <div className="w-full font-mono text-sm text-slate-400 sm:ml-auto sm:w-auto">
                  {info.width}×{info.height} · {info.fps.toFixed(1)} fps{pxPerM && ` · ${pxPerM.toFixed(1)} px/m`}
                  {roi && " · area set"}
                  {track && ` · tracked by ${TRACKERS[track.source] ?? track.source}`}
                </div>
              </div>
            </div>
            {track && result && (
              <Suspense fallback={<div className="panel h-[262px]" />}>
                <Charts track={track} result={result} />
              </Suspense>
            )}
          </section>

          <aside aria-label="Analysis controls" className="space-y-4 xl:sticky xl:top-4 xl:max-h-[calc(100vh-2rem)] xl:self-start xl:overflow-y-auto">
            <div className="panel space-y-4 p-4">
              <Checklist calibrated={calibrated} roi={!!roi} ballBox={sam ? !!ballBox : null} />
              <SettingsPanel settings={settings} options={options} onSettings={setSettings} onOptions={setOptions} />
              <button
                className="btn min-h-10 w-full bg-gradient-to-r from-cyan-400 to-violet-500 text-base font-semibold text-slate-950 hover:brightness-110"
                disabled={!calibrated || running}
                aria-busy={running}
                onClick={run}
              >
                {running ? `${stageText(stage, sam)}…` : "Analyze"}
              </button>
              <div className="sr-only" role="status">
                {running ? `${stageText(stage, sam)}…` : ""}
              </div>
              {running && (
                <button className="btn-ghost w-full" onClick={() => abort.current?.abort()}>
                  Stop
                </button>
              )}
            </div>

            {error && (
              <div role="alert" className="rise rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
                {error}
                {/* The API may have been paid for before the run failed: say so. */}
                {cost != null && cost > 0 && !result && <div className="mt-1 font-mono text-rose-300/80">API cost of this run: ${cost.toFixed(3)}</div>}
              </div>
            )}
            {detection && (
              <div className="rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-3 text-sm text-cyan-100">
                Ball found automatically: the {MOTION_WORD[detection.model] ?? detection.model} object, from {detection.t.toFixed(2)} s into the
                analysed clip, box {detection.box.map((v) => Math.round(v)).join(", ")} px. Wrong object? Use “Ball” under the video to box the right one.
              </div>
            )}
            {warnings.map((w, i) => (
              <div key={`${i}-${w}`} className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-200">
                {w}
              </div>
            ))}
            {result && <ResultCard result={result} cost={cost} />}
            {(messages.length > 0 || (running && options.agent)) && (
              <div className="panel p-4">
                <h2 className="label mb-4">Agent reasoning · live</h2>
                <AgentTimeline messages={messages} running={running && stage === "agent"} />
              </div>
            )}
          </aside>
        </div>
      )}
      </main>
      <footer className="mt-8 text-center text-sm text-slate-400">Runtime models: Meta Muse Spark and SAM 3.1 via the Meta Model API · physics fits are deterministic (SciPy)</footer>
    </div>
  );
}

function stageText(stage: string | null, sam: boolean): string {
  if (stage === "tracking" && sam) return "SAM 3.1 is tracking";
  return STAGES[stage ?? ""] ?? "Starting";
}

function Checklist({ calibrated, roi, ballBox }: { calibrated: boolean; roi: boolean; ballBox: boolean | null }) {
  const item = (done: boolean, text: string, optional = false) => (
    <li className={`flex items-center gap-2 ${done ? "text-emerald-300" : "text-slate-400"}`}>
      <span aria-hidden="true" className={`grid h-5 w-5 place-items-center rounded-full text-xs ${done ? "bg-emerald-500/20" : "bg-slate-800"}`}>{done ? "✓" : ""}</span>
      <span className="sr-only">{done ? "Done:" : "To do:"}</span>
      {text}
      {optional && <span className="text-slate-400">(optional)</span>}
    </li>
  );
  return (
    <ul className="space-y-2 text-sm">
      {item(true, "Video loaded")}
      {item(calibrated, "Scale: click both ends of a ruler")}
      {ballBox !== null && item(ballBox, "Ball box on the first frame (else found from how it moves)", true)}
      {item(roi, "Search area around the ball", true)}
    </ul>
  );
}

function Upload({ onFile, uploading, error }: { onFile: (f: File) => void; uploading: boolean; error: string | null }) {
  const [over, setOver] = useState(false);
  return (
    <label
      aria-describedby="upload-help"
      className={`panel flex min-h-[60svh] cursor-pointer flex-col items-center justify-center gap-3 border-2 border-dashed p-6 transition focus-within:ring-2 focus-within:ring-cyan-400 ${over ? "border-cyan-400 bg-cyan-400/5" : "border-slate-700"}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
      }}
    >
      <div aria-hidden="true" className="text-5xl">🎥</div>
      <div className="text-lg font-semibold">{uploading ? "Uploading…" : "Drop a video of your experiment"}</div>
      <div id="upload-help" className="max-w-md text-center text-sm text-slate-400">
        A dropped ball, a throw, or a pendulum. Keep the phone still and put a ruler or a sheet of A4 in the plane of motion.
      </div>
      {error && <div className="text-sm text-rose-300">{error}</div>}
      <input type="file" accept="video/*" className="sr-only" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
    </label>
  );
}
