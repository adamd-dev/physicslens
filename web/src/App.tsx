import { useRef, useState } from "react";
import type { AgentMessage, Point, Result, Settings, StreamEvent, TrackEvent, VideoInfo } from "./api";
import { analyze, uploadVideo } from "./api";
import AgentTimeline from "./components/AgentTimeline";
import Charts from "./components/Charts";
import ResultCard from "./components/ResultCard";
import SettingsPanel, { type Options } from "./components/SettingsPanel";
import VideoStage, { type Mode, type Roi } from "./components/VideoStage";

const DEFAULT_SETTINGS: Settings = {
  tracker: "motion",
  hsv_lower: [0, 0, 200],
  hsv_upper: [180, 60, 255],
  fields: "none",
  roi: null,
  min_area: 10,
  max_area: null,
  max_jump: null,
};
const DEFAULT_OPTIONS: Options = { start: 0, end: null, capture_fps: null, model: "auto", axis: "x", agent: true, height_m: null };
const STAGES: Record<string, string> = { loading: "Loading frames", tracking: "Tracking the ball", fitting: "Fitting models", agent: "Muse is reasoning" };

export default function App() {
  const [info, setInfo] = useState<VideoInfo | null>(null);
  const [uploading, setUploading] = useState(false);
  const [mode, setMode] = useState<Mode>("calibrate");
  const [points, setPoints] = useState<Point[]>([]);
  const [lengthM, setLengthM] = useState(1);
  const [roi, setRoi] = useState<Roi | null>(null);
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
  const abort = useRef<AbortController | null>(null);

  const reset = () => {
    setTrack(null);
    setMessages([]);
    setResult(null);
    setCost(null);
    setError(null);
    setWarnings([]);
  };

  const onFile = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const uploaded = await uploadVideo(file);
      reset();
      setPoints([]);
      setRoi(null);
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
      case "track":
        setTrack(e.data);
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
        { video_id: info.video_id, scale: { p1: points[0], p2: points[1], length_m: lengthM }, settings: { ...settings, roi }, ...options },
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
  const pxPerM = calibrated ? Math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]) / lengthM : null;

  return (
    <div className="mx-auto max-w-[1400px] px-5 py-5">
      <header className="mb-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-cyan-400 to-violet-500 text-lg font-bold text-slate-950">ϕ</div>
          <div>
            <h1 className="text-lg font-semibold leading-tight">PhysicsLens</h1>
            <p className="text-xs text-slate-400">Film an experiment. Measure the physics. Checked by an agent, not guessed.</p>
          </div>
        </div>
        {info && (
          <label className="btn-ghost cursor-pointer">
            {uploading ? "Uploading…" : "New video"}
            <input type="file" accept="video/*" className="hidden" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
          </label>
        )}
      </header>

      {!info ? (
        <Upload onFile={onFile} uploading={uploading} error={error} />
      ) : (
        <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1fr)_400px]">
          <main className="space-y-4">
            <div className="panel p-3">
              <VideoStage info={info} mode={mode} points={points} lengthM={lengthM} roi={roi} track={track} result={result} onPoints={setPoints} onRoi={setRoi} />
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button className={mode === "calibrate" ? "btn-active" : "btn-ghost"} onClick={() => setMode("calibrate")}>
                  📏 Scale
                </button>
                <button className={mode === "roi" ? "btn-active" : "btn-ghost"} onClick={() => setMode("roi")}>
                  ⬚ Search area
                </button>
                <button className={mode === "view" ? "btn-active" : "btn-ghost"} onClick={() => setMode("view")}>
                  ▶ Play
                </button>
                <div className="ml-2 flex items-center gap-2 text-sm text-slate-400">
                  reference =
                  <input className="input w-20 font-mono" type="number" step={0.01} min={0.01} value={lengthM} onChange={(e) => setLengthM(Number(e.target.value) || 1)} />
                  m
                </div>
                <div className="ml-auto font-mono text-xs text-slate-500">
                  {info.width}×{info.height} · {info.fps.toFixed(1)} fps{pxPerM && ` · ${pxPerM.toFixed(1)} px/m`}
                  {roi && " · area set"}
                </div>
              </div>
            </div>
            {track && result && <Charts track={track} result={result} />}
          </main>

          <aside className="space-y-4">
            <div className="panel space-y-4 p-4">
              <Checklist calibrated={calibrated} roi={!!roi} />
              <SettingsPanel settings={settings} options={options} onSettings={setSettings} onOptions={setOptions} />
              <button
                className="btn w-full bg-gradient-to-r from-cyan-400 to-violet-500 py-2.5 text-base font-semibold text-slate-950 hover:brightness-110"
                disabled={!calibrated || running}
                onClick={run}
              >
                {running ? `${STAGES[stage ?? ""] ?? "Starting"}…` : "Analyze"}
              </button>
              {running && (
                <button className="btn-ghost w-full" onClick={() => abort.current?.abort()}>
                  Stop
                </button>
              )}
            </div>

            {error && <div className="rise rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-200">{error}</div>}
            {warnings.map((w) => (
              <div key={w} className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-2.5 text-xs text-amber-200">
                {w}
              </div>
            ))}
            {result && <ResultCard result={result} cost={cost} />}
            {(messages.length > 0 || (running && options.agent)) && (
              <div className="panel p-4">
                <div className="label mb-3">Agent reasoning · live</div>
                <AgentTimeline messages={messages} running={running && stage === "agent"} />
              </div>
            )}
          </aside>
        </div>
      )}
      <footer className="mt-8 text-center text-xs text-slate-600">Runtime model: Meta Muse Spark via the Meta Model API · physics fits are deterministic (SciPy)</footer>
    </div>
  );
}

function Checklist({ calibrated, roi }: { calibrated: boolean; roi: boolean }) {
  const item = (done: boolean, text: string, optional = false) => (
    <li className={`flex items-center gap-2 ${done ? "text-emerald-300" : "text-slate-400"}`}>
      <span className={`grid h-4 w-4 place-items-center rounded-full text-[10px] ${done ? "bg-emerald-500/20" : "bg-slate-800"}`}>{done ? "✓" : ""}</span>
      {text}
      {optional && <span className="text-slate-600">(optional)</span>}
    </li>
  );
  return (
    <ul className="space-y-1.5 text-sm">
      {item(true, "Video loaded")}
      {item(calibrated, "Scale: click both ends of a ruler")}
      {item(roi, "Search area around the ball", true)}
    </ul>
  );
}

function Upload({ onFile, uploading, error }: { onFile: (f: File) => void; uploading: boolean; error: string | null }) {
  const [over, setOver] = useState(false);
  return (
    <label
      className={`panel flex h-[60vh] cursor-pointer flex-col items-center justify-center gap-3 border-2 border-dashed transition ${over ? "border-cyan-400 bg-cyan-400/5" : "border-slate-700"}`}
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
      <div className="text-5xl">🎥</div>
      <div className="text-lg font-semibold">{uploading ? "Uploading…" : "Drop a video of your experiment"}</div>
      <div className="max-w-md text-center text-sm text-slate-400">
        A dropped ball, a throw, or a pendulum. Keep the phone still and put a ruler or a sheet of A4 in the plane of motion.
      </div>
      {error && <div className="text-sm text-rose-300">{error}</div>}
      <input type="file" accept="video/*" className="hidden" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
    </label>
  );
}
