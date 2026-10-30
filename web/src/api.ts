// Client for the local PhysicsLens API (physicslens/server.py).

export interface VideoInfo {
  video_id: string;
  name: string;
  fps: number;
  frames: number;
  width: number;
  height: number;
  duration_s: number | null;
}

export type Point = [number, number];

export interface Scale {
  p1: Point;
  p2: Point;
  length_m: number;
}

export interface Settings {
  tracker: "motion" | "color" | "sam";
  hsv_lower: [number, number, number];
  hsv_upper: [number, number, number];
  fields: "none" | "auto" | "top" | "bottom";
  roi: [number, number, number, number] | null;
  min_area: number;
  max_area: number | null;
  max_jump: number | null;
  sam_box: [number, number, number, number] | null; // around the ball on the first frame, source pixels
}

export interface AnalyzeRequest {
  video_id: string;
  scale: Scale;
  settings: Settings;
  start: number;
  end: number | null;
  capture_fps: number | null;
  model: "auto" | "free_fall" | "projectile" | "pendulum";
  axis: "x" | "y";
  agent: boolean;
  height_m: number | null;
  detect_release: boolean;
}

export interface TrackEvent {
  fields: string;
  source: "motion" | "color" | "sam"; // the tracker that produced the points (SAM falls back to motion)
  px_per_m: number;
  t: number[];
  t_play: number[];
  x_px: number[];
  y_px: number[];
  x_m: number[];
  y_m: number[];
  notes: string[];
  rest: { t_play: number[]; x_px: number[]; y_px: number[] }; // still samples dropped before fitting
}

export interface Check {
  fit_id: string;
  ok: boolean;
  problems: string[];
  runs_z: number | null;
}

export interface Landing {
  landing_distance_m: number;
  uncertainty_m: number | null; // fit plus scale reading
  fit_uncertainty_m: number | null;
  flight_time_s: number;
  g_used: number;
}

export interface Result {
  fit_id: string;
  model: "free_fall" | "projectile" | "pendulum";
  axis: "x" | "y";
  params: Record<string, number | null>;
  stderr: Record<string, number | null>;
  stderr_total: Record<string, number | null>; // fit noise plus the scale reading
  scale_rel_err: number;
  deviation_pct: { g?: number };
  r2: number | null;
  rmse_m: number | null; // null when not finite
  check: Check;
  curve: { t_play: number[]; x_px: number[]; y_px: number[] };
  length_m?: number;
  period_s?: number;
  landing?: Landing;
  landing_error?: string; // why no landing point could be predicted
  source: "agent" | "deterministic";
  summary?: string;
  steps?: number;
}

export interface AgentMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls: { id: string; name: string; arguments: string | Record<string, unknown> }[];
  tool_call_id: string | null;
}

export type StreamEvent =
  | { event: "run"; data: { id: string } }
  | { event: "status"; data: { stage: string } }
  | { event: "video"; data: { frames: number; fps: number; scale: number; warnings: string[] } }
  | { event: "track"; data: TrackEvent }
  | { event: "detection"; data: Detection }
  | { event: "fits"; data: { fits: unknown[] } }
  | { event: "agent_message"; data: AgentMessage }
  | { event: "result"; data: Result }
  | { event: "cost"; data: { run_usd: number; total_usd: number } }
  | { event: "error"; data: { stage: string; message: string; steps?: number } }
  | { event: "done"; data: Record<string, never> };

async function failure(response: Response): Promise<Error> {
  try {
    const body = await response.json();
    return new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
  } catch {
    return new Error(`${response.status} ${response.statusText}`);
  }
}

export async function uploadVideo(file: File): Promise<VideoInfo> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/videos", { method: "POST", body: form });
  if (!response.ok) throw await failure(response);
  return response.json();
}

export const videoUrl = (id: string) => `/api/videos/${id}/file`;

/** The ball found from how it moves, when no box was drawn; box in source pixels on window frame `frame`. */
export type Detection = { box: [number, number, number, number]; frame: number; t: number; model: string; r2: number | null };

/** POST /api/analyze and hand each server-sent event to `onEvent` as it arrives. */
export async function analyze(request: AnalyzeRequest, onEvent: (e: StreamEvent) => void, signal?: AbortSignal) {
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok || !response.body) throw await failure(response);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data && event === "message") continue; // a keep-alive comment, not an event
      onEvent({ event, data: JSON.parse(data || "{}") } as StreamEvent);
    }
  }
}

export type RunStatus = "ok" | "flagged" | "none";

/** One History row, computed by the server from the run's events. */
export interface RunSummary {
  id: string;
  created_at: string;
  video_id: string;
  video_name: string | null;
  status: RunStatus;
  model: Result["model"] | null;
  key: { name: string; value: number | null; err: number | null; unit: string } | null;
  dev_pct: number | null;
  tracker: string;
  agent: boolean;
  steps: number | null;
  cost_usd: number | null;
  error: { stage: string; message: string } | null;
  thumb_t: number;
}

/** A saved run: the request and the events exactly as they were streamed. */
export interface RunRecord {
  id: string;
  created_at: string;
  video_id: string;
  video_name: string | null;
  request: AnalyzeRequest;
  events: StreamEvent[];
  summary: RunSummary;
}

export async function listRuns(signal?: AbortSignal): Promise<RunSummary[]> {
  const response = await fetch("/api/runs", { signal });
  if (!response.ok) throw await failure(response);
  return response.json();
}

export async function getRun(id: string, signal?: AbortSignal): Promise<RunRecord> {
  const response = await fetch(`/api/runs/${encodeURIComponent(id)}`, { signal });
  if (!response.ok) throw await failure(response);
  return response.json();
}

export async function deleteRun(id: string): Promise<void> {
  const response = await fetch(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) throw await failure(response);
}

/** The video as it was analysed: the info VideoStage needs, rebuilt from a saved run. */
export async function videoInfo(videoId: string, signal?: AbortSignal): Promise<VideoInfo | null> {
  const response = await fetch(`/api/videos/${encodeURIComponent(videoId)}/info`, { signal });
  return response.ok ? response.json() : null;
}

export interface SiteConfig {
  public: boolean; // a shared public site: uploads and runs are visible to everyone
  max_upload_mb: number;
}

export async function getConfig(): Promise<SiteConfig> {
  const response = await fetch("/api/config");
  if (!response.ok) throw await failure(response);
  return response.json();
}
