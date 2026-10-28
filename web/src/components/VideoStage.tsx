import { useEffect, useRef } from "react";
import type { Point, Result, TrackEvent, VideoInfo } from "../api";
import { videoUrl } from "../api";

export type Mode = "view" | "calibrate" | "roi" | "ball";
export type Roi = [number, number, number, number];

interface Props {
  info: VideoInfo;
  mode: Mode;
  points: Point[]; // calibration clicks, source pixels
  lengthM: number;
  roi: Roi | null;
  ballBox: Roi | null; // SAM's object prompt, drawn on the first frame
  startS: number;
  track: TrackEvent | null;
  result: Result | null;
  onPoints: (p: Point[]) => void;
  onRoi: (r: Roi | null) => void;
  onBallBox: (r: Roi | null) => void;
}

const CYAN = "#22d3ee";
const AMBER = "#fbbf24";
const VIOLET = "#c084fc";
const HINT: Record<Mode, string> = {
  view: "Video with the tracked points and the fitted curve drawn on top",
  calibrate: "Click both ends of the reference object",
  roi: "Drag a box around the ball's path",
  ball: "Drag a tight box around the ball on this first frame",
};

export default function VideoStage({ info, mode, points, lengthM, roi, ballBox, startS, track, result, onPoints, onRoi, onBallBox }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ start: Point; now: Point } | null>(null);
  // The draw loop reads the latest props through this ref instead of restarting on every change.
  const state = useRef({ mode, points, lengthM, roi, ballBox, track, result });
  state.current = { mode, points, lengthM, roi, ballBox, track, result };

  // SAM follows the object from the first analysed frame, so the box is drawn on that frame.
  useEffect(() => {
    const video = videoRef.current;
    if (mode === "ball" && video) {
      video.pause();
      video.currentTime = startS;
    }
  }, [mode, startS]);

  useEffect(() => {
    let frame = 0;
    const draw = () => {
      frame = requestAnimationFrame(draw);
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      const ctx = canvas.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const k = w / info.width; // source px -> CSS px
      const s = state.current;
      const now = video.currentTime;

      const dragging = drag.current ? toRoi(drag.current.start, drag.current.now) : null;
      const box = s.mode === "roi" && dragging ? dragging : s.roi;
      if (box) {
        const [x0, y0, x1, y1] = box.map((v) => v * k);
        ctx.fillStyle = "rgba(2,6,23,0.55)";
        ctx.fillRect(0, 0, w, y0);
        ctx.fillRect(0, y1, w, h - y1);
        ctx.fillRect(0, y0, x0, y1 - y0);
        ctx.fillRect(x1, y0, w - x1, y1 - y0);
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = CYAN;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.setLineDash([]);
      }

      const ball = s.mode === "ball" && dragging ? dragging : s.ballBox;
      if (ball) {
        const [x0, y0, x1, y1] = ball.map((v) => v * k);
        ctx.strokeStyle = VIOLET;
        ctx.lineWidth = 2;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        label(ctx, "SAM", (x0 + x1) / 2, y0 - 14, VIOLET);
      }

      if (s.points.length) {
        const [a, b] = s.points.map(([x, y]) => [x * k, y * k] as Point);
        ctx.strokeStyle = AMBER;
        ctx.fillStyle = AMBER;
        ctx.lineWidth = 2;
        if (b) {
          ctx.beginPath();
          ctx.moveTo(...a);
          ctx.lineTo(...b);
          ctx.stroke();
          label(ctx, `${s.lengthM} m`, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 10, AMBER);
        }
        for (const p of [a, b].filter(Boolean) as Point[]) {
          ctx.beginPath();
          ctx.arc(p[0], p[1], 4, 0, 2 * Math.PI);
          ctx.fill();
        }
      }

      if (s.result) {
        const c = s.result.curve;
        ctx.strokeStyle = VIOLET;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = VIOLET;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        c.x_px.forEach((x, i) => (i ? ctx.lineTo(x * k, c.y_px[i] * k) : ctx.moveTo(x * k, c.y_px[i] * k)));
        ctx.stroke();
        ctx.shadowBlur = 0;
        const land = landingPx(s.result, s.track?.px_per_m);
        if (land) {
          const [lx, ly] = [land[0] * k, land[1] * k];
          ctx.strokeStyle = AMBER;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(lx - 9, ly - 9);
          ctx.lineTo(lx + 9, ly + 9);
          ctx.moveTo(lx + 9, ly - 9);
          ctx.lineTo(lx - 9, ly + 9);
          ctx.stroke();
          label(ctx, "predicted landing", lx, ly - 16, AMBER);
        }
      }

      if (s.track) {
        // Still samples dropped before fitting (ball in the hand, on the floor): shown, not used.
        const r = s.track.rest ?? { t_play: [], x_px: [], y_px: [] };
        ctx.fillStyle = "rgba(148,163,184,0.7)";
        for (let i = 0; i < r.t_play.length; i++) {
          ctx.beginPath();
          ctx.arc(r.x_px[i] * k, r.y_px[i] * k, 2.6, 0, 2 * Math.PI);
          ctx.fill();
        }
        const t = s.track;
        let current = -1;
        for (let i = 0; i < t.t_play.length; i++) {
          const past = t.t_play[i] <= now + 1e-3;
          if (past) current = i;
          ctx.fillStyle = past ? CYAN : "rgba(34,211,238,0.25)";
          ctx.beginPath();
          ctx.arc(t.x_px[i] * k, t.y_px[i] * k, past ? 3.2 : 2.2, 0, 2 * Math.PI);
          ctx.fill();
        }
        if (current >= 0) {
          ctx.strokeStyle = "white";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(t.x_px[current] * k, t.y_px[current] * k, 9, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [info.width]);

  const toSource = (e: React.PointerEvent<HTMLCanvasElement>): Point => {
    const rect = e.currentTarget.getBoundingClientRect();
    const k = info.width / rect.width;
    return [Math.round((e.clientX - rect.left) * k), Math.round((e.clientY - rect.top) * k)];
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const p = toSource(e);
    if (mode === "calibrate") onPoints(points.length === 1 ? [points[0], p] : [p]);
    if (mode === "roi" || mode === "ball") {
      e.currentTarget.setPointerCapture(e.pointerId);
      drag.current = { start: p, now: p };
    }
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (drag.current) drag.current.now = toSource(e);
  };
  const onPointerUp = () => {
    if (!drag.current) return;
    const box = toRoi(drag.current.start, drag.current.now);
    drag.current = null;
    // A ball can be a few pixels wide: the SAM box only needs to be a real drag, not a big one.
    if (mode === "ball") onBallBox(box[2] - box[0] > 2 && box[3] - box[1] > 2 ? box : null);
    else onRoi(box[2] - box[0] > 8 && box[3] - box[1] > 8 ? box : null);
  };

  return (
    <div
      className="relative mx-auto overflow-hidden rounded-xl bg-black"
      // Keep the toolbar on screen: cap the height and let the width follow the aspect ratio.
      style={{ aspectRatio: `${info.width} / ${info.height}`, maxWidth: `calc(60svh * ${info.width / info.height})` }}
    >
      <video ref={videoRef} src={videoUrl(info.video_id)} className="absolute inset-0 h-full w-full" controls={mode === "view"} preload="metadata" playsInline muted />
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={HINT[mode]}
        className={`absolute inset-0 h-full w-full ${mode === "view" ? "pointer-events-none" : "cursor-crosshair"}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />
      {mode !== "view" && (
        <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-slate-950/80 px-3 py-1.5 text-sm text-slate-200">
          {HINT[mode]}
        </div>
      )}
    </div>
  );
}

function toRoi(a: Point, b: Point): Roi {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
}

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  ctx.font = "600 14px 'Inter Variable', Inter, sans-serif";
  const w = ctx.measureText(text).width + 10;
  ctx.fillStyle = "rgba(2,6,23,0.85)";
  ctx.fillRect(x - w / 2, y - 12, w, 22);
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.fillText(text, x, y + 4);
  ctx.textAlign = "start";
}

/** Where the fitted projectile meets the landing surface, in source pixels. */
function landingPx(result: Result, pxPerM?: number): Point | null {
  const p = result.params;
  if (!result.landing || !pxPerM || p.x0 == null || p.vx == null || p.y0 == null || p.vy == null || p.g == null) return null;
  const t = result.landing.flight_time_s;
  return [(p.x0 + p.vx * t) * pxPerM, (p.y0 + p.vy * t + 0.5 * p.g * t * t) * pxPerM];
}
