import { useCallback, useEffect, useRef, useState } from "react";
import { useInView } from "../hooks/useInView";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { fitFall, fitLine } from "./fit";
import { G, gaussian, mulberry32 } from "./sim";

// World in metres, y down. Samples arrive at 120 fps and play at ¼ speed, like a phone's slow motion.
const W = 2.4;
const H = 1.2;
const FLOOR = 1.12;
const FPS = 120;
const SLOW = 0.25;
const NOISE_M = 0.003; // ≈1.5 px of tracker jitter at the default size
const HOLD_S = 1.6;
const SEED = 7;

interface Launch {
  x0: number;
  y0: number;
  vx: number;
  vy: number; // up is negative
}
const DROP: Launch = { x0: 0.55, y0: 0.08, vx: 0, vy: 0 };
const THROW: Launch = { x0: 0.25, y0: 0.95, vx: 1.15, vy: -3.1 };

interface Flight {
  launch: Launch;
  t: number[];
  x: number[];
  y: number[];
  gHistory: number[];
  done: boolean;
}

/** Positions sampled until the ball reaches the floor, with seeded tracker noise. */
function sampleFlight(launch: Launch, rand: () => number): { t: number[]; x: number[]; y: number[] } {
  const t: number[] = [];
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < 400; i++) {
    const s = i / FPS;
    const yy = launch.y0 + launch.vy * s + 0.5 * G * s * s;
    const xx = launch.x0 + launch.vx * s;
    if (yy > FLOOR || xx > W - 0.05) break;
    t.push(s);
    x.push(xx + gaussian(rand) * NOISE_M);
    y.push(yy + gaussian(rand) * NOISE_M);
  }
  return { t, x, y };
}

function css(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

/** The hero: a ball falls or is thrown, tracked dots appear, and the fitted g converges on 9.81 as points arrive. */
export default function HeroFit() {
  const reduced = useReducedMotion();
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const inView = useInView(wrap);
  const [paused, setPaused] = useState(false);
  const [readout, setReadout] = useState<{ g: number; err: number; n: number } | null>(null);
  const [history, setHistory] = useState<number[]>([]);
  const [announce, setAnnounce] = useState("");
  const [aim, setAim] = useState({ angle: 60, speed: 3.4 }); // keyboard throw: degrees above horizontal, m/s
  const rand = useRef(mulberry32(SEED));
  const plan = useRef<{ full: ReturnType<typeof sampleFlight>; flight: Flight; clock: number; hold: number; next: "drop" | "throw" } | null>(null);
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const speak = useRef(false);

  const start = useCallback(
    (launch: Launch, next: "drop" | "throw", byUser = false) => {
      const full = sampleFlight(launch, rand.current);
      const flight: Flight = { launch, t: [], x: [], y: [], gHistory: [], done: false };
      speak.current = byUser; // the self-running loop never talks; a launch the user asked for does
      if (reduced) {
        // Less motion: the finished fit at once, no animation.
        flight.t = full.t;
        flight.x = full.x;
        flight.y = full.y;
        flight.done = true;
        for (let n = 3; n <= full.t.length; n++) {
          const f = fitFall(full.t.slice(0, n), full.y.slice(0, n));
          if (f) flight.gHistory.push(f.g);
        }
      }
      plan.current = { full, flight, clock: 0, hold: 0, next };
    },
    [reduced],
  );

  useEffect(() => start(DROP, "throw"), [start]);

  // Drawing and the clock share one loop; it stops when the hero is off screen, paused, or motion is reduced.
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const colors = { data: css("--data", "#22d3ee"), model: css("--model", "#c084fc"), grid: css("--grid-major", "rgba(148,163,184,.16)"), ink: css("--ink-3", "#94a3b8") };
    let frame = 0;
    let last = performance.now();
    let lastN = -1;

    const draw = () => {
      const p = plan.current;
      const ctx = el.getContext("2d");
      if (!p || !ctx) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (el.width !== Math.round(w * dpr) || el.height !== Math.round(h * dpr)) {
        el.width = Math.round(w * dpr);
        el.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const k = Math.min(w / W, h / H);
      const ox = (w - W * k) / 2;
      const oy = (h - H * k) / 2;
      const X = (m: number) => ox + m * k;
      const Y = (m: number) => oy + m * k;

      // Grid every 10 cm, floor line, axis labels.
      ctx.strokeStyle = colors.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let gx = 0; gx <= W + 1e-9; gx += 0.1) {
        ctx.moveTo(X(gx), Y(0));
        ctx.lineTo(X(gx), Y(H));
      }
      for (let gy = 0; gy <= H + 1e-9; gy += 0.1) {
        ctx.moveTo(X(0), Y(gy));
        ctx.lineTo(X(W), Y(gy));
      }
      ctx.stroke();
      ctx.fillStyle = colors.ink;
      ctx.font = "11px 'JetBrains Mono', ui-monospace, monospace";
      ctx.fillText("y (m) ↓", X(0.02), Y(0.06));
      ctx.fillText("x (m) →", X(W - 0.32), Y(H - 0.03));

      const f = p.flight;
      const fit = f.t.length >= 3 ? fitFall(f.t, f.y) : null;
      const line = f.t.length >= 2 ? fitLine(f.t, f.x) : null;
      ctx.fillStyle = colors.data;
      f.t.forEach((_, i) => {
        ctx.beginPath();
        ctx.arc(X(f.x[i]), Y(f.y[i]), 2.6, 0, Math.PI * 2);
        ctx.fill();
      });
      if (fit && line) {
        // The model over the whole sampled window: it bends as more points arrive.
        ctx.strokeStyle = colors.model;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        const tEnd = f.t[f.t.length - 1];
        for (let i = 0; i <= 60; i++) {
          const s = (tEnd * i) / 60;
          const px = X(line.intercept + line.slope * s);
          const py = Y(fit.y0 + fit.v0 * s + 0.5 * fit.g * s * s);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
      const n = f.t.length;
      if (n > 0 && !f.done) {
        // The tracker's reticle on the newest sample.
        ctx.strokeStyle = colors.data;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(X(f.x[n - 1]), Y(f.y[n - 1]), 9, 0, Math.PI * 2);
        ctx.stroke();
      }
      const d = drag.current;
      if (d) {
        // Slingshot: pull back, the dashed arrow is the launch velocity.
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = colors.ink;
        ctx.beginPath();
        ctx.moveTo(d.x, d.y);
        ctx.lineTo(2 * d.x - d.px, 2 * d.y - d.py);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (n !== lastN) {
        lastN = n;
        setReadout(fit ? { g: fit.g, err: fit.gErr, n } : null);
        setHistory([...f.gHistory]);
      }
      return { X, Y, k, ox, oy };
    };

    const tick = (now: number) => {
      frame = requestAnimationFrame(tick);
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const p = plan.current;
      if (p && !p.flight.done) {
        p.clock += dt * SLOW;
        const f = p.flight;
        while (f.t.length < p.full.t.length && p.full.t[f.t.length] <= p.clock) {
          const i = f.t.length;
          f.t.push(p.full.t[i]);
          f.x.push(p.full.x[i]);
          f.y.push(p.full.y[i]);
          const fit = f.t.length >= 3 ? fitFall(f.t, f.y) : null;
          if (fit) f.gHistory.push(fit.g);
        }
        if (f.t.length >= p.full.t.length) {
          f.done = true;
          const fit = fitFall(f.t, f.y);
          if (fit && speak.current) {
            const off = (100 * (fit.g - G)) / G;
            setAnnounce(`Fit from ${f.t.length} points: g = ${fit.g.toFixed(2)} ± ${fit.gErr.toFixed(2)} metres per second squared, ${Math.abs(off).toFixed(1)}% ${off < 0 ? "below" : "above"} 9.81.`);
          }
        }
      } else if (p && p.flight.done) {
        p.hold += dt;
        if (p.hold > HOLD_S) start(p.next === "throw" ? THROW : DROP, p.next === "throw" ? "drop" : "throw");
      }
      draw();
    };

    if (reduced || paused || !inView) {
      draw();
      // Redraw on resize even when still.
      const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => draw()) : null;
      ro?.observe(el);
      return () => ro?.disconnect();
    }
    frame = requestAnimationFrame((now) => {
      last = now;
      tick(now);
    });
    return () => cancelAnimationFrame(frame);
  }, [reduced, paused, inView, start]);

  const launchAim = () => {
    const a = (aim.angle * Math.PI) / 180;
    start({ x0: 0.25, y0: 0.95, vx: aim.speed * Math.cos(a), vy: -aim.speed * Math.sin(a) }, "drop", true);
  };

  // Pointer: drag back and release to throw from the launch point; a tap throws towards the tap.
  const toWorld = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const k = Math.min(r.width / W, r.height / H);
    return { mx: (e.clientX - r.left - (r.width - W * k) / 2) / k, my: (e.clientY - r.top - (r.height - H * k) / 2) / k, px: e.clientX - r.left, py: e.clientY - r.top };
  };
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const { px, py } = toWorld(e);
    drag.current = { x: px, y: py, px, py };
    e.currentTarget.style.touchAction = "none"; // a pull-back must not scroll the page
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drag.current) return;
    const { px, py } = toWorld(e);
    drag.current = { ...drag.current, px, py };
  };
  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    drag.current = null;
    e.currentTarget.style.touchAction = "pan-y";
    if (!d) return;
    const r = e.currentTarget.getBoundingClientRect();
    const k = Math.min(r.width / W, r.height / H);
    const pull = Math.hypot(d.px - d.x, d.py - d.y);
    let vx: number;
    let vy: number;
    if (pull > 12) {
      vx = ((d.x - d.px) / k) * 3;
      vy = ((d.y - d.py) / k) * 3;
    } else {
      const { mx, my } = toWorld(e); // a tap: throw towards it
      const dx = Math.max(0.1, mx - 0.25);
      const dy = my - 0.95;
      const len = Math.hypot(dx, dy) || 1;
      vx = (3.6 * dx) / len;
      vy = (3.6 * dy) / len - 1.2;
    }
    const clamp = (v: number, m: number) => Math.max(-m, Math.min(m, v));
    start({ x0: 0.25, y0: 0.95, vx: clamp(Math.max(0.2, vx), 3), vy: clamp(vy, 4.5) }, "drop", true);
    setPaused(false);
  };
  const onPointerCancel = () => {
    drag.current = null;
  };

  const off = readout ? (100 * (readout.g - G)) / G : null;
  return (
    <div ref={wrap}>
      <div className="instrument p-3">
        <canvas
          ref={canvas}
          tabIndex={0}
          role="group"
          aria-roledescription="throw simulator"
          aria-label="Ball throw simulator. Cyan dots are the tracked positions, the violet curve is the fitted fall. Arrow keys aim, Enter throws, Space pauses."
          aria-describedby="hero-keys"
          className="block aspect-[4/3] w-full cursor-crosshair rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 sm:aspect-[2/1]"
          style={{ touchAction: "pan-y" }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerCancel}
          onLostPointerCapture={onPointerCancel}
          onKeyDown={(e) => {
            if (e.key === "ArrowUp") setAim((a) => ({ ...a, angle: Math.min(85, a.angle + 5) }));
            else if (e.key === "ArrowDown") setAim((a) => ({ ...a, angle: Math.max(5, a.angle - 5) }));
            else if (e.key === "ArrowRight") setAim((a) => ({ ...a, speed: Math.min(4.5, +(a.speed + 0.2).toFixed(1)) }));
            else if (e.key === "ArrowLeft") setAim((a) => ({ ...a, speed: Math.max(1, +(a.speed - 0.2).toFixed(1)) }));
            else if (e.key === "Enter") launchAim();
            else if (e.key === " " && !reduced) setPaused((p) => !p);
            else return;
            e.preventDefault();
          }}
        />
        <div className="readout mt-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs text-slate-400">
          <span>{FPS} fps · shown at ¼ speed · n = {readout?.n ?? 0} · seed {SEED}</span>
          <span id="hero-keys">drag back to throw · keys: ←→ speed {aim.speed} m/s · ↑↓ angle {aim.angle}° · Enter throws · Space pauses</span>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-3">
        <div className="readout text-2xl text-white sm:text-3xl" aria-hidden="true">
          g = <span className={off != null && Math.abs(off) < 1 ? "text-emerald-300" : "text-cyan-300"}>{readout ? readout.g.toFixed(2) : "—"}</span>
          <span className="text-lg text-slate-400"> ± {readout && Number.isFinite(readout.err) ? readout.err.toFixed(2) : "—"}</span>
          <span className="ml-1 text-sm text-slate-400">m/s²</span>
          {off != null && <span className="ml-2 text-sm text-slate-400">({off > 0 ? "+" : "−"}{Math.abs(off).toFixed(1)}%)</span>}
        </div>
        <Sparkline values={history} />
        <div className="flex gap-2">
          <button className="btn-ghost" onClick={() => start(DROP, "throw", true)}>Drop</button>
          <button className="btn-ghost" onClick={launchAim}>Throw</button>
          {!reduced && (
            <button className="btn-ghost" aria-pressed={paused} onClick={() => setPaused((p) => !p)}>
              {paused ? "Play" : "Pause"}
            </button>
          )}
        </div>
      </div>
      <p className="sr-only" role="status">{announce || (readout ? `Last fit: g = ${readout.g.toFixed(2)} metres per second squared from ${readout.n} points.` : "")}</p>
    </div>
  );
}

/** g against the number of points, converging on the dashed 9.81 line. */
function Sparkline({ values }: { values: number[] }) {
  const w = 132;
  const h = 34;
  const lo = 6;
  const hi = 14;
  const y = (v: number) => h - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * h;
  const pts = values.map((v, i) => `${values.length < 2 ? 0 : (i / (values.length - 1)) * w},${y(v).toFixed(1)}`).join(" ");
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true" className="mr-8 overflow-visible">
      <line x1={0} x2={w} y1={y(G)} y2={y(G)} stroke="var(--ink-3)" strokeDasharray="3 3" strokeWidth={1} />
      <text x={w + 4} y={y(G) + 3} fontSize="10" fill="var(--ink-3)" fontFamily="JetBrains Mono, monospace">9.81</text>
      {values.length > 1 && <polyline points={pts} fill="none" stroke="var(--data)" strokeWidth={1.5} />}
    </svg>
  );
}
