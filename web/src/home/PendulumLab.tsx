import { useEffect, useId, useRef, useState } from "react";
import { useInView } from "../hooks/useInView";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { G, pendulumPeriod, stepPendulum } from "./sim";

const PIVOT = { x: 180, y: 30 };
const PX_PER_M = 300;
const DT = 1 / 240;
const TRAIL = 40;

/** A pendulum to play with: set its length and amplitude, and compare the period with the small-angle formula. */
export default function PendulumLab() {
  const reduced = useReducedMotion();
  const box = useRef<HTMLDivElement>(null);
  const inView = useInView(box);
  const [length, setLength] = useState(0.6);
  const [amplitudeDeg, setAmplitudeDeg] = useState(20);
  const [swinging, setSwinging] = useState(false);
  const [measured, setMeasured] = useState<number | null>(null);
  const [announce, setAnnounce] = useState("");
  const bob = useRef<SVGGElement>(null);
  const trail = useRef<(SVGCircleElement | null)[]>([]);
  const lengthId = useId();
  const ampId = useId();

  const amplitude = (amplitudeDeg * Math.PI) / 180;
  const small = 2 * Math.PI * Math.sqrt(length / G);
  const exact = pendulumPeriod(length, amplitude);
  const period = measured ?? exact;
  const lengthFromPeriod = (G * period * period) / (4 * Math.PI * Math.PI);
  const gap = 100 * (exact / small - 1);

  useEffect(() => {
    setMeasured(null);
    setAnnounce("");
    const g = bob.current;
    if (g) g.setAttribute("transform", `rotate(${amplitudeDeg} ${PIVOT.x} ${PIVOT.y})`);
  }, [length, amplitudeDeg]);

  useEffect(() => {
    if (!swinging || reduced || !inView) return;
    let state = { theta: amplitude, omega: 0 };
    let acc = 0;
    let t = 0;
    let last = performance.now();
    let frame = 0;
    const crossings: number[] = [];
    const points: { x: number; y: number }[] = [];
    const tick = (now: number) => {
      frame = requestAnimationFrame(tick);
      acc += Math.min(0.05, (now - last) / 1000);
      last = now;
      while (acc >= DT) {
        const next = stepPendulum(state, DT, length, 0);
        if (state.theta < 0 && next.theta >= 0) {
          crossings.push(t + DT * (state.theta / (state.theta - next.theta)));
          if (crossings.length >= 4) {
            const recent = crossings.slice(-4);
            const T = (recent[3] - recent[0]) / 3;
            setMeasured(T);
            if (crossings.length === 4) setAnnounce(`Period ${T.toFixed(2)} seconds, length from the period ${((G * T * T) / (4 * Math.PI * Math.PI)).toFixed(3)} metres.`);
          }
        }
        state = next;
        t += DT;
        acc -= DT;
      }
      const deg = (state.theta * 180) / Math.PI;
      bob.current?.setAttribute("transform", `rotate(${deg} ${PIVOT.x} ${PIVOT.y})`);
      const r = length * PX_PER_M;
      // SVG rotate() turns clockwise with y down: a positive angle puts the bob left of the pivot.
      points.push({ x: PIVOT.x - r * Math.sin(state.theta), y: PIVOT.y + r * Math.cos(state.theta) });
      if (points.length > TRAIL) points.shift();
      // A strobe trail: a fixed pool of circles, no re-render per frame.
      trail.current.forEach((c, i) => {
        const p = points[points.length - 1 - i * 2];
        if (!c) return;
        if (p) {
          c.setAttribute("cx", String(p.x));
          c.setAttribute("cy", String(p.y));
          c.setAttribute("opacity", String(0.6 * (1 - i / (TRAIL / 2))));
        } else c.setAttribute("opacity", "0");
      });
    };
    frame = requestAnimationFrame((now) => {
      last = now;
      tick(now);
    });
    return () => cancelAnimationFrame(frame);
  }, [swinging, reduced, inView, length, amplitude]);

  const r = length * PX_PER_M;
  const fan = reduced ? Array.from({ length: 9 }, (_, i) => -amplitude + (2 * amplitude * i) / 8) : [];
  return (
    <div ref={box} className="instrument grid gap-6 p-5 md:grid-cols-[minmax(0,360px)_1fr]">
      <svg viewBox="0 0 360 360" className="w-full" role="img" aria-label={`A ${length.toFixed(2)} metre pendulum released from ${amplitudeDeg} degrees.`}>
        <line x1={PIVOT.x - 60} x2={PIVOT.x + 60} y1={PIVOT.y} y2={PIVOT.y} stroke="var(--line-strong)" strokeWidth="3" />
        {/* A centimetre ruler beside the string, as in the experiment. */}
        <g stroke="var(--line-strong)">
          {Array.from({ length: 31 }, (_, i) => (
            <line key={i} x1={20} x2={i % 10 === 0 ? 34 : i % 5 === 0 ? 30 : 26} y1={PIVOT.y + i * 10} y2={PIVOT.y + i * 10} />
          ))}
        </g>
        {fan.map((th, i) => (
          <circle key={i} cx={PIVOT.x - r * Math.sin(th)} cy={PIVOT.y + r * Math.cos(th)} r="5" fill="var(--data)" opacity={0.35} />
        ))}
        {Array.from({ length: TRAIL / 2 }, (_, i) => (
          <circle key={i} ref={(el) => (trail.current[i] = el)} r="3.5" fill="var(--data)" opacity="0" />
        ))}
        <g ref={bob} transform={`rotate(${amplitudeDeg} ${PIVOT.x} ${PIVOT.y})`}>
          <line x1={PIVOT.x} y1={PIVOT.y} x2={PIVOT.x} y2={PIVOT.y + r} stroke="var(--ink-2)" strokeWidth="1.5" />
          <circle cx={PIVOT.x} cy={PIVOT.y + r} r="12" fill="var(--data)" stroke="var(--bg)" strokeWidth="2" />
        </g>
        <circle cx={PIVOT.x} cy={PIVOT.y} r="4" fill="var(--ink-2)" />
      </svg>
      <div className="flex flex-col gap-4">
        <div>
          <label htmlFor={lengthId} className="label">Length · <span className="readout normal-case text-slate-200">{length.toFixed(2)} m</span></label>
          <input id={lengthId} type="range" min={0.15} max={1} step={0.01} value={length} onChange={(e) => setLength(+e.target.value)} className="mt-2 h-11 w-full accent-cyan-400" />
        </div>
        <div>
          <label htmlFor={ampId} className="label">Release angle · <span className="readout normal-case text-slate-200">{amplitudeDeg}°</span></label>
          <input id={ampId} type="range" min={2} max={35} step={1} value={amplitudeDeg} onChange={(e) => setAmplitudeDeg(+e.target.value)} className="mt-2 h-11 w-full accent-cyan-400" />
        </div>
        <div className="flex flex-wrap gap-2">
          {!reduced && (
            <button className="btn-ghost min-h-11" aria-pressed={swinging} onClick={() => setSwinging((s) => !s)}>
              {swinging ? "Stop" : "Release"}
            </button>
          )}
          <button
            className="btn-ghost min-h-11"
            onClick={() => {
              setLength(0.6);
              setAmplitudeDeg(2);
            }}
          >
            Commons pendulum: 0.60 m, 2°
          </button>
        </div>
        <dl className="readout grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
          <dt className="text-slate-400">T {measured ? "measured" : "exact"}</dt>
          <dd className="text-white">{period.toFixed(3)} s</dd>
          <dt className="text-slate-400">2π√(L/g)</dt>
          <dd className="text-white">{small.toFixed(3)} s</dd>
          <dt className="text-slate-400">L from T</dt>
          <dd className="text-white">{lengthFromPeriod.toFixed(3)} m</dd>
        </dl>
        <p className="text-sm text-slate-400">
          {gap > 0.5 ? (
            <>
              At {amplitudeDeg}° the swing is <span className="text-amber-300">{gap.toFixed(1)}% slower</span> than the small-angle formula: read L from that formula and it
              comes out {(100 * ((exact / small) ** 2 - 1)).toFixed(1)}% too long. That is why PhysicsLens fits the swing instead.
            </>
          ) : (
            <>At small angles the formula holds: the period alone gives the length.</>
          )}
        </p>
        <p className="sr-only" aria-live="polite">{announce}</p>
      </div>
    </div>
  );
}
