import { useRef } from "react";
import { useInView } from "../hooks/useInView";
import { type Bench, G, G_BENCH, L_BENCH, L_TAPE, pct } from "./demoData";

function Ruler({ title, unit, lo, hi, reference, refLabel, rows }: { title: string; unit: string; lo: number; hi: number; reference: number; refLabel: string; rows: Bench[] }) {
  const wrap = useRef<HTMLDivElement>(null);
  const shown = useInView(wrap, "-10% 0px");
  const at = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  return (
    <div ref={wrap} className="instrument p-5">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="font-medium">{title}</h3>
        <span className="readout text-xs text-slate-400">{unit}</span>
      </div>
      <div className="relative mt-10 h-10">
        <div className="absolute inset-x-0 top-5 h-px bg-slate-600" />
        {Array.from({ length: 13 }, (_, i) => (
          <div key={i} className="absolute top-3 h-4 w-px bg-slate-700" style={{ left: `${(i / 12) * 100}%` }} />
        ))}
        <div className="absolute -top-6 bottom-0 w-0.5 bg-slate-200" style={{ left: at(reference) }}>
          <span className="readout absolute -top-5 left-1/2 -translate-x-1/2 whitespace-nowrap text-xs text-slate-200">{refLabel}</span>
        </div>
        {rows.map((r, i) => (
          <div
            key={r.label}
            className="absolute top-5 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-slate-950 bg-cyan-400 transition-[left] duration-[var(--dur-slow)] ease-[var(--ease-out)]"
            style={{ left: shown ? at(r.value) : at(reference), top: `${20 + i * 12}px` }}
            aria-hidden="true"
          />
        ))}
        <span className="readout absolute -bottom-2 left-0 text-xs text-slate-400">{lo.toFixed(2)}</span>
        <span className="readout absolute -bottom-2 right-0 text-xs text-slate-400">{hi.toFixed(2)}</span>
      </div>
      <ul className="mt-6 space-y-2">
        {rows.map((r) => (
          <li key={r.label} className="flex flex-wrap items-baseline justify-between gap-x-3 text-sm">
            <span className="text-slate-300">{r.label}</span>
            <span className="readout text-white">
              {r.value.toFixed(3)}
              {r.err != null && <span className="text-slate-400"> ± {r.err.toFixed(3)}</span>}
              <span className="ml-2 text-cyan-300">{pct(r.value, reference) > 0 ? "+" : "−"}{Math.abs(pct(r.value, reference)).toFixed(1)}%</span>
            </span>
            <span className="w-full text-xs text-slate-400">{r.points} points · {r.detail}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** How close the measurements came, on instrument rulers; the same numbers are in the table below for reading. */
export default function AccuracyRuler() {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Ruler title="Gravity from a dropped ball" unit="g · m/s²" lo={9.5} hi={10.1} reference={G} refLabel="9.81" rows={G_BENCH} />
      <Ruler title="Pendulum length from its swing" unit="L · m" lo={0.5} hi={0.7} reference={L_TAPE} refLabel="tape 0.60" rows={L_BENCH} />
    </div>
  );
}
