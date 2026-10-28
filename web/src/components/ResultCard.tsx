import type { Result } from "../api";

const G = 9.81;
const MODEL_NAME = { free_fall: "Free fall", projectile: "Projectile", pendulum: "Pendulum" };

function Quantity({ name, value, err, unit, note }: { name: string; value: number; err?: number | null; unit: string; note?: string }) {
  return (
    <div>
      <div className="label">{name}</div>
      <div className="mt-1 whitespace-nowrap font-mono text-xl font-semibold text-white">
        {value.toFixed(3)}
        {err != null && <span className="text-base text-slate-400"> ± {err.toFixed(3)}</span>}
        <span className="ml-1 text-sm font-normal text-slate-400">{unit}</span>
      </div>
      {note && <div className="text-sm text-slate-400">{note}</div>}
    </div>
  );
}

export default function ResultCard({ result, cost }: { result: Result; cost: number | null }) {
  const p = result.params;
  const s = result.stderr_total;
  const fitErr = result.stderr.g;
  const dev = result.deviation_pct.g;
  const ok = result.check.ok;
  return (
    <div role="status" className="rise panel p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="label">Result · {result.source === "agent" ? `Muse agent, ${result.steps} steps` : "deterministic"}</h2>
          <div className="text-lg font-semibold">{MODEL_NAME[result.model]}</div>
        </div>
        <span className={`shrink-0 rounded-full px-2.5 py-1 text-sm font-semibold ${ok ? "bg-emerald-500/15 text-emerald-300" : "bg-amber-500/15 text-amber-300"}`}>
          {ok ? "✓ all checks pass" : "⚠ flagged"}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
        {p.g != null && (
          <div>
            <Quantity name="gravity g" value={p.g} err={s.g} unit="m/s²" note={dev != null ? `${Math.abs(dev).toFixed(1)}% from ${G}` : undefined} />
            {fitErr != null && (
              <div className="text-xs text-slate-500">
                ± = fit {fitErr.toFixed(3)} and scale {(Math.abs(p.g) * result.scale_rel_err).toFixed(3)} · perspective not included
              </div>
            )}
          </div>
        )}
        {result.model === "projectile" && p.vx != null && <Quantity name="horizontal speed" value={p.vx} err={s.vx} unit="m/s" />}
        {result.length_m != null && <Quantity name="pendulum length" value={result.length_m} unit="m" />}
        {result.period_s != null && <Quantity name="period" value={result.period_s} unit="s" />}
        {result.landing && (
          <Quantity name="predicted landing" value={result.landing.landing_distance_m} err={result.landing.uncertainty_m} unit="m" note={`flight ${result.landing.flight_time_s.toFixed(3)} s`} />
        )}
      </div>

      <div className="mt-4 flex flex-wrap gap-x-4 gap-y-1 font-mono text-sm text-slate-400">
        <span>R² {result.r2?.toFixed(5)}</span>
        <span>RMSE {(result.rmse_m * 1000).toFixed(2)} mm</span>
        {result.check.runs_z != null && <span>runs z {result.check.runs_z}</span>}
        {cost != null && <span className="ml-auto">{cost === 0 ? "cached · $0" : `$${cost.toFixed(4)}`}</span>}
      </div>

      {!ok && (
        <ul className="mt-4 space-y-1 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-200">
          {result.check.problems.map((problem) => (
            <li key={problem}>• {problem}</li>
          ))}
        </ul>
      )}

      {result.summary && (
        <div className="mt-4 border-t border-slate-800 pt-4">
          <div className="label">Written by the model, not verified</div>
          <p className="mt-1 text-sm leading-relaxed text-slate-300">{result.summary}</p>
        </div>
      )}
    </div>
  );
}
