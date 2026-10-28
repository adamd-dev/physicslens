import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import type { Result, TrackEvent } from "../api";

const AXIS = { stroke: "#475569", fontSize: 11, tickLine: false };

/** Tracked position and the fitted curve on the axis the model describes, plus residuals in mm. */
export default function Charts({ track, result }: { track: TrackEvent; result: Result }) {
  const useX = result.model === "pendulum" && result.axis === "x";
  const fitted = new Map(result.curve.t_play.map((t, i) => [t.toFixed(4), (useX ? result.curve.x_px[i] : result.curve.y_px[i]) / track.px_per_m]));
  const rows = track.t_play.map((t, i) => {
    const measured = useX ? track.x_m[i] : track.y_m[i];
    const fit = fitted.get(t.toFixed(4));
    return { t, measured, fit, residual: fit == null ? undefined : (measured - fit) * 1000 };
  });

  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
      <div className="panel p-3 lg:col-span-2">
        <div className="label mb-2">{useX ? "x position" : "height (y, down)"} · m</div>
        <ResponsiveContainer width="100%" height={190}>
          <ComposedChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => v.toFixed(2)} {...AXIS} />
            <YAxis reversed={!useX} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} {...AXIS} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} formatter={(v: number) => v.toFixed(4)} labelFormatter={(t: number) => `t = ${t.toFixed(3)} s`} />
            <Line dataKey="fit" stroke="#c084fc" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
            <Scatter dataKey="measured" fill="#22d3ee" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="panel p-3">
        <div className="label mb-2">residuals · mm</div>
        <ResponsiveContainer width="100%" height={190}>
          <ComposedChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => v.toFixed(1)} {...AXIS} />
            <YAxis domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(1)} {...AXIS} />
            <ReferenceLine y={0} stroke="#475569" />
            <Scatter dataKey="residual" fill={result.check.ok ? "#34d399" : "#fbbf24"} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
