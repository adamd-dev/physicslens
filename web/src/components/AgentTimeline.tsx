import { useEffect, useRef } from "react";
import type { AgentMessage } from "../api";

interface Step {
  id: string;
  name: string;
  args: Record<string, unknown>;
  thought: string;
  result?: Record<string, unknown>;
}

/** Pair each tool call with the tool reply that answers it; the model's text becomes the "thought". */
function toSteps(messages: AgentMessage[]): Step[] {
  const steps: Step[] = [];
  const byId = new Map<string, Step>();
  for (const m of messages) {
    if (m.role === "assistant") {
      m.tool_calls.forEach((call, i) => {
        const step: Step = {
          id: call.id || `${steps.length}`,
          name: call.name,
          args: parseArgs(call.arguments),
          thought: i === 0 ? m.content : "",
        };
        steps.push(step);
        byId.set(step.id, step);
      });
    } else if (m.role === "tool" && m.tool_call_id && byId.has(m.tool_call_id)) {
      try {
        byId.get(m.tool_call_id)!.result = JSON.parse(m.content);
      } catch {
        byId.get(m.tool_call_id)!.result = { error: m.content };
      }
    }
  }
  return steps;
}

function parseArgs(a: string | Record<string, unknown>): Record<string, unknown> {
  if (typeof a !== "string") return a ?? {};
  try {
    return JSON.parse(a || "{}");
  } catch {
    return { raw: a };
  }
}

function brief(step: Step): { text: string; tone: "ok" | "flag" | "error" | "info" } | null {
  const r = step.result;
  if (!r) return null;
  if ("error" in r) return { text: String(r.error), tone: "error" };
  if ("ok" in r) {
    const problems = (r.problems as string[]) ?? [];
    return r.ok ? { text: "all checks pass", tone: "ok" } : { text: problems.join(" · "), tone: "flag" };
  }
  if ("params" in r) {
    const params = Object.entries(r.params as Record<string, number | null>)
      .filter(([k]) => ["g", "vx", "omega"].includes(k))
      .map(([k, v]) => `${k} = ${v?.toFixed(3)}`);
    const extra = r.length_m != null ? [`L = ${(r.length_m as number).toFixed(3)} m`] : [];
    return { text: [...params, ...extra, `R² ${r.r2 == null ? "n/a" : (r.r2 as number).toFixed(4)}`, `${r.n_points} pts`].join(" · "), tone: "info" };
  }
  if ("n_points" in r) return { text: `${r.n_points} points · ${r.duration_s} s · x ${r.x_span_m} m · y ${r.y_span_m} m`, tone: "info" };
  if ("landing_distance_m" in r) return { text: `lands ${(r.landing_distance_m as number).toFixed(3)} m away`, tone: "info" };
  return { text: JSON.stringify(r).slice(0, 120), tone: "info" };
}

const TONE = {
  ok: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  flag: "border-amber-500/40 bg-amber-500/10 text-amber-200",
  error: "border-rose-500/40 bg-rose-500/10 text-rose-300",
  info: "border-slate-700 bg-slate-800/50 text-slate-300",
};

export default function AgentTimeline({ messages, running }: { messages: AgentMessage[]; running: boolean }) {
  const steps = toSteps(messages);
  const box = useRef<HTMLDivElement>(null);
  // Follow new steps inside the panel only; scrollIntoView would drag the whole page down.
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [messages.length]);

  if (!steps.length && !running) return null;
  return (
    <div ref={box} role="log" aria-live="polite" aria-relevant="additions" aria-label="Agent reasoning" className="max-h-[460px] space-y-3 overflow-y-auto pr-1">
      {steps.map((step, i) => {
        const b = brief(step);
        const final = step.name === "final_answer";
        const args = Object.entries(step.args).filter(([k]) => k !== "summary");
        return (
          <div key={step.id + i} className="rise flex gap-3">
            <div className="flex flex-col items-center">
              <div className={`grid h-7 w-7 place-items-center rounded-full text-xs font-semibold ${final ? "bg-violet-500 text-white" : "bg-slate-800 text-slate-300"}`}>
                {i + 1}
              </div>
              {i < steps.length - 1 && <div className="w-px flex-1 bg-slate-800" />}
            </div>
            <div className="min-w-0 flex-1 pb-3">
              {step.thought && <p className="mb-1 text-sm italic text-slate-400">“{step.thought}”</p>}
              <div className="break-words font-mono text-sm text-cyan-300">
                {step.name}
                <span className="text-slate-400">({args.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})</span>
              </div>
              {b && <div className={`mt-1.5 rounded-md border px-2.5 py-1.5 text-sm ${TONE[b.tone]}`}>{b.text}</div>}
              {final && typeof step.args.summary === "string" && (
                <p className="mt-1.5 text-sm text-slate-300">
                  {step.args.accept_flagged ? <span className="font-semibold text-amber-300">Accepted with flag: </span> : null}
                  {step.args.summary}
                </p>
              )}
            </div>
          </div>
        );
      })}
      {running && (
        <div className="flex items-center gap-2 pl-10 text-sm text-slate-400">
          <span aria-hidden="true" className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-400" /> Muse is thinking…
        </div>
      )}
    </div>
  );
}
