import type { RunSummary } from "../api";

export const MODEL_LABEL: Record<string, string> = { free_fall: "Free fall", projectile: "Throw", pendulum: "Pendulum" };
export const TRACKER_LABEL: Record<string, string> = { motion: "motion", color: "color", sam: "SAM 3.1" };

/** "g 9.728 ± 0.034 m/s² · −0.8%", or what went wrong for a run with no measurement. */
export function keyText(run: RunSummary): string {
  if (!run.key || run.key.value == null) return run.error ? `${run.error.stage}: ${run.error.message}` : "no value";
  const err = run.key.err != null ? ` ± ${run.key.err.toFixed(3)}` : "";
  const dev = run.dev_pct != null ? ` · ${run.dev_pct > 0 ? "+" : "−"}${Math.abs(run.dev_pct).toFixed(1)}%` : "";
  return `${run.key.name} ${run.key.value.toFixed(3)}${err} ${run.key.unit}${dev}`;
}

export function costText(cost: number | null): string {
  if (cost == null) return "";
  return cost === 0 ? "cached · $0" : `$${cost.toFixed(4)}`;
}

const RTF = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

export function relativeTime(iso: string, now = Date.now()): string {
  const seconds = (new Date(iso).getTime() - now) / 1000;
  const steps: [Intl.RelativeTimeFormatUnit, number][] = [["day", 86400], ["hour", 3600], ["minute", 60]];
  for (const [unit, size] of steps) {
    if (Math.abs(seconds) >= size) return RTF.format(Math.round(seconds / size), unit);
  }
  return "just now";
}

export function runTitle(run: RunSummary): string {
  return `${run.model ? MODEL_LABEL[run.model] ?? run.model : "No measurement"} · ${run.video_name || "video"}`;
}
