import type { RunStatus } from "../api";

const STYLE: Record<RunStatus, { text: string; cls: string }> = {
  ok: { text: "✓ checks pass", cls: "bg-emerald-500/15 text-emerald-300" },
  flagged: { text: "⚠ flagged", cls: "bg-amber-500/15 text-amber-300" },
  none: { text: "– no measurement", cls: "bg-slate-800 text-slate-300" },
};

/** Status in words as well as colour: colour alone never carries the meaning. */
export default function StatusBadge({ status, className = "" }: { status: RunStatus; className?: string }) {
  const { text, cls } = STYLE[status];
  return <span className={`inline-flex shrink-0 items-center whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-semibold ${cls} ${className}`}>{text}</span>;
}
