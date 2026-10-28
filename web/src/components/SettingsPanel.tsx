import { useState } from "react";
import type { AnalyzeRequest, Settings } from "../api";

export type Options = Pick<AnalyzeRequest, "start" | "end" | "capture_fps" | "model" | "axis" | "agent" | "height_m">;

function Num({ label, value, onChange, step = 1, placeholder }: { label: string; value: number | null; onChange: (v: number | null) => void; step?: number; placeholder?: string }) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <input
        className="input mt-1 font-mono"
        type="number"
        step={step}
        value={value ?? ""}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
      />
    </label>
  );
}

function Select<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: [T, string][]; onChange: (v: T) => void }) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <select className="input mt-1" value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map(([v, text]) => (
          <option key={v} value={v}>
            {text}
          </option>
        ))}
      </select>
    </label>
  );
}

interface Props {
  settings: Settings;
  options: Options;
  onSettings: (s: Settings) => void;
  onOptions: (o: Options) => void;
}

export default function SettingsPanel({ settings, options, onSettings, onOptions }: Props) {
  const [open, setOpen] = useState(false);
  const set = (patch: Partial<Settings>) => onSettings({ ...settings, ...patch });
  const opt = (patch: Partial<Options>) => onOptions({ ...options, ...patch });

  return (
    <div className="space-y-4">
      <label className="flex cursor-pointer items-center justify-between gap-3 rounded-xl border border-violet-500/30 bg-violet-500/10 px-4 py-3 focus-within:ring-2 focus-within:ring-violet-400">
        <div>
          <div className="text-sm font-semibold text-violet-200">Muse agent</div>
          <div className="text-sm text-violet-300">Meta's model chooses, checks and explains the fit</div>
        </div>
        <input type="checkbox" className="h-5 w-5 shrink-0 accent-violet-400" checked={options.agent} onChange={(e) => opt({ agent: e.target.checked })} />
      </label>

      <div className="grid grid-cols-2 gap-3">
        <Select label="Tracker" value={settings.tracker} options={[["motion", "Motion"], ["color", "Color"]]} onChange={(tracker) => set({ tracker })} />
        <Num label="Launch height (m)" value={options.height_m} step={0.01} placeholder="cup shot" onChange={(height_m) => opt({ height_m })} />
      </div>

      <button className="inline-flex min-h-6 items-center gap-1 rounded py-1 text-sm text-slate-400 hover:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span aria-hidden="true">{open ? "▾" : "▸"}</span> Advanced
      </button>
      {open && (
        <div className="grid grid-cols-2 gap-3">
          <Num label="Min area (px)" value={settings.min_area} onChange={(v) => set({ min_area: v ?? 1 })} />
          <Num label="Max area (px)" value={settings.max_area} placeholder="auto" onChange={(max_area) => set({ max_area })} />
          <Num label="Max jump (px)" value={settings.max_jump} placeholder="off" onChange={(max_jump) => set({ max_jump })} />
          <Select label="Interlaced fields" value={settings.fields} options={[["none", "No"], ["auto", "Auto"], ["top", "Top"], ["bottom", "Bottom"]]} onChange={(fields) => set({ fields })} />
          <Num label="Start (s)" value={options.start} step={0.1} onChange={(v) => opt({ start: v ?? 0 })} />
          <Num label="End (s)" value={options.end} step={0.1} placeholder="end" onChange={(end) => opt({ end })} />
          <Num label="Capture fps" value={options.capture_fps} placeholder="slow-mo, e.g. 240" onChange={(capture_fps) => opt({ capture_fps })} />
          {!options.agent && (
            <Select label="Model" value={options.model} options={[["auto", "Auto"], ["free_fall", "Free fall"], ["projectile", "Projectile"], ["pendulum", "Pendulum"]]} onChange={(model) => opt({ model })} />
          )}
          <Select label="Pendulum axis" value={options.axis} options={[["x", "x"], ["y", "y"]]} onChange={(axis) => opt({ axis })} />
          {settings.tracker === "color" && (
            <>
              <label className="block">
                <span className="label">HSV low</span>
                <input className="input mt-1 font-mono" placeholder="0,0,200" value={settings.hsv_lower.join(",")} onChange={(e) => set({ hsv_lower: triple(e.target.value, settings.hsv_lower) })} />
              </label>
              <label className="block">
                <span className="label">HSV high</span>
                <input className="input mt-1 font-mono" placeholder="180,60,255" value={settings.hsv_upper.join(",")} onChange={(e) => set({ hsv_upper: triple(e.target.value, settings.hsv_upper) })} />
              </label>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function triple(text: string, fallback: [number, number, number]): [number, number, number] {
  const v = text.split(",").map(Number);
  return v.length === 3 && v.every(Number.isFinite) ? (v as [number, number, number]) : fallback;
}
