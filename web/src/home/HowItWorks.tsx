import { AGENT_STEPS } from "./demoData";

const STEPS = [
  { n: "1", title: "Film", text: "A phone on a stand, the ball, and something of known length in the same plane: a ruler or a sheet of A4." },
  { n: "2", title: "Track", text: "A motion or colour tracker finds the ball in every frame. When they lose it, Meta SAM 3.1 segments and follows it." },
  { n: "3", title: "Fit", text: "SciPy fits a fall, a throw or a swing to the track and gives g, the landing point or the length, with an uncertainty." },
  { n: "4", title: "Check", text: "A Meta Muse Spark agent calls the fit and check tools, rejects fits that fail, and explains what it kept. It never writes the numbers." },
];

/** The pipeline in four steps, and the agent's real reasoning on one clip. */
export default function HowItWorks() {
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <ol className="space-y-5">
        {STEPS.map((s) => (
          <li key={s.n} className="flex gap-4">
            <span className="readout grid h-8 w-8 shrink-0 place-items-center rounded-full border border-slate-700 text-sm text-cyan-300">{s.n}</span>
            <div>
              <h3 className="font-medium">{s.title}</h3>
              <p className="mt-1 text-sm leading-relaxed text-slate-400">{s.text}</p>
            </div>
          </li>
        ))}
      </ol>
      <figure className="instrument p-5">
        <figcaption className="label">Step 4 on a real clip · Commons pendulum, 105 points</figcaption>
        <ol className="readout mt-4 space-y-3 text-sm">
          {AGENT_STEPS.map((s, i) => (
            <li key={i} className="border-l-2 pl-3" style={{ borderColor: s.tone === "good" ? "var(--pass)" : s.tone === "bad" ? "var(--flag)" : "var(--line-strong)" }}>
              <div className="text-violet-300">{s.call}</div>
              <div className={s.tone === "good" ? "text-emerald-300" : s.tone === "bad" ? "text-amber-200" : "text-slate-300"}>→ {s.out}</div>
            </li>
          ))}
        </ol>
        <p className="mt-4 text-sm text-slate-400">
          In this run (Oct 28) Muse tried four models, rejected three, and kept the swing fit with its flag stated. With today's stricter rules it declines to certify the same fit and says why: the residuals are not random. Either way the length comes from the fit, not from the model's text.
        </p>
      </figure>
    </div>
  );
}
