import AccuracyRuler from "../home/AccuracyRuler";
import { G, G_BENCH, L_BENCH, L_TAPE, pct } from "../home/demoData";
import Faq from "../home/Faq";
import HeroFit from "../home/HeroFit";
import HowItWorks from "../home/HowItWorks";
import MetaStrip from "../home/MetaStrip";
import PendulumLab from "../home/PendulumLab";
import RecentRuns from "../home/RecentRuns";
import { useFocusHeading } from "../hooks/useFocusHeading";
import { href } from "../router";

function Section({ id, label, title, children }: { id: string; label: string; title: string; children: React.ReactNode }) {
  return (
    <section aria-labelledby={id} className="border-t border-slate-800 py-12 lg:py-20">
      <p className="label">{label}</p>
      <h2 id={id} className="mb-6 mt-2 text-2xl font-semibold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

/** Home: what PhysicsLens does, shown with the physics itself. `local` adds the parts that need the local server. */
export default function HomePage({ local = true }: { local?: boolean }) {
  const heading = useFocusHeading(local); // a page opened from a link keeps the browser's own focus
  const appLink = (path: string) => (local ? href(path) : "#get-the-code");
  return (
    <div>
      <section className="grid items-center gap-8 pb-12 pt-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:py-16">
        <div>
          <p className="label">Measure · fit · check</p>
          <h1 ref={heading} tabIndex={-1} className="mt-3 text-4xl font-semibold tracking-tight focus:outline-none sm:text-5xl lg:text-6xl">
            Measure gravity with your phone.
          </h1>
          <p className="mt-4 max-w-xl text-lg leading-relaxed text-slate-300">
            Film a ball drop, a throw or a pendulum. From the video, PhysicsLens tracks the ball, fits the physics with SciPy, and a Meta Muse Spark agent checks the fit before you see a
            number.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <a href={appLink("/analyze")} className="btn min-h-11 bg-gradient-to-r from-cyan-400 to-violet-500 px-5 text-base font-semibold text-slate-950 hover:brightness-110">
              {local ? "Analyze a video" : "Get the code"}
            </a>
            {local && <a href={href("/history")} className="btn-ghost min-h-11 px-5 text-base">See past runs</a>}
          </div>
        </div>
        <HeroFit />
      </section>

      <Section id="how-it-works" label="How it works" title="From a phone video to a checked number">
        <HowItWorks />
      </Section>

      <Section id="pendulum-lab" label="Try it" title="Why fit the swing instead of using a formula">
        <PendulumLab />
      </Section>

      <Section id="results" label="Results" title="How accurate is it?">
        <AccuracyRuler />
        <div className="mt-6 overflow-x-auto" tabIndex={0} role="region" aria-label="Accuracy table">
        <table className="readout w-full min-w-[560px] text-left text-sm">
          <caption className="mb-2 text-left font-sans text-sm text-slate-400">
            Measured from video against the reference. Two clips, not a benchmark; ± is the fit (and scale reading where stated), perspective not included.
          </caption>
          <thead className="text-slate-400">
            <tr>
              <th scope="col" className="py-2 font-normal">Clip</th>
              <th scope="col" className="py-2 font-normal">Quantity</th>
              <th scope="col" className="py-2 font-normal">Measured</th>
              <th scope="col" className="py-2 font-normal">Reference</th>
              <th scope="col" className="py-2 font-normal">Error</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800 text-slate-200">
            {G_BENCH.map((r) => (
              <tr key={r.label}>
                <th scope="row" className="py-2 font-sans font-normal">{r.label}</th>
                <td>g</td>
                <td>{r.value.toFixed(3)} m/s²</td>
                <td>{G} m/s²</td>
                <td>{pct(r.value, G).toFixed(1)}%</td>
              </tr>
            ))}
            {L_BENCH.map((r) => (
              <tr key={r.label}>
                <th scope="row" className="py-2 font-sans font-normal">{r.label}</th>
                <td>length L</td>
                <td>{r.value.toFixed(3)} m</td>
                <td>{L_TAPE.toFixed(2)} m (tape)</td>
                <td>{pct(r.value, L_TAPE).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
        <p className="mt-3 text-xs text-slate-400">
          Pendulum clip: “Fadenpendelgeschwindigkeit (2 grad)” by Mathe-AC Lernwerkstatt Aachen, Wikimedia Commons, CC BY 4.0. BallDrop: Open Source Physics Tracker sample.
        </p>
      </Section>

      <Section id="built-with" label="Built with" title="Meta models where they help, SciPy for the numbers">
        <MetaStrip />
      </Section>

      {local && <RecentRuns />}

      <Section id="faq" label="FAQ" title="Frequently asked questions">
        <Faq />
      </Section>

      <Section id="get-the-code" label="Get the code" title="Run it on your computer">
        <p className="max-w-2xl text-sm leading-relaxed text-slate-400">
          PhysicsLens runs locally: a Python server with the trackers and fits, and this web app. Set <code className="readout text-slate-200">MODEL_API_KEY</code> in{" "}
          <code className="readout text-slate-200">.env</code> for the Meta models, then start <code className="readout text-slate-200">physicslens-ui</code>.
        </p>
        <a href="https://github.com/adamd-dev/physicslens" className="btn-ghost mt-4 min-h-11">Source code on GitHub</a>
      </Section>
    </div>
  );
}
