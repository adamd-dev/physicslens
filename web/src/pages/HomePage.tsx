import RecentRuns from "../home/RecentRuns";
import { href } from "../router";

export default function HomePage({ showRuns = true }: { showRuns?: boolean }) {
  return (
    <div>
      <section className="py-8 lg:py-16">
        <p className="label">Measure · fit · check</p>
        <h1 tabIndex={-1} className="mt-3 text-4xl font-semibold tracking-tight focus:outline-none sm:text-5xl lg:text-6xl">Measure g with your phone.</h1>
        <p className="mt-4 max-w-xl text-lg text-slate-300">
          Film a ball drop, a throw or a pendulum. PhysicsLens tracks the ball, fits the physics with SciPy, and a Meta Muse Spark agent checks the fit before you
          see a number.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <a href={href("/analyze")} className="btn min-h-11 bg-gradient-to-r from-cyan-400 to-violet-500 px-5 text-base font-semibold text-slate-950 hover:brightness-110">
            Analyze a video
          </a>
          {showRuns && <a href={href("/history")} className="btn-ghost min-h-11 px-5 text-base">See past runs</a>}
        </div>
      </section>
      {showRuns && <RecentRuns />}
    </div>
  );
}
