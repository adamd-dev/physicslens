import HomePage from "../pages/HomePage";

/** The public page: the Home content without the parts that need the local server (runs, analysis). */
export default function Landing() {
  return (
    <div className="mx-auto max-w-[1400px] px-5 pb-4">
      <header className="-mx-5 mb-6 border-b border-slate-800 px-5 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <a href="#top" className="flex items-center gap-3 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400">
            <span aria-hidden="true" className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-cyan-400 to-violet-500 text-lg font-bold text-slate-950">ϕ</span>
            <span className="text-lg font-semibold leading-tight">PhysicsLens</span>
          </a>
          <nav aria-label="Sections">
            <ul className="flex flex-wrap items-center gap-1 text-sm">
              {[["#how-it-works", "How it works"], ["#results", "Results"], ["#faq", "FAQ"], ["https://github.com/adamd-dev/physicslens", "Code"]].map(([to, label]) => (
                <li key={to}>
                  <a href={to} className="inline-flex min-h-10 items-center rounded-lg px-3 text-slate-400 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400">{label}</a>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>
      <main id="top">
        <HomePage local={false} />
      </main>
      <footer className="mt-12 border-t border-slate-800 pt-4 text-center text-sm text-slate-400">
        Runtime models: Meta Muse Spark and SAM 3.1 via the Meta Model API · physics fits are deterministic (SciPy) · a hackathon project by Adam D., not endorsed by Meta
      </footer>
    </div>
  );
}
