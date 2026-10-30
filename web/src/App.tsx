import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { getConfig, listRuns, type SiteConfig } from "./api";
import AppHeader from "./components/AppHeader";
import AnalyzePage from "./pages/AnalyzePage";
import { useRoute } from "./router";

const HomePage = lazy(() => import("./pages/HomePage"));
const HistoryPage = lazy(() => import("./pages/HistoryPage"));
const RunDetailPage = lazy(() => import("./pages/RunDetailPage"));

const TITLES = { home: "PhysicsLens: Measure gravity with your phone", analyze: "Analyze a video · PhysicsLens", history: "History · PhysicsLens", run: "Run · PhysicsLens" };

export default function App() {
  const route = useRoute();
  const [running, setRunning] = useState(false);
  const [runCount, setRunCount] = useState<number | null>(null);
  const [site, setSite] = useState<SiteConfig>({ public: false, max_upload_mb: 500 });
  useEffect(() => {
    getConfig().then(setSite).catch(() => {});
  }, []);
  // Analyze mounts on first visit and then stays: an analysis in progress and the loaded video survive tab switches.
  const [analyzeMounted, setAnalyzeMounted] = useState(route.page === "analyze");

  const refreshCount = useCallback(() => {
    listRuns()
      .then((runs) => setRunCount(runs.length))
      .catch(() => setRunCount(null));
  }, []);
  useEffect(refreshCount, [refreshCount]);
  const onRunning = useCallback(
    (r: boolean) => {
      setRunning(r);
      if (!r) refreshCount(); // a finished run is now in History
    },
    [refreshCount],
  );

  // Only a new page scrolls to the top: a filter change on the same page keeps the reader's place.
  const pageKey = route.page === "run" ? `run/${route.id}` : route.page;
  useEffect(() => {
    if (route.page === "analyze") setAnalyzeMounted(true);
    document.title = TITLES[route.page];
    window.scrollTo({ top: 0 });
  }, [pageKey]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="mx-auto max-w-[1400px] px-5 pb-4">
      <AppHeader route={route} running={running} runCount={runCount} />
      <main id="main" tabIndex={-1} className="focus:outline-none">
        {analyzeMounted && (
          <div hidden={route.page !== "analyze"}>
            <AnalyzePage active={route.page === "analyze"} site={site} onRunning={onRunning} />
          </div>
        )}
        <Suspense fallback={<div className="panel h-64 animate-pulse" aria-busy="true" />}>
          {route.page === "home" && <HomePage />}
          {route.page === "history" && <HistoryPage query={route.query} shared={site.public} onChange={refreshCount} />}
          {route.page === "run" && <RunDetailPage id={route.id} onDeleted={refreshCount} />}
        </Suspense>
      </main>
      <footer className="mt-12 border-t border-slate-800 pt-4 text-center text-sm text-slate-400">
        Runtime models: Meta Muse Spark and SAM 3.1 via the Meta Model API · physics fits are deterministic (SciPy) · a hackathon project, not endorsed by Meta
      </footer>
    </div>
  );
}
