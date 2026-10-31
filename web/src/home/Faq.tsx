/** Visible FAQ. The landing page's FAQPage JSON-LD is built from this same list, so the two never differ. */
export const FAQ: { q: string; a: string }[] = [
  {
    q: "How does PhysicsLens measure g from a video?",
    a: "It finds the ball in every frame, turns pixels into metres with a length you mark on the video (two clicks on a ruler), and fits y = y0 + v0·t + g·t²/2 with SciPy. The fitted g comes with an uncertainty from the fit and from reading the scale.",
  },
  {
    q: "How accurate is it?",
    a: "On the BallDrop sample video it measured g = 9.728 m/s², 0.8% below 9.81. On a Wikimedia Commons video of a 0.60 m pendulum it measured L = 0.590 m, 1.6% short. These are two clips, not a benchmark; the depth difference between the ball and the ruler is the main remaining error.",
  },
  {
    q: "What do I need to film an experiment?",
    a: "A phone held still, side-on to the motion, and something of known length in the same plane as the ball. Slow motion helps: give its real frame rate, for example 240 fps, so the clock is right.",
  },
  {
    q: "What do Meta SAM 3.1 and Meta Muse Spark do here?",
    a: "SAM 3.1 segments and follows the ball when you box it, or when the simple trackers lose it. Muse Spark is an agent that calls the fit and check tools, rejects fits that fail the checks, and explains the result. The numbers always come from SciPy, never from the model's text.",
  },
  {
    q: "Is my video uploaded anywhere?",
    a: "The app runs on your own computer and keeps its runs there. The clip is sent to the Meta Model API only when you choose the SAM 3.1 tracker; the Muse agent receives the tracked positions and fit results, not the video.",
  },
];

export default function Faq() {
  return (
    <div className="divide-y divide-slate-800 border-y border-slate-800">
      {FAQ.map((item) => (
        <details key={item.q} className="group py-4">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-4 font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400">
            <h3>{item.q}</h3>
            <span aria-hidden="true" className="text-slate-400 transition-transform duration-[var(--dur-fast)] group-open:rotate-45">+</span>
          </summary>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">{item.a}</p>
        </details>
      ))}
    </div>
  );
}
