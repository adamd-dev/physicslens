const PARTS = [
  { name: "Meta SAM 3.1", role: "Segments and follows the ball from one box, when the motion tracker loses it." },
  { name: "Meta Muse Spark", role: "The agent: calls the fit and check tools, rejects bad fits, writes the explanation." },
  { name: "SciPy", role: "The numbers: deterministic least-squares fits, never generated text." },
];

export default function MetaStrip() {
  return (
    <ul className="grid gap-4 md:grid-cols-3">
      {PARTS.map((p) => (
        <li key={p.name} className="border-t border-slate-700 pt-4">
          <div className="font-semibold">{p.name}</div>
          <p className="mt-1 text-sm text-slate-400">{p.role}</p>
        </li>
      ))}
    </ul>
  );
}
