"""Draw web/landing-public/og-image.png (1200x630): a tracked drop, its fitted fall and the measured g.

Run with the project venv: .venv/bin/python web/scripts/og_image.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, INK, MUTED, DATA, MODEL = "#020617", "#f1f5f9", "#94a3b8", "#22d3ee", "#c084fc"
G = 9.81

rng = np.random.default_rng(7)
t = np.arange(0, 0.42, 1 / 120)
y = 0.5 * G * t**2 + rng.normal(0, 0.003, t.size)
g_fit = 2 * np.polyfit(t, y, 2)[0]

fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
ax = fig.add_axes([0.66, 0.13, 0.31, 0.74], facecolor=BG)
ax.plot(t, np.polyval(np.polyfit(t, y, 2), t), color=MODEL, lw=3, zorder=2)
ax.scatter(t, y, s=16, color=DATA, zorder=3)
ax.invert_yaxis()
ax.grid(color=MUTED, alpha=0.15)
for side in ax.spines.values():
    side.set_color("#334155")
ax.tick_params(colors=MUTED, labelsize=12)
ax.set_xlabel("t (s)", color=MUTED, fontsize=13)
ax.set_ylabel("fall (m)", color=MUTED, fontsize=13)

fig.text(0.05, 0.80, "ϕ  PhysicsLens", color=INK, fontsize=26, weight="bold")
fig.text(0.05, 0.60, "Measure gravity with\nyour phone.", color=INK, fontsize=42, weight="bold", va="top", linespacing=1.05)
# The headline number is the real BallDrop measurement; the plot beside it is an illustration.
fig.text(0.05, 0.30, "g = 9.728 m/s²", color=DATA, fontsize=34, family="monospace")
fig.text(0.05, 0.16, "tracked by Meta SAM 3.1 · fitted with SciPy\nchecked by a Meta Muse Spark agent", color=MUTED, fontsize=14, linespacing=1.4)
fig.text(0.05, 0.06, "measured from the BallDrop video · 0.8% below 9.81", color=MUTED, fontsize=13, family="monospace")

out = Path(__file__).resolve().parent.parent / "landing-public" / "og-image.png"
fig.savefig(out, facecolor=BG)
print(out, f"g_fit={g_fit:.3f}")
