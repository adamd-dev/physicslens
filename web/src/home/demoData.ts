/** Real numbers for the Home page. Every value here comes from a run in this repo, never from a mock-up.
 *  - BallDrop (OSP Tracker sample): motion tracker g = 9.728 ± 0.021 (fit), 23 points; SAM 3.1 box 9.653 ± 0.065.
 *  - Wikimedia Commons "Fadenpendelgeschwindigkeit (2 grad)" by Mathe-AC Lernwerkstatt Aachen (Rhetos), CC BY-SA 4.0: a 0.60 m
 *    pendulum; the Muse agent's run (runs/commons_pendulum_agent.json) gave L = 0.590 m from ω = 4.078 rad/s. */

export const G = 9.81;

export interface Bench {
  label: string;
  value: number;
  err: number | null;
  points: number | null;
  detail: string;
}

export const G_BENCH: Bench[] = [
  { label: "BallDrop · motion tracker", value: 9.728, err: 0.021, points: 23, detail: "± fit only · R² 1.0000" },
  { label: "BallDrop · SAM 3.1 box", value: 9.653, err: 0.065, points: 13, detail: "± fit and scale reading" },
];

export const L_BENCH: Bench[] = [
  { label: "Commons pendulum · Muse agent", value: 0.59, err: 0.002, points: 105, detail: "R² 0.994 · flagged: structured residuals" },
];
export const L_TAPE = 0.6;

export const pct = (value: number, ref: number) => (100 * (value - ref)) / ref;

/** The Muse Spark agent's real steps on the Commons pendulum, shortened for display. */
export const AGENT_STEPS: { call: string; out: string; tone: "info" | "bad" | "good" }[] = [
  { call: "describe_track()", out: "105 points over 4.16 s · x swings at 0.659 Hz", tone: "info" },
  { call: 'fit_pendulum(axis="x")', out: "fit_1 · ω 4.078 rad/s · R² 0.994", tone: "info" },
  { call: 'check_plausibility("fit_1")', out: "flagged: residuals are structured (runs z = −7.2)", tone: "bad" },
  { call: "fit_free_fall · fit_projectile · fit_pendulum(y)", out: "R² 0.027 · 0.015 · 0.862: all rejected", tone: "bad" },
  { call: 'final_answer("fit_1", accept_flagged)', out: "L = 0.590 m (tape: 0.60 m), the flag explained", tone: "good" },
];
