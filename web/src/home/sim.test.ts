import { describe, expect, it } from "vitest";
import { mulberry32, pendulumPeriod, stepPendulum } from "./sim";

const G = 9.81;

describe("mulberry32", () => {
  it("repeats for a seed, so every recording take is the same", () => {
    const a = mulberry32(7);
    const b = mulberry32(7);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
});

/** Period from upward zero crossings of a simulated swing. */
function simulatedPeriod(length: number, amplitudeRad: number): number {
  let state = { theta: amplitudeRad, omega: 0 };
  const dt = 1 / 240;
  const crossings: number[] = [];
  for (let i = 0; i < 240 * 20; i++) {
    const next = stepPendulum(state, dt, length, 0);
    if (state.theta < 0 && next.theta >= 0) crossings.push((i + state.theta / (state.theta - next.theta)) * dt);
    state = next;
  }
  return (crossings[crossings.length - 1] - crossings[0]) / (crossings.length - 1);
}

describe("stepPendulum", () => {
  it("swings with 2π√(L/g) at small angles", () => {
    expect(simulatedPeriod(0.6, (5 * Math.PI) / 180)).toBeCloseTo(2 * Math.PI * Math.sqrt(0.6 / G), 2);
  });

  it("swings slower than the small-angle formula at 30°, as a real pendulum does", () => {
    const ratio = simulatedPeriod(0.6, Math.PI / 6) / (2 * Math.PI * Math.sqrt(0.6 / G));
    expect(ratio).toBeGreaterThan(1.015);
    expect(ratio).toBeCloseTo(pendulumPeriod(0.6, Math.PI / 6) / (2 * Math.PI * Math.sqrt(0.6 / G)), 3);
  });
});
