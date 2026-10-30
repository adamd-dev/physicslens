import { describe, expect, it } from "vitest";
import { fitFall, fitLine } from "./fit";
import { gaussian, mulberry32 } from "./sim";

const G = 9.81;
const times = (n: number, fps = 120) => Array.from({ length: n }, (_, i) => i / fps);

describe("fitFall", () => {
  it("recovers g exactly from a noiseless drop", () => {
    const t = times(40);
    const y = t.map((s) => 0.2 + 0.5 * s + 0.5 * G * s * s);
    const fit = fitFall(t, y)!;
    expect(fit.g).toBeCloseTo(G, 9);
    expect(fit.v0).toBeCloseTo(0.5, 9);
    expect(fit.y0).toBeCloseTo(0.2, 9);
  });

  it("converges as points arrive: the uncertainty shrinks", () => {
    const rand = mulberry32(7);
    const t = times(60);
    const y = t.map((s) => 0.5 * G * s * s + gaussian(rand) * 0.002);
    const early = fitFall(t.slice(0, 12), y.slice(0, 12))!;
    const late = fitFall(t, y)!;
    expect(late.g).toBeCloseTo(G, 1);
    expect(late.gErr).toBeLessThan(early.gErr);
  });

  it("needs three points", () => {
    expect(fitFall([0, 1], [0, 1])).toBeNull();
  });
});

describe("fitLine", () => {
  it("recovers a constant horizontal speed", () => {
    const t = times(20);
    const fit = fitLine(t, t.map((s) => 1 + 2.5 * s));
    expect(fit?.slope).toBeCloseTo(2.5, 9);
    expect(fit?.intercept).toBeCloseTo(1, 9);
  });
});
