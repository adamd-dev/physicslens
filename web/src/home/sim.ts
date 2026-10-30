/** The physics behind the Home page demos: seeded noise, and a pendulum integrated without the small-angle shortcut. */

export const G = 9.81;

/** Seeded PRNG (mulberry32): every take of a recording shows the same dots. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A standard normal sample (Box–Muller): the tracker's centroid jitter. */
export function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

export interface PendulumState {
  theta: number; // rad from vertical
  omega: number; // rad/s
}

/** One RK4 step of θ'' = −(g/L)·sin θ − γ·θ'. */
export function stepPendulum(s: PendulumState, dt: number, length: number, gamma: number, g = G): PendulumState {
  const f = (th: number, om: number) => [om, -(g / length) * Math.sin(th) - gamma * om];
  const [k1t, k1o] = f(s.theta, s.omega);
  const [k2t, k2o] = f(s.theta + (dt / 2) * k1t, s.omega + (dt / 2) * k1o);
  const [k3t, k3o] = f(s.theta + (dt / 2) * k2t, s.omega + (dt / 2) * k2o);
  const [k4t, k4o] = f(s.theta + dt * k3t, s.omega + dt * k3o);
  return {
    theta: s.theta + (dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t),
    omega: s.omega + (dt / 6) * (k1o + 2 * k2o + 2 * k3o + k4o),
  };
}

/** The exact period at amplitude θ0: T = 4·√(L/g)·K(sin(θ0/2)), K by the arithmetic–geometric mean. */
export function pendulumPeriod(length: number, amplitude: number, g = G): number {
  let a = 1;
  let b = Math.cos(amplitude / 2);
  for (let i = 0; i < 12; i++) [a, b] = [(a + b) / 2, Math.sqrt(a * b)];
  return (2 * Math.PI * Math.sqrt(length / g)) / a;
}
