/** Least squares in closed form, small enough to refit on every new point of the hero animation.
 *  The same models the app fits with SciPy: y = y0 + v0·t + g·t²/2, and x = x0 + vx·t. */

export interface FallFit {
  y0: number;
  v0: number;
  g: number;
  gErr: number; // one standard error, from the residual scatter
}

/** Solve the 3×3 system A·x = b (Gaussian elimination with partial pivoting); null when singular. */
function solve3(a: number[][], b: number[]): number[] | null {
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    if (Math.abs(m[pivot][col]) < 1e-15) return null;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const f = m[r][col] / m[col][col];
      for (let c = col; c < 4; c++) m[r][c] -= f * m[col][c];
    }
  }
  return m.map((row, i) => row[3] / row[i]);
}

/** Fit a fall through (t, y), y downward in metres. Null with fewer than three points. */
export function fitFall(t: number[], y: number[]): FallFit | null {
  const n = t.length;
  if (n < 3) return null;
  // Normal equations for the basis [1, t, t²].
  const s = [0, 0, 0, 0, 0];
  const r = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    let p = 1;
    for (let k = 0; k < 5; k++) {
      s[k] += p;
      if (k < 3) r[k] += p * y[i];
      p *= t[i];
    }
  }
  const ata = [
    [s[0], s[1], s[2]],
    [s[1], s[2], s[3]],
    [s[2], s[3], s[4]],
  ];
  const coef = solve3(ata, r);
  if (!coef) return null;
  const [c0, c1, c2] = coef;
  let sse = 0;
  for (let i = 0; i < n; i++) sse += (y[i] - (c0 + c1 * t[i] + c2 * t[i] * t[i])) ** 2;
  // Var(c2) = σ²·[(AᵀA)⁻¹]₂₂; the column of the inverse comes from solving against e₃.
  const e3 = solve3(ata, [0, 0, 1]);
  const sigma2 = n > 3 ? sse / (n - 3) : 0;
  const c2Err = e3 ? Math.sqrt(Math.max(0, sigma2 * e3[2])) : Infinity;
  return { y0: c0, v0: c1, g: 2 * c2, gErr: 2 * c2Err };
}

/** Straight-line fit: the horizontal motion of a throw. */
export function fitLine(t: number[], x: number[]): { intercept: number; slope: number } | null {
  const n = t.length;
  if (n < 2) return null;
  const mt = t.reduce((a, b) => a + b, 0) / n;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  let stt = 0;
  let stx = 0;
  for (let i = 0; i < n; i++) {
    stt += (t[i] - mt) ** 2;
    stx += (t[i] - mt) * (x[i] - mx);
  }
  if (stt === 0) return null;
  const slope = stx / stt;
  return { slope, intercept: mx - slope * mt };
}
