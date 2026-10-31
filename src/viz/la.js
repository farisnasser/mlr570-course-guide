// Small, exact-as-floats linear algebra for the visuals. 2x2 matrices are [[a, b], [c, d]] (rows); vectors [x, y].
// Every function here is mirrored by numpy in scripts/check/*.py on the notes' examples.

export const EPS = 1e-9;
export const det2 = ([[a, b], [c, d]]) => a * d - b * c;
export const tr2 = ([[a], [, d]]) => a + d;
export const mv = (M, v) => M.map(r => r.reduce((s, m, j) => s + m * v[j], 0));
export const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));
export const T = A => A[0].map((_, j) => A.map(r => r[j]));
export const I2 = () => [[1, 0], [0, 1]];
export const lerpM = (A, B, t) => A.map((r, i) => r.map((a, j) => a + (B[i][j] - a) * t));
export const rot = th => [[Math.cos(th), -Math.sin(th)], [Math.sin(th), Math.cos(th)]];
export const dot = (u, v) => u.reduce((s, a, i) => s + a * v[i], 0);
export const norm = v => Math.hypot(...v);
export const unit = v => { const n = norm(v); return n < EPS ? v.map(() => 0) : v.map(a => a / n); };
export const add = (u, v) => u.map((a, i) => a + v[i]);
export const sub = (u, v) => u.map((a, i) => a - v[i]);
export const scale = (v, s) => v.map(a => a * s);

export function inv2(M) {
  const d = det2(M); if (Math.abs(d) < EPS) return null;
  const [[a, b], [c, dd]] = M; return [[dd / d, -b / d], [-c / d, a / d]];
}

/** rank of a small dense matrix by Gaussian elimination with partial pivoting */
export function rank(M, tol = 1e-9) {
  const A = M.map(r => r.slice()); const m = A.length, n = A[0].length; let r = 0;
  for (let c = 0; c < n && r < m; c++) {
    let p = r; for (let i = r + 1; i < m; i++) if (Math.abs(A[i][c]) > Math.abs(A[p][c])) p = i;
    if (Math.abs(A[p][c]) < tol) continue;
    [A[r], A[p]] = [A[p], A[r]];
    for (let i = r + 1; i < m; i++) { const f = A[i][c] / A[r][c]; for (let j = c; j < n; j++) A[i][j] -= f * A[r][j]; }
    r++;
  }
  return r;
}

/** Real eigen-decomposition of a 2x2 matrix. Returns {real, vals:[l1,l2] (l1 >= l2), vecs:[v1,v2] unit} or {real:false, re, im}. */
export function eig2(M) {
  const [[a, b], [c, d]] = M, t = a + d, D = a * d - b * c, disc = t * t / 4 - D;
  if (disc < -1e-12) return { real: false, re: t / 2, im: Math.sqrt(-disc) };
  const s = Math.sqrt(Math.max(0, disc)), l1 = t / 2 + s, l2 = t / 2 - s;
  const vec = l => {
    // null space of (M - l I): pick the better-conditioned row
    const r1 = [a - l, b], r2 = [c, d - l];
    const r = Math.hypot(...r1) >= Math.hypot(...r2) ? r1 : r2;
    if (Math.hypot(...r) < 1e-12) return null;        // M = l I: every direction
    return unit([-r[1], r[0]]);
  };
  let v1 = vec(l1), v2 = vec(l2);
  if (!v1 && !v2) { v1 = [1, 0]; v2 = [0, 1]; }
  else if (Math.abs(l1 - l2) < 1e-12) v2 = v1 ? [-v1[1], v1[0]] : v2;
  const canon = v => (v[0] < -1e-12 || (Math.abs(v[0]) <= 1e-12 && v[1] < 0)) ? [-v[0], -v[1]] : v;
  return { real: true, vals: [l1, l2], vecs: [canon(v1), canon(v2)], repeated: Math.abs(l1 - l2) < 1e-12 };
}

/** SVD of a 2x2 matrix as rotation · diag · rotation, for animation:
 *  A = R(phi) · diag(s1, s2) · R(-theta), s1 >= |s2|, s2 < 0 when det A < 0.
 *  v1 = (cos theta, sin theta), v2 = (-sin theta, cos theta) are the input directions; u1 = (cos phi, sin phi).
 *  sigma = [s1, |s2|] are the singular values. */
export function svd2(A) {
  const [[a, b], [c, d]] = A;
  const E = (a + d) / 2, F = (a - d) / 2, G = (c + b) / 2, H = (c - b) / 2;
  const Q = Math.hypot(E, H), R = Math.hypot(F, G);
  const s1 = Q + R, s2 = Q - R;
  const a1 = Math.atan2(G, F), a2 = Math.atan2(H, E);
  const theta = (a1 - a2) / 2, phi = (a2 + a1) / 2;
  return { s1, s2, sigma: [s1, Math.abs(s2)], theta, phi,
    v1: [Math.cos(theta), Math.sin(theta)], v2: [-Math.sin(theta), Math.cos(theta)],
    u1: [Math.cos(phi), Math.sin(phi)], u2: [-Math.sin(phi), Math.cos(phi)] };
}

/** Jacobi eigen-decomposition of a symmetric n x n matrix: {vals (desc), vecs (columns, as arrays)} */
export function eigSym(S) {
  const n = S.length, A = S.map(r => r.slice()), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] ** 2;
    if (off < 1e-26) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const idx = [...Array(n).keys()].sort((i, j) => A[j][j] - A[i][i]);
  const canon = v => { const k = v.findIndex(x => Math.abs(x) > 1e-9); return k >= 0 && v[k] < 0 ? v.map(x => -x) : v; };
  return { vals: idx.map(i => A[i][i]), vecs: idx.map(i => canon(V.map(r => r[i]))) };
}

/** SVD of a real m x n matrix via the eigen-decomposition of A^T A (fine for the small matrices in the notes).
 *  Returns {s, U (list of u_i), V (list of v_i)}, singular values descending, v_i sign-canonical, u_i = A v_i / s_i. */
export function svd(A) {
  const AtA = mm(T(A), A), { vals, vecs } = eigSym(AtA);
  const s = vals.map(l => Math.sqrt(Math.max(0, l)));
  const U = vecs.map((v, i) => s[i] > 1e-10 ? mv(A, v).map(x => x / s[i]) : null);
  return { s, U, V: vecs };
}

export const outer = (u, v) => u.map(a => v.map(b => a * b));
export const frob = M => Math.sqrt(M.flat().reduce((s, x) => s + x * x, 0));
export const msub = (A, B) => A.map((r, i) => r.map((a, j) => a - B[i][j]));
export const madd = (A, B) => A.map((r, i) => r.map((a, j) => a + B[i][j]));
export const mscale = (A, s) => A.map(r => r.map(a => a * s));
export const zeros = (m, n) => Array.from({ length: m }, () => Array(n).fill(0));
