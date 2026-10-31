// Probability helpers for the visuals: exact pdf/cdf of the notes' named distributions and a seeded RNG.
export function erf(x) {   // power series below 3, continued fraction for erfc above (|error| < 1e-14)
  const s = Math.sign(x), a = Math.abs(x);
  if (a < 3) { let sum = a, term = a, n = 0; while (Math.abs(term) > 1e-17 * Math.abs(sum)) { n++; term *= -a * a / n; sum += term / (2 * n + 1); } return s * 2 / Math.sqrt(Math.PI) * sum; }
  let f = a, C = a, D = 0;   // erfc(a) = exp(-a^2)/sqrt(pi) / (a + (1/2)/(a + 1/(a + (3/2)/(a + ...))))
  for (let k = 1; k < 300; k++) { const an = k / 2; D = a + an * D; D = 1 / D; C = a + an / C; const del = C * D; f *= del; if (Math.abs(del - 1) < 1e-16) break; }
  return s * (1 - Math.exp(-a * a) / Math.sqrt(Math.PI) / f);
}
export const normPdf = (x, mu = 0, s = 1) => Math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
export const normCdf = (x, mu = 0, s = 1) => 0.5 * (1 + erf((x - mu) / (s * Math.SQRT2)));
export const DIST = {
  normal: { name: 'Normal$(\mu,\sigma^2)$', pdf: (x, p) => normPdf(x, p.mu, p.s), cdf: (x, p) => normCdf(x, p.mu, p.s) },
  uniform: { name: 'Uniform$(a,b)$', pdf: (x, p) => (x >= p.a && x <= p.b ? 1 / (p.b - p.a) : 0), cdf: (x, p) => Math.max(0, Math.min(1, (x - p.a) / (p.b - p.a))) },
  exponential: { name: 'Exponential$(\lambda)$', pdf: (x, p) => (x < 0 ? 0 : p.l * Math.exp(-p.l * x)), cdf: (x, p) => (x < 0 ? 0 : 1 - Math.exp(-p.l * x)) },
};
/** mulberry32 seeded RNG */
export function rng(seed = 1) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
export function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
export const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
export const varPop = a => { const m = mean(a); return a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length; };
export const varSample = a => { const m = mean(a); return a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1); };
