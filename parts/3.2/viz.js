// Part 3.2 OLS, WLS & Ridge — interactive visuals: OLS as orthogonal projection on the lecture's X, y; WLS on
// heteroscedastic data; ridge paths on the lecture's 4x3 example (recomputed in scripts/check/3_2.py).
import { Plane, Chart, COL, FONT } from '../../src/viz/plane.js';
import * as L from '../../src/viz/la.js';
import { el, T, TD, fmt, matTex, vecTex, seg, slider, buttons, readout, legend } from '../../src/viz/ui.js';
import { rng, gauss } from '../../src/viz/stats.js';
import { loadThree, Scene3 } from '../../src/viz/three-lite.js';

const H = (f, h) => { f.stage.style.height = h; };
const MONO = '600 11.5px "JetBrains Mono", monospace';
function split(f, titles) {
  const sp = el('div', 'split'); f.stage.appendChild(sp);
  return titles.map(t => { const d = el('div'); sp.appendChild(d); if (t) d.appendChild(el('div', 'stage-t', t)); return d; });
}
/** small dense solve (Gaussian elimination with partial pivoting) */
export function solve(A, b) {
  const n = A.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i; [M[c], M[p]] = [M[p], M[c]];
    for (let i = c + 1; i < n; i++) { const f = M[i][c] / M[c][c]; for (let j = c; j <= n; j++) M[i][j] -= f * M[c][j]; }
  }
  const x = Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; } return x;
}
const XtX = (X, w) => X[0].map((_, i) => X[0].map((_, j) => X.reduce((s, r, k) => s + (w ? w[k] : 1) * r[i] * r[j], 0)));
const Xty = (X, y, w) => X[0].map((_, i) => X.reduce((s, r, k) => s + (w ? w[k] : 1) * r[i] * y[k], 0));
export const ridge = (X, y, lam) => solve(XtX(X).map((r, i) => r.map((v, j) => v + (i === j ? lam : 0))), Xty(X, y));
export const wls = (X, y, w) => solve(XtX(X, w), Xty(X, y, w));

// ---------------------------------------------------------------------------------------------------------------
const projection = {
  title: 'OLS is an orthogonal projection onto $\\operatorname{Col}(X)$',
  async mount(f) {
    H(f, 'clamp(330px, 50vw, 440px)');
    const THREE = await loadThree();
    const [hl, hr] = split(f, [`${T('\\mathbb R^3')}: ${T('\\mathbf y')} and the plane ${T('\\operatorname{Col}(X)')}`, `${T('\\boldsymbol\\beta=(\\beta_0,\\beta_1)')} and contours of RSS`]);
    const one = [1, 1, 1], xv = [1, 2, 3], y = [1, 2, 2], X = [[1, 1], [1, 2], [1, 3]];
    const bhat = solve(XtX(X), Xty(X, y));
    let beta = [0.2, 0.9];
    const Xb = b => one.map((o, i) => b[0] * o + b[1] * xv[i]);
    // look at the plane from the side of its normal n = (-1,2,-1) so 1, x, y and the residual spread out
    const nrm = L.unit([-1, 2, -1]), dir = L.unit([1, 1.6, 2]), side = L.unit([nrm[1] * dir[2] - nrm[2] * dir[1], nrm[2] * dir[0] - nrm[0] * dir[2], nrm[0] * dir[1] - nrm[1] * dir[0]]);
    const tgt = [0.95, 1.45, 1.9], S = new Scene3(hl, { cam: L.add(tgt, L.add(L.scale(side, 2.6), L.scale(nrm, 1.9))), target: tgt, minDist: 1 });   // close to the tips, where the right angle is
    S.axes(3.2);
    const u1 = L.unit(one), u2 = L.unit(L.sub(xv, L.scale(u1, L.dot(xv, u1))));
    const pl = S.planeSpan(u1, u2, 0x7cc4ff, 3.4, 0.14);
    const a1 = S.arrow(0x7cc4ff, { r: 0.022, head: 0.16 }); a1.set([0, 0, 0], one);
    const a2 = S.arrow(0x7cc4ff, { r: 0.022, head: 0.16 }); a2.set([0, 0, 0], xv);
    const ay = S.arrow(0xe8ecf8), ab = S.arrow(0xb28cff), ah = S.arrow(0xff7a59, { r: 0.028 });
    const res = S.line([[0, 0, 0]], 0xff5d73, { dash: [0.08, 0.06] }), eLine = S.line([[0, 0, 0]], 0x2fb7a6);
    ay.set([0, 0, 0], y); ah.set([0, 0, 0], Xb(bhat)); S.setLine(eLine, [Xb(bhat), y]);
    const lab = (t, p, c) => { const l = S.label(t); l.p = p; l.e.style.color = c; return l; };
    lab('1', L.scale(one, 1.12), COL.span); lab('x', L.scale(xv, 1.06), COL.span); lab('y', L.scale(y, 1.08), COL.ink); lab('ŷ = P_X y', L.add(Xb(bhat), [0.25, 0, -0.25]), COL.c2);
    const lb = lab('Xβ', [0, 0, 0], COL.free);
    const C = new Chart(hr, { x: [-1.2, 2.4], y: [-0.4, 1.6], xlabel: 'β₀', ylabel: 'β₁', pad: [24, 12, 26, 36] });
    const A2 = XtX(X), e2 = L.eig2(A2), rss = b => { const r = L.sub(y, Xb(b)); return L.dot(r, r); }, rmin = rss(bhat);
    C.handle({ get: () => beta, set: p => { beta = p; upd(); }, color: COL.free });
    C.render = () => {
      for (const lev of [0.05, 0.2, 0.5, 1, 2, 4, 8]) {
        C.path(Array.from({ length: 121 }, (_, i) => { const t = i * Math.PI / 60; return L.add(bhat, L.add(L.scale(e2.vecs[0], Math.sqrt(lev / e2.vals[0]) * Math.cos(t)), L.scale(e2.vecs[1], Math.sqrt(lev / e2.vals[1]) * Math.sin(t)))); }), 'rgba(255,107,107,.35)', { lw: 1 });
      }
      C.dot(bhat, 5, COL.c2); C.label(bhat, 'β̂', COL.c2, { dx: 8, dy: -10, font: FONT.math });
      const g = L.scale(L.mv(L.T(X), L.sub(y, Xb(beta))), -2), gn = L.norm(g);
      if (gn > 1e-6) C.arrow(beta, L.sub(beta, L.scale(g, 0.35 / Math.max(1, gn))), COL.eig, { lw: 2, head: 8 });
    };
    buttons(f.side, [{ label: 'drop the perpendicular', pri: true, on: () => { const b0 = beta.slice(); const t0 = performance.now(); const step = () => { const u = Math.min(1, (performance.now() - t0) / 700); beta = L.add(b0, L.scale(L.sub(bhat, b0), 1 - (1 - u) ** 3)); upd(); if (u < 1) requestAnimationFrame(step); }; requestAnimationFrame(step); } }]);
    legend(f.side, [[COL.free, '$X\\boldsymbol\\beta$ for your $\\boldsymbol\\beta$'], [COL.nul, 'residual $\\mathbf y-X\\boldsymbol\\beta$'], [COL.c2, '$\\hat{\\mathbf y}=P_X\\mathbf y$'], [COL.eig, '$-\\nabla L$ (direction)']]);
    const ro = readout(f.side);
    function upd() {
      const p = Xb(beta), r = L.sub(y, p);
      ab.set([0, 0, 0], p); S.setLine(res, [p, y]); lb.p = L.add(p, [0.15, -0.15, -0.2]);
      ro(`<div class="line">${T(`\\boldsymbol\\beta=(${fmt(beta[0], 3)},\\ ${fmt(beta[1], 3)})`)}, ${T(`\\text{RSS}=${fmt(L.dot(r, r), 4)}`)}</div>
        <div class="line">${T(`X^T(\\mathbf y-X\\boldsymbol\\beta)=(${fmt(L.dot(one, r), 3)},\\ ${fmt(L.dot(xv, r), 3)})`)}</div>
        <div class="line">${T(`\\hat{\\boldsymbol\\beta}=(${fmt(bhat[0], 4)},\\ ${fmt(bhat[1], 4)})`)}: ${T(`\\hat y=${fmt(bhat[0], 4)}+${fmt(bhat[1], 4)}x`)}, RSS ${T(fmt(rmin, 4))}</div>`);
      S.draw(); C.draw();
    }
    upd();
    return { reset() { beta = [0.2, 0.9]; upd(); }, state: () => ({ bhat, beta }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
export const HET = (() => { const r = rng(61); return Array.from({ length: 30 }, () => { const x = r() * 10, s = 0.3 + 0.35 * x; return [x, 1 + 0.8 * x + s * gauss(r), s]; }); })();
const wlsViz = {
  title: 'Weighted least squares: trust the quiet observations more',
  mount(f) {
    H(f, 'clamp(300px, 44vw, 380px)');
    let t = 1;
    const X = HET.map(p => [1, p[0]]), y = HET.map(p => p[1]), inv = HET.map(p => 1 / (p[2] * p[2]));
    const wNorm = inv.map(v => v / Math.max(...inv));
    const weights = () => HET.map((_, i) => (1 - t) + t * inv[i]);
    const C = new Chart(f.stage, { x: [0, 10], y: [-4, 18], xlabel: 'x', ylabel: 'y', pad: [14, 14, 26, 34] });
    C.render = () => {
      const bo = wls(X, y, null), bw = wls(X, y, weights());
      C.fn(x => bo[0] + bo[1] * x, 'rgba(232,236,248,.6)', { lw: 1.8, dash: [6, 4] });
      C.fn(x => bw[0] + bw[1] * x, COL.c1, { lw: 2.6 });
      HET.forEach((p, i) => C.dot([p[0], p[1]], 2 + 6 * ((1 - t) * 0.35 + t * Math.sqrt(wNorm[i])), '#ff6b6b', { alpha: 0.85 }));
    };
    const tS = slider(f.side, { label: '$W$: from $I$ to $1/\\sigma_i^2$', min: 0, max: 1, step: 0.01, value: t, fmt: v => v.toFixed(2) }, v => { t = v; upd(); });
    legend(f.side, [['rgba(232,236,248,.7)', 'OLS ($W=I$)'], [COL.c1, 'current weights'], ['#ff6b6b', 'dot size = weight']]);
    const ro = readout(f.side);
    function upd() {
      const bo = wls(X, y, null), bw = wls(X, y, weights());
      ro(`<div class="line">OLS: ${T(`\\hat y=${fmt(bo[0], 3)}+${fmt(bo[1], 3)}x`)}</div><div class="line">${t === 1 ? 'WLS' : 'blend'}: ${T(`\\hat y=${fmt(bw[0], 3)}+${fmt(bw[1], 3)}x`)}</div>
        <div class="line" style="color:var(--muted)">noise spread ${T('\\sigma_i')} grows from ${T(fmt(Math.min(...HET.map(p => p[2])), 2))} to ${T(fmt(Math.max(...HET.map(p => p[2])), 2))} across ${T('x')}</div>`);
      C.draw();
    }
    upd();
    return { reset() { t = 1; tS.set(1); upd(); }, fits: () => ({ ols: wls(X, y, null), wls: wls(X, y, inv) }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
export const RX = [[1, 1, 1], [1, 1, 0.9], [1, 0.9, 1], [1, 0.9, 0.9]], RY = [2.8, 2.5, 2.6, 2.4];
const ridgeViz = {
  title: 'Ridge: coefficient paths as $\\lambda$ grows',
  mount(f) {
    H(f, 'clamp(300px, 44vw, 380px)');
    const [hl, hr] = split(f, [`${T('\\hat\\beta_{\\text{ridge}}')} against ${T('\\log_{10}\\lambda')}`, 'coefficients now']);
    let lg = 0, y = RY.slice(), R = rng(71), nudged = false;
    const lam = () => (lg <= -4 ? 0 : 10 ** lg);
    const cols = ['#ffc857', '#ff6b6b', '#6cb6ff'], names = ['β₀', 'β₁', 'β₂'];
    const A = new Chart(hl, { x: [-4, 2], y: [-1.6, 3], xticks: [[-4, '0'], [-2, '0.01'], [0, '1'], [2, '100']], xlabel: 'λ', pad: [22, 12, 26, 36] });
    const B = new Chart(hr, { x: [-0.5, 2.5], y: [-1.6, 3], xticks: [[0, 'β₀'], [1, 'β₁'], [2, 'β₂']], pad: [22, 12, 26, 36] });
    A.render = () => {
      const pts = Array.from({ length: 121 }, (_, i) => { const l = -4 + 6 * i / 120; return [l, ridge(RX, y, l <= -4 ? 0 : 10 ** l)]; });
      [0, 1, 2].forEach(k => { A.path(pts.map(([l, b]) => [l, b[k]]), cols[k], { lw: 2.2 }); A.label([2, pts[120][1][k]], names[k], cols[k], { dx: -14, dy: k === 1 ? -8 : 8, font: FONT.math }); });
      A.seg([lg, -1.6], [lg, 3], COL.ink, { lw: 1.4, dash: [4, 4] });
      if (nudged) { const b0 = ridge(RX, RY, lam()); [0, 1, 2].forEach(k => A.dot([lg, b0[k]], 3.5, 'rgba(232,236,248,.5)')); }
    };
    B.render = () => {
      const b = ridge(RX, y, lam()), bo = ridge(RX, y, 0);
      [0, 1, 2].forEach(k => { B.rect(k - 0.36, 0, k - 0.02, bo[k], 'rgba(255,107,107,.35)'); B.rect(k + 0.02, 0, k + 0.36, b[k], cols[k]); B.label([k + 0.19, b[k]], fmt(b[k], 3), COL.ink, { dy: b[k] >= 0 ? -9 : 11, align: 'center', font: MONO }); });
      B.seg([-0.5, 0], [2.5, 0], 'rgba(232,236,248,.4)');
    };
    const lS = slider(f.side, { label: '$\\log_{10}\\lambda$', min: -4, max: 2, step: 0.02, value: lg, fmt: v => (v <= -4 ? 'λ=0' : 'λ=' + (+(10 ** v).toPrecision(3))) }, v => { lg = v; upd(); });
    buttons(f.side, [{ label: '$\\lambda=0$ (OLS)', on: () => { lg = -4; lS.set(-4); upd(); } }, { label: '$\\lambda=1$', on: () => { lg = 0; lS.set(0); upd(); } },
      { label: 'nudge $\\mathbf y$', pri: true, on: () => { y = RY.map(v => v + 0.02 * gauss(R)); nudged = true; upd(); } }, { label: 'original $\\mathbf y$', on: () => { y = RY.slice(); nudged = false; upd(); } }]);
    legend(f.side, [['rgba(255,107,107,.5)', 'OLS on the same $\\mathbf y$ (left bar)']]);
    const ro = readout(f.side);
    function upd() {
      const l = lam(), b = ridge(RX, y, l), bo = ridge(RX, y, 0), ev = L.eigSym(XtX(RX)).vals.map(v => v + l);
      ro(`<div class="line">${T(`\\lambda=${+l.toPrecision(3)}`)}: ${T(`\\hat{\\boldsymbol\\beta}_{\\text{ridge}}\\approx(${b.map(v => fmt(v, 3)).join(',\\ ')})`)}</div>
        <div class="line">${T(`\\hat{\\boldsymbol\\beta}_{\\text{OLS}}\\approx(${bo.map(v => fmt(v, 3)).join(',\\ ')})`)}${nudged ? ' (nudged ' + T('\\mathbf y') + ')' : ''}</div>
        <div class="line">eigenvalues of ${T('X^TX+\\lambda I')}: ${T(ev.map(v => fmt(v, 4)).join(',\\ '))}</div>
        <div class="line">${T(`\\det(X^TX)=${fmt(L.eigSym(XtX(RX)).vals.reduce((a, c) => a * c, 1), 5)}`)}</div>`);
      A.draw(); B.draw();
    }
    upd();
    return { reset() { lg = 0; lS.set(0); y = RY.slice(); nudged = false; R = rng(71); upd(); }, at: l => ridge(RX, RY, l) };
  },
};

export default { 'projection': projection, 'wls': wlsViz, 'ridge': ridgeViz };
