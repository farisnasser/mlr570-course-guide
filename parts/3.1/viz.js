// Part 3.1 Hard-Margin SVM — interactive visuals: line geometry on x + y - 4 = 0, the toy dual g(alpha) = alpha - alpha^2/4,
// and an exact hard-margin solver on the lecture's ten points (recomputed in scripts/check/3_1.py).
import { Plane, Chart, COL, FONT } from '../../src/viz/plane.js';
import * as L from '../../src/viz/la.js';
import { el, T, TD, fmt, seg, slider, buttons, readout, legend } from '../../src/viz/ui.js';

const H = (f, h) => { f.stage.style.height = h; };
const MONO = '600 11.5px "JetBrains Mono", monospace';
function split(f, titles) {
  const sp = el('div', 'split'); f.stage.appendChild(sp);
  return titles.map(t => { const d = el('div'); sp.appendChild(d); if (t) d.appendChild(el('div', 'stage-t', t)); return d; });
}

// ---------------------------------------------------------------------------------------------------------------
const line = {
  title: 'The normal vector, the signed score and the distance',
  mount(f) {
    H(f, 'clamp(300px, 46vw, 400px)');
    let w = [1, 1], b = -4, k = 1, x0 = [0, 0];
    const P = new Plane(f.stage, { extent: 4.6, center: [1.8, 1.8] });
    P.handle({ get: () => x0, set: p => { x0 = p.map(v => Math.round(v * 4) / 4); upd(); }, color: COL.free });
    P.handle({ get: () => L.add(foot0(), w), set: p => { w = L.sub(p, foot0()).map(v => Math.round(v * 4) / 4); if (L.norm(w) < 0.25) w = [0.25, 0]; upd(); }, color: COL.span });
    const foot0 = () => L.scale(w, -b / L.dot(w, w));          // the point of the line closest to the origin
    P.render = () => {
      const g = P.ctx, W = L.scale(w, k), B = b * k, step = 14;
      for (let px = 0; px < P.w; px += step) for (let py = 0; py < P.h; py += step) {
        const q = P.inv(px + step / 2, py + step / 2), s = L.dot(W, q) + B, a = Math.min(0.32, Math.abs(s) / (6 * k * L.norm(w)) * 0.32);
        g.fillStyle = s > 0 ? `rgba(255,107,107,${a})` : `rgba(108,182,255,${a})`; g.fillRect(px, py, step, step);
      }
      P.lineDir(foot0(), [-w[1], w[0]], COL.ink, { lw: 2.4 });
      P.arrow(foot0(), L.add(foot0(), w), COL.span, { lw: 2.6 }); P.label(L.add(foot0(), w), 'w', COL.span, { dx: 10, font: FONT.math });
      const z = L.sub(x0, L.scale(w, (L.dot(w, x0) + b) / L.dot(w, w)));
      P.seg(x0, z, COL.eig, { lw: 2, dash: [5, 4] }); P.dot(z, 4, COL.ink); P.label(z, 'z', COL.ink, { dx: 8, dy: 10, font: FONT.math });
      P.dot(x0, 5, COL.free); P.label(x0, 'x₀', COL.free, { dx: -18, dy: -8, font: FONT.math });
    };
    const kS = slider(f.side, { label: 'scale $(\\mathbf{w},b)$ by', min: 0.5, max: 10, step: 0.5, value: 1, fmt: v => '×' + v }, v => { k = v; upd(); });
    const bS = slider(f.side, { label: '$b$', min: -8, max: 4, step: 0.5, value: b, fmt: v => v.toFixed(1) }, v => { b = v; upd(); });
    buttons(f.side, [{ label: '$\\mathbf{x}_0=(0,0)$', on: () => { x0 = [0, 0]; upd(); } }, { label: '$\\mathbf{x}_0=(3,3)$', on: () => { x0 = [3, 3]; upd(); } }]);
    legend(f.side, [['rgba(255,107,107,.7)', 'positive side'], ['rgba(108,182,255,.7)', 'negative side']]);
    const ro = readout(f.side);
    function upd() {
      const W = L.scale(w, k), B = b * k, s = L.dot(W, x0) + B, d = Math.abs(s) / L.norm(W);
      const eq = `${fmt(W[0], 2)}x${W[1] >= 0 ? '+' : ''}${fmt(W[1], 2)}y${B >= 0 ? '+' : ''}${fmt(B, 2)}=0`;
      ro(`<div class="line">line ${T(eq)}</div>
        <div class="line">${T(`s(\\mathbf{x}_0)=\\mathbf{w}^T\\mathbf{x}_0+b=${fmt(s, 3)}`)}: ${s > 1e-12 ? 'positive side' : s < -1e-12 ? 'negative side' : 'on the boundary'}</div>
        <div class="line">${T(`\\frac{|\\mathbf{w}^T\\mathbf{x}_0+b|}{\\|\\mathbf{w}\\|}=\\frac{${fmt(Math.abs(s), 3)}}{${fmt(L.norm(W), 3)}}=${fmt(d, 3)}`)}</div>`);
      P.draw();
    }
    upd();
    return { reset() { w = [1, 1]; b = -4; k = 1; x0 = [0, 0]; kS.set(1); bS.set(-4); upd(); }, state: () => { const W = L.scale(w, k), B = b * k; return { score: L.dot(W, x0) + B, dist: Math.abs(L.dot(W, x0) + B) / L.norm(W) }; }, set(o) { if (o.x0) x0 = o.x0; if (o.k) { k = o.k; kS.set(k); } upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
const dual = {
  title: 'The dual is the best lower bound: $g(\\alpha)=\\min_x\\mathcal{L}(x,\\alpha)$',
  mount(f) {
    H(f, 'clamp(280px, 40vw, 340px)');
    const [hl, hr] = split(f, [`${T('\\mathcal{L}(x,\\alpha)=x^2-\\alpha(x-1)')}`, `${T('g(\\alpha)=\\alpha-\\alpha^2/4')}`]);
    let a = 1;
    const g = t => t - t * t / 4;
    const A = new Chart(hl, { x: [-1, 3], y: [-1.5, 4], xlabel: 'x', pad: [22, 12, 26, 32] });
    const B = new Chart(hr, { x: [0, 4.2], y: [-0.5, 1.4], xlabel: 'α', pad: [22, 12, 26, 32] });
    A.render = () => {
      A.rect(-1, -1.5, 1, 4, 'rgba(255,107,107,.06)'); A.label([0, 4], 'infeasible: x < 1', COL.nul, { dy: 10, align: 'center', base: 'top', font: MONO });
      A.fn(x => x * x, 'rgba(232,236,248,.55)', { lw: 1.6, dash: [5, 4] });
      A.fn(x => x * x - a * (x - 1), COL.span, { lw: 2.4 });
      A.dot([1, 1], 4.5, COL.ink); A.label([1, 1], 'p* = 1', COL.ink, { dx: 8, dy: -10, font: MONO });
      A.dot([a / 2, g(a)], 5.5, COL.eig); A.label([a / 2, g(a)], `g(α) = ${fmt(g(a), 3)}`, COL.eig, { dx: 8, dy: 14, font: MONO });
      A.seg([-1, 1], [3, 1], 'rgba(232,236,248,.25)', { dash: [3, 4] });
    };
    B.render = () => {
      B.seg([0, 1], [4.2, 1], COL.ink, { dash: [5, 4] }); B.label([4.2, 1], 'p* = 1', COL.ink, { dx: -4, dy: -10, align: 'right', font: MONO });
      B.fn(g, COL.nul, { lw: 2.4 });
      [0, 1, 2, 3].forEach(t => B.dot([t, g(t)], 3.5, COL.span));
      B.dot([a, g(a)], 6, COL.eig);
    };
    const aS = slider(f.side, { label: '$\\alpha$', min: 0, max: 4, step: 0.05, value: a, fmt: v => v.toFixed(2) }, v => { a = v; upd(); });
    legend(f.side, [['rgba(232,236,248,.6)', 'objective $x^2$'], [COL.span, '$\\mathcal{L}(x,\\alpha)$'], [COL.nul, '$g(\\alpha)$']]);
    const ro = readout(f.side);
    function upd() {
      ro(`<div class="line">${T(`2x-\\alpha=0\\Rightarrow x=\\alpha/2=${fmt(a / 2, 3)}`)}</div>
        <div class="line">${T(`g(${fmt(a, 2)})=${fmt(a, 2)}-\\frac{${fmt(a, 2)}^2}{4}=${fmt(g(a), 4)}\\le p^*=1`)}</div>
        ${Math.abs(a - 2) < 1e-9 ? `<div class="flag ok">${T('\\max_\\alpha g(\\alpha)=g(2)=1=p^*')}</div>` : ''}`);
      A.draw(); B.draw();
    }
    upd();
    return { reset() { a = 1; aS.set(1); upd(); }, g };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// Exact hard-margin SVM in 2D: the optimum is fixed by 2 or 3 support vectors, so try every candidate set, keep the
// feasible solution with the smallest ||w||; then recover alpha from w = sum alpha_i y_i x_i and sum alpha_i y_i = 0.
export const SVM_DATA = [[2, 2, 1], [3, 1.6, 1], [2.6, 2.6, 1], [3, 2.6, 1], [2, 3, 1], [-1, -1, -1], [0, -2, -1], [-2, 0, -1], [-1.4, -1.2, -1], [0.6, -1.6, -1]];
function solve3(M, r) {   // 3x3 linear solve (Cramer), null when singular
  const d = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = d(M); if (Math.abs(D) < 1e-10) return null;
  return [0, 1, 2].map(c => d(M.map((row, i) => row.map((v, j) => (j === c ? r[i] : v)))) / D);
}
export function hardMargin(P) {
  const n = P.length, feasible = (w, b) => P.every(([x, y, t]) => t * (w[0] * x + w[1] * y + b) >= 1 - 1e-9);
  let best = null;
  const consider = (w, b, sv) => { if (w && isFinite(b) && feasible(w, b)) { const nw = L.norm(w); if (!best || nw < best.nw - 1e-12) best = { w, b, nw, sv }; } };
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (P[i][2] !== 1 || P[j][2] !== -1) continue;
    const d = [P[i][0] - P[j][0], P[i][1] - P[j][1]], dd = L.dot(d, d); if (dd < 1e-12) continue;
    const w = L.scale(d, 2 / dd), b = 1 - L.dot(w, [P[i][0], P[i][1]]); consider(w, b, [i, j]);
  }
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
    const S = [i, j, k]; if (S.every(s => P[s][2] === 1) || S.every(s => P[s][2] === -1)) continue;
    const sol = solve3(S.map(s => [P[s][0], P[s][1], 1]), S.map(s => P[s][2])); if (sol) consider([sol[0], sol[1]], sol[2], S);
  }
  if (!best) return null;
  const sv = P.map(([x, y, t], i) => i).filter(i => Math.abs(P[i][2] * (best.w[0] * P[i][0] + best.w[1] * P[i][1] + best.b) - 1) < 1e-7);
  // alpha on the support vectors: least squares of [x_i y_i ; y_i] alpha = [w ; 0] (exact here)
  const A = [[...sv.map(i => P[i][2] * P[i][0])], [...sv.map(i => P[i][2] * P[i][1])], [...sv.map(i => P[i][2])]], rhs = [best.w[0], best.w[1], 0];
  let alpha = null;
  if (sv.length === 3) alpha = solve3(A, rhs);
  else if (sv.length === 2) { const t = L.dot(best.w, best.w) / 2; alpha = [t, t]; }   // two SVs: alpha_+ = alpha_- = ||w||^2 / 2
  return { w: best.w, b: best.b, margin: 2 / best.nw, sv, alpha };
}
const svm = {
  title: 'The maximum-margin separator on the lecture\u2019s ten points',
  mount(f) {
    H(f, 'clamp(340px, 52vw, 460px)');
    let P = SVM_DATA.map(p => p.slice()), showAny = false;
    const C = new Plane(f.stage, { extent: 3.2, center: [0.5, 0.5] });
    P.forEach((_, i) => C.handle({ get: () => P[i], set: q => { P[i][0] = Math.round(q[0] * 10) / 10; P[i][1] = Math.round(q[1] * 10) / 10; upd(); }, color: P[i][2] > 0 ? '#ff6b6b' : '#6cb6ff', ring: false }));
    C.render = () => {
      const s = hardMargin(P);
      if (showAny) C.lineDir([0.25, 0.25], [1, -1], 'rgba(232,236,248,.45)', { lw: 1.5, dash: [3, 5] });
      if (s) {
        const { w, b } = s, n2 = L.dot(w, w), dir = [-w[1], w[0]];
        const on = c => L.scale(w, (c - b) / n2);
        const g = C.ctx; g.save(); g.beginPath();
        const Lb = C.o.extent * 6 / L.norm(dir), pts = [L.add(on(1), L.scale(dir, -Lb)), L.add(on(1), L.scale(dir, Lb)), L.add(on(-1), L.scale(dir, Lb)), L.add(on(-1), L.scale(dir, -Lb))];
        pts.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, ...C.P(p))); g.closePath(); g.fillStyle = 'rgba(47,183,166,.12)'; g.fill(); g.restore();
        C.lineDir(on(1), dir, '#ff6b6b', { lw: 1.5, dash: [6, 5] }); C.lineDir(on(-1), dir, '#6cb6ff', { lw: 1.5, dash: [6, 5] });
        C.lineDir(on(0), dir, COL.ink, { lw: 2.4 });
        C.arrow(on(0), L.add(on(0), L.scale(w, 0.9 / L.norm(w))), COL.eig, { lw: 2.2, head: 9 }); C.label(L.add(on(0), L.scale(w, 0.9 / L.norm(w))), 'w', COL.eig, { dx: 8, font: FONT.math });
        s.sv.forEach(i => C.dot(P[i], 11, P[i][2] > 0 ? '#ff6b6b' : '#6cb6ff', { ring: true, lw: 2 }));
      }
      P.forEach(p => C.dot(p, 5, p[2] > 0 ? '#ff6b6b' : '#6cb6ff'));
    };
    buttons(f.side, [{ label: 'show $x_1+x_2=0.5$', on: () => { showAny = !showAny; C.draw(); } }]);
    legend(f.side, [['#ff6b6b', '$y=+1$'], ['#6cb6ff', '$y=-1$'], ['rgba(47,183,166,.6)', 'margin corridor'], [COL.ink, 'circled: support vectors']]);
    const ro = readout(f.side);
    function upd() {
      const s = hardMargin(P);
      if (!s) { ro('<div class="flag bad">not linearly separable: no hard-margin solution</div>'); C.draw(); return; }
      const svTxt = s.sv.map(i => `(${fmt(P[i][0], 1)},${fmt(P[i][1], 1)})`).join(',\\ ');
      ro(`<div class="line">${T(`\\mathbf{w}^*\\approx(${fmt(s.w[0], 3)},\\ ${fmt(s.w[1], 3)}),\\ b^*\\approx${fmt(s.b, 3)}`)}</div>
        <div class="line">${T(`\\|\\mathbf{w}^*\\|\\approx${fmt(L.norm(s.w), 3)},\\ \\frac{2}{\\|\\mathbf{w}^*\\|}\\approx${fmt(s.margin, 2)}`)}</div>
        <div class="line">support vectors ${T(svTxt)}</div>
        ${s.alpha ? `<div class="line">${T(s.alpha.map((a, k) => `\\alpha_{${s.sv[k] + 1}}=${fmt(a, 4)}`).join(',\\ '))}; every other ${T('\\alpha_i=0')}</div>` : ''}
        <div class="line">${T(`\\min_i y_i(\\mathbf{w}^{*T}\\mathbf{x}_i+b^*)=${fmt(Math.min(...P.map(([x, y, t]) => t * (s.w[0] * x + s.w[1] * y + s.b))), 4)}`)}</div>`);
      C.draw();
    }
    upd();
    return { reset() { P = SVM_DATA.map(p => p.slice()); showAny = false; upd(); }, state: () => hardMargin(P), move(i, x, y) { P[i][0] = x; P[i][1] = y; upd(); } };
  },
};

export default { 'line': line, 'dual': dual, 'svm': svm };
