// Part 2.3 Model Evaluation — interactive visuals: real least-squares fits for under/overfitting and K-fold CV, a
// threshold sweeping through scored samples, P/R means on the lecture rows, MSE/MAE on the lecture examples.
import { Chart, COL, FONT } from '../../src/viz/plane.js';
import { el, T, TD, fmt, seg, slider, buttons, readout, legend } from '../../src/viz/ui.js';
import { rng, gauss, mean } from '../../src/viz/stats.js';

const H = (f, h) => { f.stage.style.height = h; };
const MONO = '600 11.5px "JetBrains Mono", monospace';
function split(f, titles) {
  const sp = el('div', 'split'); f.stage.appendChild(sp);
  return titles.map(t => { const d = el('div'); sp.appendChild(d); if (t) d.appendChild(el('div', 'stage-t', t)); return d; });
}

/** least squares by Householder QR: minimize ||A b - y||; A is m x p (m >= p) */
export function lstsq(A, y) {
  const m = A.length, p = A[0].length, R = A.map(r => r.slice()), q = y.slice();
  for (let k = 0; k < p; k++) {
    let nrm = 0; for (let i = k; i < m; i++) nrm += R[i][k] ** 2; nrm = Math.sqrt(nrm); if (nrm < 1e-14) continue;
    const alpha = R[k][k] > 0 ? -nrm : nrm, v = Array(m).fill(0); v[k] = R[k][k] - alpha; for (let i = k + 1; i < m; i++) v[i] = R[i][k];
    const vv = v.reduce((s, x) => s + x * x, 0); if (vv < 1e-300) continue;
    for (let j = k; j < p; j++) { let s = 0; for (let i = k; i < m; i++) s += v[i] * R[i][j]; s = 2 * s / vv; for (let i = k; i < m; i++) R[i][j] -= s * v[i]; }
    let s = 0; for (let i = k; i < m; i++) s += v[i] * q[i]; s = 2 * s / vv; for (let i = k; i < m; i++) q[i] -= s * v[i];
  }
  const b = Array(p).fill(0);
  for (let k = p - 1; k >= 0; k--) { let s = q[k]; for (let j = k + 1; j < p; j++) s -= R[k][j] * b[j]; b[k] = Math.abs(R[k][k]) < 1e-14 ? 0 : s / R[k][k]; }
  return b;
}
export const polyFit = (xs, ys, deg) => lstsq(xs.map(x => Array.from({ length: deg + 1 }, (_, j) => x ** j)), ys);
export const polyVal = (b, x) => b.reduce((s, c, j) => s + c * x ** j, 0);
export const mse = (b, xs, ys) => mean(xs.map((x, i) => (polyVal(b, x) - ys[i]) ** 2));
const truth = x => Math.sin(2.2 * x) + 0.3 * x;
export const DATA = (() => {
  const r = rng(31), mk = n => Array.from({ length: n }, () => { const x = r() * 2 - 1; return [x, truth(x) + 0.25 * gauss(r)]; });
  return { train: mk(14), val: mk(40) };
})();

// ---------------------------------------------------------------------------------------------------------------
const complexity = {
  title: 'Underfitting and overfitting with real polynomial fits',
  mount(f) {
    H(f, 'clamp(300px, 44vw, 380px)');
    const [hl, hr] = split(f, ['fit on the training points', `error against degree (model complexity)`]);
    let deg = 1;
    const tx = DATA.train.map(p => p[0]), ty = DATA.train.map(p => p[1]), vx = DATA.val.map(p => p[0]), vy = DATA.val.map(p => p[1]);
    const curves = Array.from({ length: 13 }, (_, d) => { const b = polyFit(tx, ty, d); return [mse(b, tx, ty), mse(b, vx, vy)]; });
    const A = new Chart(hl, { x: [-1.05, 1.05], y: [-2, 2], xlabel: 'x', ylabel: 'y', pad: [22, 12, 26, 32] });
    const B = new Chart(hr, { x: [0, 12], y: [-3, 1], yticks: [-3, -2, -1, 0, 1].map(v => [v, `10^${v}`]), xlabel: 'degree', ylabel: 'MSE (log)', pad: [22, 12, 26, 44] });
    A.render = () => {
      const b = polyFit(tx, ty, deg);
      DATA.val.forEach(p => A.dot(p, 3, 'rgba(255,200,87,.45)'));
      A.fn(x => Math.max(-3, Math.min(3, polyVal(b, x))), COL.nul, { lw: 2.4, n: 300 });
      DATA.train.forEach(p => A.dot(p, 4.5, COL.span));
    };
    B.render = () => {
      const lg = v => Math.min(1, Math.log10(Math.max(v, 1e-3)));   // log scale, clipped at the top of the chart
      B.path(curves.map((c, d) => [d, lg(c[0])]), COL.span, { lw: 2.2 }); B.path(curves.map((c, d) => [d, lg(c[1])]), COL.nul, { lw: 2.2 });
      const best = curves.reduce((bi, c, d) => (c[1] < curves[bi][1] ? d : bi), 0);
      B.seg([best, -3], [best, 1], 'rgba(232,236,248,.4)', { dash: [4, 4] }); B.label([best, 1], 'optimum', COL.ink, { dy: 8, base: 'top', align: 'center', font: MONO });
      B.seg([deg, -3], [deg, 1], COL.eig, { lw: 1.5 }); B.dot([deg, lg(curves[deg][0])], 4.5, COL.span); B.dot([deg, lg(curves[deg][1])], 4.5, COL.nul);
    };
    const dS = slider(f.side, { label: 'degree', min: 0, max: 12, step: 1, value: deg, fmt: v => String(v) }, v => { deg = v; upd(); });
    legend(f.side, [[COL.span, 'training points / training error'], [COL.eig, 'validation points'], [COL.nul, 'fit / validation error']]);
    const ro = readout(f.side);
    function upd() {
      const [tr, va] = curves[deg], best = curves.reduce((bi, c, d) => (c[1] < curves[bi][1] ? d : bi), 0);
      const verdict = deg < best ? ['underfitting: both errors high', 'bad'] : deg === best ? ['minimum validation error', 'ok'] : va > 2 * curves[best][1] ? ['overfitting: training error low, validation error much higher', 'bad'] : ['past the optimum: the gap is opening', 'eig'];
      ro(`<div class="line">training MSE ${T(fmt(tr, 4))}, validation MSE ${T(fmt(va, 4))}</div><div class="flag ${verdict[1]}">${verdict[0]}</div>`);
      A.draw(); B.draw();
    }
    upd();
    return { reset() { deg = 1; dS.set(1); upd(); }, curves };
  },
};

// ---------------------------------------------------------------------------------------------------------------
const kfold = {
  title: 'K-fold cross-validation, run by run',
  mount(f) {
    H(f, 'clamp(300px, 42vw, 370px)');
    const [hl, hr] = split(f, ['', 'held-out error per run']);
    const r = rng(41), pts = Array.from({ length: 30 }, () => { const x = r() * 2 - 1; return [x, truth(x) + 0.25 * gauss(r)]; });
    let K = 5, run = 0;
    const folds = () => pts.map((_, i) => i % K);
    const fitRun = k => { const fo = folds(), tr = pts.filter((_, i) => fo[i] !== k), te = pts.filter((_, i) => fo[i] === k); const b = polyFit(tr.map(p => p[0]), tr.map(p => p[1]), 3); return { b, err: mse(b, te.map(p => p[0]), te.map(p => p[1])) }; };
    const A = new Chart(hl, { x: [-1.05, 1.05], y: [-2, 2], xlabel: 'x', pad: [22, 12, 26, 32] });
    const B = new Chart(hr, { x: [0.4, 10.6], y: [0, 0.2], xlabel: 'run', pad: [22, 12, 26, 44] });
    A.render = () => {
      const fo = folds(), { b } = fitRun(run);
      A.fn(x => polyVal(b, x), COL.span, { lw: 2.2 });
      pts.forEach((p, i) => A.dot(p, fo[i] === run ? 5.5 : 4, fo[i] === run ? COL.nul : 'rgba(124,196,255,.75)'));
    };
    B.render = () => {
      const errs = Array.from({ length: K }, (_, k) => fitRun(k).err), cv = mean(errs);
      B.o.x = [0.4, K + 0.6]; B.o.xticks = Array.from({ length: K }, (_, k) => k + 1); B.o.y = [0, Math.max(...errs) * 1.3];
      errs.forEach((e, k) => B.rect(k + 0.65, 0, k + 1.35, e, k === run ? 'rgba(255,107,107,.85)' : 'rgba(255,107,107,.35)'));
      B.seg([0.4, cv], [K + 0.6, cv], COL.eig, { dash: [5, 4] }); B.label([K + 0.6, cv], 'CV error', COL.eig, { dx: -4, dy: -9, align: 'right', font: MONO });
    };
    const kS = slider(f.side, { label: '$K$', min: 2, max: 10, step: 1, value: K, fmt: v => String(v) }, v => { K = v; run = Math.min(run, K - 1); upd(); });
    buttons(f.side, [{ label: '◀', on: () => { run = (run + K - 1) % K; upd(); } }, { label: 'next run ▶', pri: true, on: () => { run = (run + 1) % K; upd(); } }]);
    legend(f.side, [[COL.nul, 'VAL fold (held out)'], ['rgba(124,196,255,.8)', 'Train folds'], [COL.span, 'cubic fit on Train']]);
    const ro = readout(f.side);
    function upd() {
      const errs = Array.from({ length: K }, (_, k) => fitRun(k).err);
      ro(`<div class="line">run ${run + 1} of ${K}: ${T(`\\text{Error}_{${run + 1}}=${fmt(errs[run], 4)}`)}</div>
        <div class="line">${T(`\\text{CV error}=\\frac1{${K}}\\sum_{i=1}^{${K}}\\text{Error}_i=${fmt(mean(errs), 4)}`)}</div>
        <div class="line">${K} fits, each validated on ${T(String(Math.round(pts.length / K)))} of 30 points</div>`);
      A.draw(); B.draw();
    }
    upd();
    return { reset() { K = 5; run = 0; kS.set(5); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
export const prf = (TP, FN, FP, TN, beta = 1) => { const P = TP / (TP + FP), R = TP / (TP + FN), b2 = beta * beta; return { P, R, F1: 2 * P * R / (P + R), Fb: (1 + b2) * P * R / (b2 * P + R), acc: (TP + TN) / (TP + FN + FP + TN) }; };
const means = {
  title: 'Harmonic $\\le$ geometric $\\le$ arithmetic mean',
  mount(f) {
    H(f, 'clamp(260px, 36vw, 320px)');
    let P = 0.1, R = 0.9;
    const C = new Chart(f.stage, { x: [0, 1], y: [0, 1], xlabel: 'Recall', ylabel: 'mean', pad: [16, 14, 26, 36] });
    C.handle({ get: () => [R, 0], set: p => { R = Math.max(0.01, Math.round(p[0] * 100) / 100); rS.set(R); upd(); }, color: COL.ink });
    C.render = () => {
      C.fn(r => (P + r) / 2, COL.span, { lw: 2, dom: [0.001, 1] }); C.fn(r => Math.sqrt(P * r), COL.c1, { lw: 2, dom: [0.001, 1] }); C.fn(r => 2 * P * r / (P + r), COL.nul, { lw: 2.4, dom: [0.001, 1] });
      for (const [v, c] of [[(P + R) / 2, COL.span], [Math.sqrt(P * R), COL.c1], [2 * P * R / (P + R), COL.nul]]) C.dot([R, v], 4.5, c);
      C.seg([R, 0], [R, 1], 'rgba(232,236,248,.3)', { dash: [4, 4] });
    };
    const pS = slider(f.side, { label: 'Precision', min: 0.01, max: 1, step: 0.01, value: P, fmt: v => v.toFixed(2) }, v => { P = v; upd(); });
    const rS = slider(f.side, { label: 'Recall', min: 0.01, max: 1, step: 0.01, value: R, fmt: v => v.toFixed(2) }, v => { R = v; upd(); });
    const rows = [['Balanced', 50, 10, 10, 50], ['High P, low R', 10, 40, 2, 48], ['High R, low P', 45, 5, 40, 10]];
    buttons(f.side, rows.map(([n, TP, FN, FP, TN]) => ({ label: n, on: () => { const s = prf(TP, FN, FP, TN); P = s.P; R = s.R; pS.set(+P.toFixed(2)); rS.set(+R.toFixed(2)); upd(); } })));
    legend(f.side, [[COL.span, 'arithmetic'], [COL.c1, 'geometric'], [COL.nul, 'harmonic = F1']]);
    const ro = readout(f.side);
    function upd() {
      const am = (P + R) / 2, gm = Math.sqrt(P * R), hm = 2 * P * R / (P + R);
      ro(`<div class="line">${T(`F_1=\\frac{2(${fmt(P, 3)})(${fmt(R, 3)})}{${fmt(P, 3)}+${fmt(R, 3)}}=${fmt(hm, 3)}`)}</div><div class="line">${T(`${fmt(hm, 3)}\\le${fmt(gm, 3)}\\le${fmt(am, 3)}`)}</div>`);
      C.draw();
    }
    upd();
    return { reset() { P = 0.1; R = 0.9; pS.set(P); rS.set(R); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
export const SCORES = (() => { const r = rng(51), cl = v => Math.max(0.005, Math.min(0.995, v)); return { pos: Array.from({ length: 100 }, () => cl(0.64 + 0.13 * gauss(r))), neg: Array.from({ length: 100 }, () => cl(0.36 + 0.13 * gauss(r))) }; })();
export const confusionAt = t => ({ TP: SCORES.pos.filter(s => s >= t).length, FN: SCORES.pos.filter(s => s < t).length, FP: SCORES.neg.filter(s => s >= t).length, TN: SCORES.neg.filter(s => s < t).length });
const threshold = {
  title: 'Slide the decision threshold: same model, new confusion matrix',
  mount(f) {
    H(f, 'clamp(300px, 44vw, 380px)');
    const [hl, hr] = split(f, ['scores of actual − and actual +', `precision–recall as ${T('t')} moves`]);
    let t = 0.5, beta = 1;
    const A = new Chart(hl, { x: [0, 1], y: [0, 2.2], yticks: [[0.6, 'actual −'], [1.6, 'actual +']], xlabel: 'score', pad: [22, 12, 26, 60], grid: false });
    const B = new Chart(hr, { x: [0, 1], y: [0, 1.02], xlabel: 'Recall', ylabel: 'Precision', pad: [22, 12, 26, 36] });
    A.handle({ get: () => [t, 2.1], set: p => { t = Math.round(p[0] * 100) / 100; tS.set(t); upd(); }, color: COL.eig });
    const jit = (s, k) => ((Math.sin(k * 12.9898 + s * 78.233) * 43758.5453) % 1 + 1) % 1;
    A.render = () => {
      A.rect(t, 0, 1, 2.2, 'rgba(255,200,87,.06)');
      SCORES.neg.forEach((s, k) => A.dot([s, 0.25 + 0.7 * jit(s, k)], 3.2, s >= t ? '#6cb6ff' : 'rgba(108,182,255,.35)'));
      SCORES.pos.forEach((s, k) => A.dot([s, 1.25 + 0.7 * jit(s, k + 7)], 3.2, s >= t ? '#ff6b6b' : 'rgba(255,107,107,.35)'));
      A.seg([t, 0], [t, 2.2], COL.eig, { lw: 2, dash: [5, 4] }); A.label([t, 2.2], 'threshold t', COL.eig, { dy: 10, dx: 6, base: 'top', font: MONO });
    };
    const curve = Array.from({ length: 101 }, (_, i) => { const c = confusionAt(i / 100); return c.TP + c.FP ? [c.TP / (c.TP + c.FN), c.TP / (c.TP + c.FP)] : null; }).filter(Boolean);
    B.render = () => { B.path(curve, COL.free, { lw: 2 }); const c = confusionAt(t); if (c.TP + c.FP) B.dot([c.TP / (c.TP + c.FN), c.TP / (c.TP + c.FP)], 6, COL.eig); };
    const tS = slider(f.side, { label: '$t$', min: 0, max: 1, step: 0.01, value: t, fmt: v => v.toFixed(2) }, v => { t = v; upd(); });
    const bS = slider(f.side, { label: '$\\beta$', min: 0.25, max: 4, step: 0.25, value: beta, fmt: v => v.toFixed(2) }, v => { beta = v; upd(); });
    const cm = el('div'); f.side.appendChild(cm);
    const ro = readout(f.side);
    function upd() {
      const c = confusionAt(t), s = prf(c.TP, c.FN, c.FP, c.TN, beta), cs = 'padding:5px 8px;border:1px solid var(--line);text-align:center;font:600 12.5px var(--mono)';
      cm.innerHTML = `<table style="border-collapse:collapse;margin:0 auto;color:var(--ink)"><tr><td></td><th style="${cs};color:var(--muted)">Pred +</th><th style="${cs};color:var(--muted)">Pred −</th></tr><tr><th style="${cs};color:var(--muted)">Actual +</th><td style="${cs};background:rgba(47,183,166,.25)">TP ${c.TP}</td><td style="${cs};background:rgba(255,107,107,.18)">FN ${c.FN}</td></tr><tr><th style="${cs};color:var(--muted)">Actual −</th><td style="${cs};background:rgba(255,107,107,.18)">FP ${c.FP}</td><td style="${cs};background:rgba(47,183,166,.25)">TN ${c.TN}</td></tr></table>`;
      const f3 = v => (isFinite(v) ? fmt(v, 3) : '\\text{undefined}');
      ro(`<div class="line">accuracy ${T(f3(s.acc))}, precision ${T(f3(s.P))}, recall ${T(f3(s.R))}</div><div class="line">${T(`F_1=${f3(s.F1)},\\ F_{${fmt(beta, 2)}}=${f3(s.Fb)}`)}</div>`);
      A.draw(); B.draw();
    }
    upd();
    return { reset() { t = 0.5; beta = 1; tS.set(t); bS.set(beta); upd(); }, at: confusionAt };
  },
};

// ---------------------------------------------------------------------------------------------------------------
const mseMae = {
  title: 'MSE vs MAE: one outlier, and the best constant',
  mount(f) {
    H(f, 'clamp(300px, 42vw, 360px)');
    const [hl, hr] = split(f, [`targets ${T('y_i')} and predictions ${T('\\hat y_i')}`, `one constant ${T('c')} for every point`]);
    const y0 = [3, 0.5, 2, 7, 100], yh = [2.5, 0, 2, 8, 8];
    let y = y0.slice(), n = 5, c = 5;
    const lo = () => Math.min(0, ...y.slice(0, n)) - 2, hi = () => Math.max(10, ...y.slice(0, n)) * 1.08;
    const A = new Chart(hl, { x: [0.4, 5.6], y: [-2, 110], xticks: [1, 2, 3, 4, 5], xlabel: 'i', pad: [22, 12, 26, 40] });
    const B = new Chart(hr, { x: [-2, 110], y: [0, 1], xlabel: 'c', pad: [22, 12, 26, 44] });
    A.handle({ get: () => [5, y[4]], set: p => { y[4] = Math.round(p[1] * 2) / 2; upd(); }, color: COL.nul, visible: () => n === 5 });
    A.render = () => {
      for (let i = 0; i < n; i++) { A.seg([i + 1, yh[i]], [i + 1, y[i]], 'rgba(255,107,107,.6)', { lw: 2 }); A.dot([i + 1, yh[i]], 4.5, COL.span); A.dot([i + 1, y[i]], 4.5, COL.nul); }
    };
    const fM = (cc, ys) => mean(ys.map(v => (v - cc) ** 2)), fA = (cc, ys) => mean(ys.map(v => Math.abs(v - cc)));
    B.render = () => {
      const ys = y.slice(0, n), maxM = Math.max(...[B.o.x[0], B.o.x[1]].map(cc => fM(cc, ys))), maxA = Math.max(...[B.o.x[0], B.o.x[1]].map(cc => fA(cc, ys)));
      B.o.y = [0, 1.05]; B.o.yticks = [[0, '0'], [1, 'max']];
      B.fn(cc => fM(cc, ys) / maxM, COL.nul, { lw: 2.2 }); B.fn(cc => fA(cc, ys) / maxA, COL.c1, { lw: 2.2 });
      const m = mean(ys), md = [...ys].sort((a, b) => a - b)[Math.floor(n / 2)];
      B.dot([m, fM(m, ys) / maxM], 5, COL.nul); B.label([m, fM(m, ys) / maxM], `mean ${fmt(m, 2)}`, COL.nul, { dy: -12, align: 'center', font: MONO });
      B.dot([md, fA(md, ys) / maxA], 5, COL.c1); B.label([md, fA(md, ys) / maxA], `median ${fmt(md, 2)}`, COL.c1, { dy: 14, align: 'center', font: MONO });
    };
    const pick = seg(f.side, [{ label: 'example 1 (4 points)', value: 4 }, { label: 'example 2 (with 100)', value: 5 }], v => { n = v; y = y0.slice(); reframe(); upd(); }, 5);
    legend(f.side, [[COL.nul, '$y_i$ / MSE (scaled)'], [COL.span, '$\\hat y_i$'], [COL.c1, 'MAE (scaled)']]);
    const ro = readout(f.side);
    function reframe() { A.o.x = [0.4, n + 0.6]; A.o.xticks = Array.from({ length: n }, (_, i) => i + 1); A.o.y = [lo(), hi()]; B.o.x = [lo(), hi()]; }
    function upd() {
      if (n === 5) reframe();
      const e = y.slice(0, n).map((v, i) => v - yh[i]), M = mean(e.map(v => v * v)), Ab = mean(e.map(Math.abs));
      ro(`<div class="line">errors ${T(`[${e.map(v => fmt(v, 2)).join(',\\ ')}]`)}</div>
        <div class="line">${T(`\\text{MSE}=${fmt(M, 4)}`)}, ${T(`\\text{RMSE}=${fmt(Math.sqrt(M), 3)}`)}</div><div class="line">${T(`\\text{MAE}=${fmt(Ab, 4)}`)}</div>`);
      A.draw(); B.draw();
    }
    reframe(); upd();
    return { reset() { n = 5; y = y0.slice(); pick.set(5); reframe(); upd(); }, state: () => { const e = y.slice(0, n).map((v, i) => v - yh[i]); return { mse: mean(e.map(v => v * v)), mae: mean(e.map(Math.abs)) }; }, setN(v) { n = v; pick.set(v); y = y0.slice(); reframe(); upd(); } };
  },
};

export default { 'complexity': complexity, 'kfold': kfold, 'means': means, 'threshold': threshold, 'mse-mae': mseMae };
