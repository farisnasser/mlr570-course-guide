// Part 3.3 Nonlinear Models, Kernels & k-NN — interactive visuals: the circular lift, the polynomial kernel check
// ((1,2),(3,1) -> 36), a real kernel perceptron, RBF/Laplacian similarity, Gram-matrix validity, kernel ridge regression,
// k-NN classification and regression (recomputed in scripts/check/3_3.py).
import { Plane, Chart, COL, FONT } from '../../src/viz/plane.js';
import * as L from '../../src/viz/la.js';
import { el, T, TD, fmt, seg, slider, buttons, readout, legend } from '../../src/viz/ui.js';
import { rng, gauss, mean } from '../../src/viz/stats.js';
import { loadThree, Scene3 } from '../../src/viz/three-lite.js';

const H = (f, h) => { f.stage.style.height = h; };
const MONO = '600 11.5px "JetBrains Mono", monospace';
const RED = '#ff6b6b', BLUE = '#6cb6ff';
function split(f, titles) {
  const sp = el('div', 'split'); f.stage.appendChild(sp);
  return titles.map(t => { const d = el('div'); sp.appendChild(d); if (t) d.appendChild(el('div', 'stage-t', t)); return d; });
}
/** solve (A) x = b, A small dense */
export function solve(A, b) {
  const n = A.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) { let p = c; for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i; [M[c], M[p]] = [M[p], M[c]]; for (let i = c + 1; i < n; i++) { const f = M[i][c] / M[c][c]; for (let j = c; j <= n; j++) M[i][j] -= f * M[c][j]; } }
  const x = Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; } return x;
}
export const KERN = {
  linear: (a, b) => L.dot(a, b),
  poly: (a, b, p = 2) => (1 + L.dot(a, b)) ** p,
  rbf: (a, b, g = 1) => Math.exp(-g * L.dot(L.sub(a, b), L.sub(a, b))),
  laplacian: (a, b, g = 1) => Math.exp(-g * L.norm(L.sub(a, b))),
  sigmoid: (a, b) => Math.tanh(L.dot(a, b) - 1),
};
// the enclosed-class data: inner class near the origin (y = -1), outer ring (y = +1)
export const RING = (() => { const r = rng(81), P = []; for (let i = 0; i < 20; i++) { const a = r() * 2 * Math.PI, d = Math.sqrt(r()) * 0.95; P.push([d * Math.cos(a), d * Math.sin(a), -1]); } for (let i = 0; i < 28; i++) { const a = r() * 2 * Math.PI, d = 1.9 + r() * 0.6; P.push([d * Math.cos(a), d * Math.sin(a), 1]); } return P; })();

// ---------------------------------------------------------------------------------------------------------------
const lift = {
  title: 'The lift $\\phi(\\mathbf{x})=(x_1,x_2,x_1^2+x_2^2)$ makes a flat plane enough',
  async mount(f) {
    H(f, 'clamp(320px, 48vw, 420px)');
    const THREE = await loadThree();
    let c = 2.1;
    const S = new Scene3(f.stage, { cam: [6.2, -6.8, 5.2], target: [0, 0, 1.6], minDist: 4 });
    S.axes(2.8);
    const hz = 0.55;
    RING.forEach(([x, y, t]) => { S.ball(t > 0 ? 0xff6b6b : 0x6cb6ff, 0.07).set([x, y, (x * x + y * y) * hz]); S.ball(t > 0 ? 0xff6b6b : 0x6cb6ff, 0.035).set([x, y, 0]); });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshBasicMaterial({ color: 0x2fb7a6, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }));
    plane.rotation.x = -Math.PI / 2; S.scene.add(plane);
    const circ = S.line([[0, 0, 0]], 0xffc857), circTop = S.line([[0, 0, 0]], 0xffc857, { opacity: 0.6 });
    const lb = S.label(''); lb.e.style.color = COL.eig; lb.e.style.fontStyle = 'normal'; lb.e.style.font = MONO;
    const ax = S.label('x₁² + x₂²'); ax.p = [0, 0, 3.6 * hz + 0.6]; ax.e.style.color = COL.muted;
    const cS = slider(f.side, { label: 'plane height', min: 0.4, max: 5, step: 0.05, value: c, fmt: v => v.toFixed(2) }, v => { c = v; upd(); });
    legend(f.side, [[BLUE, 'inner class'], [RED, 'outer ring'], [COL.eig, 'circle $x_1^2+x_2^2=$ height']]);
    const ro = readout(f.side);
    function upd() {
      plane.position.set(0, c * hz, 0);
      const r = Math.sqrt(c), ring = Array.from({ length: 129 }, (_, i) => [r * Math.cos(i * Math.PI / 64), r * Math.sin(i * Math.PI / 64)]);
      S.setLine(circ, ring.map(([x, y]) => [x, y, 0.005])); S.setLine(circTop, ring.map(([x, y]) => [x, y, c * hz]));
      lb.p = [r, 0, 0.15]; lb.e.textContent = `r = ${fmt(r, 3)}`;
      const wrong = RING.filter(([x, y, t]) => Math.sign(x * x + y * y - c) !== t).length;
      ro(`<div class="line">plane ${T(`x_1^2+x_2^2=${fmt(c, 2)}`)} ↔ circle of radius ${T(`\\sqrt{${fmt(c, 2)}}\\approx${fmt(r, 3)}`)}</div>
        <div class="flag ${wrong ? 'bad' : 'ok'}">${wrong ? `${wrong} point${wrong > 1 ? 's' : ''} on the wrong side` : 'every point on its own side: separated by a flat plane'}</div>`);
      S.draw();
    }
    upd();
    return { reset() { c = 2.1; cS.set(c); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
export const phi2 = ([a, b]) => [1, Math.SQRT2 * a, Math.SQRT2 * b, Math.SQRT2 * a * b, a * a, b * b];
export const binom = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = r * (n - k + i) / i; return Math.round(r); };
const polykernel = {
  title: 'The polynomial kernel is a dot product in 6D, computed in 2D',
  mount(f) {
    H(f, 'clamp(280px, 40vw, 340px)');
    let x = [1, 2], xp = [3, 1], n = 20, d = 2;
    const P = new Plane(f.stage, { extent: 3.4, center: [1.2, 1.2] });
    P.handle({ get: () => x, set: p => { x = p.map(v => Math.round(v * 4) / 4); upd(); }, color: COL.c1 });
    P.handle({ get: () => xp, set: p => { xp = p.map(v => Math.round(v * 4) / 4); upd(); }, color: COL.c2 });
    P.render = () => { P.arrow([0, 0], x, COL.c1); P.arrow([0, 0], xp, COL.c2); P.tipLabel(x, 'x', COL.c1, { gap: 22 }); P.tipLabel(xp, "x′", COL.c2, { gap: 22 }); };
    const ro = readout(f.side);
    f.side.appendChild(el('h5', '', `Feature count ${T('\\binom{n+d}{d}')}`));
    slider(f.side, { label: '$n$', min: 2, max: 100, step: 1, value: n, fmt: v => String(v) }, v => { n = v; upd(); });
    slider(f.side, { label: '$d$', min: 1, max: 5, step: 1, value: d, fmt: v => String(v) }, v => { d = v; upd(); });
    const ro2 = readout(f.side);
    function upd() {
      const a = phi2(x), b = phi2(xp), terms = a.map((v, i) => v * b[i]), K = (1 + L.dot(x, xp)) ** 2;
      ro(`<div class="line">${T(`\\mathbf{x}^T\\mathbf{x}'=${fmt(L.dot(x, xp), 3)},\\ K=(1+${fmt(L.dot(x, xp), 3)})^2=${fmt(K, 4)}`)}</div>
        <div class="line">${T(`\\phi(\\mathbf{x})^T\\phi(\\mathbf{x}')=${terms.map(v => fmt(v, 3)).join('+').replace(/\+-/g, '-')}=${fmt(terms.reduce((s, v) => s + v, 0), 4)}`)}</div>`);
      ro2(`<div class="line">${T(`n=${n},\\ d=${d}`)}: ${T(`\\binom{${n + d}}{${d}}=${binom(n + d, d).toLocaleString('en-US').replace(/,/g, '{,}')}`)} features</div><div class="line">the kernel: one ${T(`${n}`)}-dimensional dot product</div>`);
      P.draw();
    }
    upd();
    return { reset() { x = [1, 2]; xp = [3, 1]; upd(); }, state: () => ({ K: (1 + L.dot(x, xp)) ** 2, dot: L.dot(phi2(x), phi2(xp)) }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
export function perceptronPass(P, alpha, bb, K) {
  let b = bb, updates = 0;
  for (let i = 0; i < P.length; i++) {
    let s = b; for (let j = 0; j < P.length; j++) if (alpha[j]) s += alpha[j] * P[j][2] * K([P[j][0], P[j][1]], [P[i][0], P[i][1]]);
    if (P[i][2] * s <= 0) { alpha[i]++; b += P[i][2]; updates++; }
  }
  return { b, updates };
}
const kperceptron = {
  title: 'The kernel perceptron: $\\alpha_j$ counts the updates',
  mount(f) {
    H(f, 'clamp(320px, 48vw, 420px)');
    const pts = RING.map(([x, y, t]) => [x, y, t]);
    let kind = 'poly', alpha = pts.map(() => 0), b = 0, passes = 0, last = null;
    const xy = p => [p[0], p[1]];   // kernels see the inputs only, never the label
    const K = (u, v) => (kind === 'poly' ? KERN.poly(xy(u), xy(v), 2) : KERN.rbf(xy(u), xy(v), 1));
    const score = q => { let s = b; pts.forEach((p, j) => { if (alpha[j]) s += alpha[j] * p[2] * K(p, q); }); return s; };
    const P = new Plane(f.stage, { extent: 2.9 });
    P.render = () => {
      const g = P.ctx, step = 10;
      if (alpha.some(a => a)) for (let px = 0; px < P.w; px += step) for (let py = 0; py < P.h; py += step) { const q = P.inv(px + step / 2, py + step / 2); g.fillStyle = score(q) > 0 ? 'rgba(255,107,107,.13)' : 'rgba(108,182,255,.13)'; g.fillRect(px, py, step, step); }
      pts.forEach((p, j) => { P.dot(p, alpha[j] ? 6 : 4, p[2] > 0 ? RED : BLUE); if (alpha[j]) P.label(p, String(alpha[j]), COL.ink, { dx: 7, dy: -8, font: '600 10.5px "JetBrains Mono", monospace' }); });
    };
    const kk = seg(f.side, [{ label: 'polynomial $d=2$', value: 'poly' }, { label: 'RBF $\\gamma=1$', value: 'rbf' }], v => { kind = v; clear(); }, kind);
    buttons(f.side, [{ label: 'one pass', pri: true, on: () => run(1) }, { label: '10 passes', on: () => run(10) }, { label: 'clear', on: () => clear() }]);
    const ro = readout(f.side);
    function run(k) { for (let i = 0; i < k; i++) { const r = perceptronPass(pts, alpha, b, K); b = r.b; last = r.updates; passes++; if (!r.updates) break; } upd(); }
    function clear() { alpha = pts.map(() => 0); b = 0; passes = 0; last = null; upd(); }
    function upd() {
      const errs = pts.filter(p => p[2] * score(p) <= 0).length;
      ro(`<div class="line">passes ${passes}${last != null ? `, updates in the last pass: ${last}` : ''}</div>
        <div class="line">${T(`\\sum_j\\alpha_j=${alpha.reduce((s, a) => s + a, 0)}`)} updates so far, ${T(`b=${b}`)}</div>
        <div class="line">${T('\\alpha_j>0')} for ${alpha.filter(a => a).length} of ${pts.length} points</div>
        <div class="flag ${passes && !errs ? 'ok' : passes ? 'eig' : ''}">${passes ? (errs ? `${errs} training points still misclassified` : 'every training point classified correctly') : ''}</div>`);
      P.draw();
    }
    upd();
    return { reset() { kind = 'poly'; kk.set('poly'); clear(); }, run, state: () => ({ alpha, b, passes }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
export function krrFit(X, y, k, lam) { const Kmat = X.map(a => X.map(c => k(a, c))); return solve(Kmat.map((r, i) => r.map((v, j) => v + (i === j ? lam : 0))), y); }
const rbf = {
  title: 'RBF $\\gamma$: from broad hills to very local ones',
  mount(f) {
    H(f, 'clamp(300px, 44vw, 380px)');
    const [hl, hr] = split(f, [`similarity against distance`, `RBF kernel classifier, ${T('\\gamma')} as set`]);
    let lg = 0;
    const g = () => 10 ** lg;
    const A = new Chart(hl, { x: [0, 3], y: [0, 1.05], xlabel: '‖x − x′‖', pad: [22, 12, 26, 32] });
    const P = new Plane(hr, { extent: 2.9 });
    A.render = () => {
      A.fn(r => Math.exp(-g() * r * r), BLUE, { lw: 2.4 }); A.fn(r => Math.exp(-g() * r), RED, { lw: 2, dash: [5, 4] });
      for (const r of [1, 2]) A.dot([r, Math.exp(-g() * r * r)], 4, BLUE);
    };
    let alpha = null;
    P.render = () => {
      const k = (u, v) => KERN.rbf(u, v, g()), step = 10, ctx = P.ctx;
      for (let px = 0; px < P.w; px += step) for (let py = 0; py < P.h; py += step) { const q = P.inv(px + step / 2, py + step / 2); let s = 0; RING.forEach((p, j) => { s += alpha[j] * k([p[0], p[1]], q); }); ctx.fillStyle = s > 0 ? 'rgba(255,107,107,.15)' : 'rgba(108,182,255,.15)'; ctx.fillRect(px, py, step, step); }
      RING.forEach(p => P.dot(p, 4, p[2] > 0 ? RED : BLUE));
    };
    const gS = slider(f.side, { label: '$\\gamma$', min: -2, max: 1.7, step: 0.05, value: lg, fmt: v => (+(10 ** v).toPrecision(2)).toString() }, v => { lg = v; upd(); });
    buttons(f.side, [0.3, 1, 5].map(v => ({ label: `$\\gamma=${v}$`, on: () => { lg = Math.log10(v); gS.set(lg); upd(); } })));
    legend(f.side, [[BLUE, 'RBF $e^{-\\gamma r^2}$'], [RED, 'Laplacian $e^{-\\gamma r}$ (dashed)']]);
    const ro = readout(f.side);
    function upd() {
      alpha = krrFit(RING.map(p => [p[0], p[1]]), RING.map(p => p[2]), (u, v) => KERN.rbf(u, v, g()), 0.1);
      ro(`<div class="line">${T(`\\gamma=${+g().toPrecision(3)}`)}: distance 1 gives ${T(fmt(Math.exp(-g()), 3))}, distance 2 gives ${T(fmt(Math.exp(-4 * g()), 3))}</div>
        <div class="line" style="color:var(--muted)">classifier: sign of the kernel ridge fit to the ${T('\\pm1')} labels (${T('\\lambda=0.1')})</div>`);
      A.draw(); P.draw();
    }
    upd();
    return { reset() { lg = 0; gS.set(0); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
const gram = {
  title: 'A valid kernel gives a Gram matrix with no negative eigenvalues',
  mount(f) {
    H(f, 'clamp(280px, 40vw, 340px)');
    const [hl, hr] = split(f, ['six samples', `${T('K_{ij}=K(\\mathbf{x}_i,\\mathbf{x}_j)')}`]);
    let pts = [[-1.2, 0.4], [-0.3, 1.1], [0.6, 0.8], [1.3, -0.2], [0.2, -1.0], [-0.8, -0.7]], kind = 'rbf';
    const P = new Plane(hl, { extent: 2 });
    pts.forEach((_, i) => P.handle({ get: () => pts[i], set: p => { pts[i] = p; upd(); }, color: COL.free }));
    P.render = () => pts.forEach((p, i) => P.label(p, String(i + 1), COL.ink, { dx: 9, dy: -9, font: MONO }));
    const box = el('div'); box.style.cssText = 'position:absolute;inset:28px 8px 8px;display:grid;place-items:center;overflow:auto'; hr.appendChild(box);
    seg(f.side, [{ label: 'linear', value: 'linear' }, { label: 'poly $d=2$', value: 'poly' }, { label: 'RBF', value: 'rbf' }, { label: 'Laplacian', value: 'laplacian' }, { label: 'sigmoid', value: 'sigmoid' }], v => { kind = v; upd(); }, kind);
    const ro = readout(f.side);
    function upd() {
      const Km = pts.map(a => pts.map(c => KERN[kind](a, c))), ev = L.eigSym(Km).vals, mx = Math.max(...Km.flat().map(Math.abs));
      box.innerHTML = `<table class="mtab">${Km.map(r => `<tr>${r.map(v => `<td style="background:${v >= 0 ? `rgba(124,196,255,${(0.08 + 0.5 * Math.abs(v) / mx).toFixed(3)})` : `rgba(255,122,89,${(0.08 + 0.5 * Math.abs(v) / mx).toFixed(3)})`}">${fmt(v, 2)}</td>`).join('')}</tr>`).join('')}</table>`;
      const minEv = ev[ev.length - 1];
      ro(`<div class="line">eigenvalues ${T(ev.map(v => fmt(v, 3)).join(',\\ '))}</div>
        <div class="flag ${minEv >= -1e-9 ? 'ok' : 'bad'}">${minEv >= -1e-9 ? 'no negative eigenvalue: symmetric PSD here' : `a negative eigenvalue (${fmt(minEv, 3)}): not a valid kernel for these points`}</div>
        ${kind === 'sigmoid' ? `<div class="line" style="color:var(--muted)">${T("\\tanh(\\mathbf{x}^T\\mathbf{x}'-1)")}</div>` : ''}`);
      P.draw();
    }
    upd();
    return { reset() { pts = [[-1.2, 0.4], [-0.3, 1.1], [0.6, 0.8], [1.3, -0.2], [0.2, -1.0], [-0.8, -0.7]]; upd(); }, minEig: k => { const Km = pts.map(a => pts.map(c => KERN[k](a, c))); const ev = L.eigSym(Km).vals; return ev[ev.length - 1]; } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
export const KRR_DATA = (() => { const r = rng(91); return Array.from({ length: 16 }, (_, i) => { const x = 0.2 + 5.6 * (i + r() * 0.6) / 16; return [x, Math.sin(x * 1.1) + 0.12 * gauss(r)]; }); })();
const krr = {
  title: 'Kernel ridge regression: one weight per training sample',
  mount(f) {
    H(f, 'clamp(300px, 42vw, 360px)');
    let lg = 0.3, ll = -2;
    const xs = KRR_DATA.map(p => [p[0]]), ys = KRR_DATA.map(p => p[1]);
    const C = new Chart(f.stage, { x: [0, 6], y: [-1.8, 1.8], xlabel: 'x', ylabel: 'y', pad: [14, 14, 26, 34] });
    C.render = () => {
      const g = 10 ** lg, lam = 10 ** ll, k = (u, v) => KERN.rbf(u, v, g), a = krrFit(xs, ys, k, lam);
      const X = xs.map(([x]) => [1, x]), XtX = [[0, 0], [0, 0]], Xty = [0, 0]; X.forEach((r, i) => { for (let p = 0; p < 2; p++) { Xty[p] += r[p] * ys[i]; for (let q = 0; q < 2; q++) XtX[p][q] += r[p] * r[q]; } });
      const bo = solve(XtX, Xty); C.fn(x => bo[0] + bo[1] * x, BLUE, { lw: 2 });
      a.forEach((ai, i) => C.fn(x => ai * k([x], xs[i]), 'rgba(178,140,255,.35)', { lw: 1, n: 120 }));
      C.fn(x => a.reduce((s, ai, i) => s + ai * k([x], xs[i]), 0), RED, { lw: 2.6, n: 300 });
      KRR_DATA.forEach(p => C.dot(p, 4, COL.ink));
    };
    const gS = slider(f.side, { label: '$\\gamma$', min: -1.5, max: 2, step: 0.05, value: lg, fmt: v => (+(10 ** v).toPrecision(2)).toString() }, v => { lg = v; upd(); });
    const lS = slider(f.side, { label: '$\\lambda$', min: -4, max: 1, step: 0.05, value: ll, fmt: v => (+(10 ** v).toPrecision(2)).toString() }, v => { ll = v; upd(); });
    legend(f.side, [[BLUE, 'OLS line'], [RED, 'RBF kernel regression'], ['rgba(178,140,255,.7)', '$\\alpha_iK(\\mathbf{x}_i,x)$']]);
    const ro = readout(f.side);
    function upd() {
      const g = 10 ** lg, lam = 10 ** ll, k = (u, v) => KERN.rbf(u, v, g), a = krrFit(xs, ys, k, lam);
      const fit = xs.map(q => a.reduce((s, ai, i) => s + ai * k(q, xs[i]), 0)), mse = mean(fit.map((v, i) => (v - ys[i]) ** 2));
      ro(`<div class="line">${T(`\\hat{\\boldsymbol\\alpha}=(K+\\lambda I)^{-1}\\mathbf{y}`)}, ${T(`n=${xs.length}`)} weights, ${T(`K`)} is ${T(`${xs.length}\\times${xs.length}`)}</div>
        <div class="line">largest ${T(`|\\alpha_i|=${fmt(Math.max(...a.map(Math.abs)), 3)}`)}, training MSE ${T(fmt(mse, 4))}</div>`);
      C.draw();
    }
    upd();
    return { reset() { lg = 0.3; ll = -2; gS.set(lg); lS.set(ll); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
export const KNN_DATA = (() => { const r = rng(101), P = []; for (let i = 0; i < 30; i++) P.push([-0.7 + 0.9 * gauss(r), 0.3 + 0.8 * gauss(r), 0]); for (let i = 0; i < 30; i++) P.push([0.8 + 0.9 * gauss(r), -0.2 + 0.8 * gauss(r), 1]); return P; })();
export function knnClassify(P, q, k, weighted) {
  const ds = P.map(p => [Math.hypot(p[0] - q[0], p[1] - q[1]), p[2]]).sort((a, b) => a[0] - b[0]).slice(0, k), v = [0, 0];
  ds.forEach(([d, c]) => { v[c] += weighted ? 1 / (d + 1e-9) : 1; });
  return { cls: v[1] > v[0] ? 1 : v[0] > v[1] ? 0 : ds[0][1], votes: v, ds };
}
const knn = {
  title: 'k-NN: vote among the $k$ nearest stored samples',
  mount(f) {
    H(f, 'clamp(320px, 48vw, 420px)');
    let q = [0.1, 0.2], k = 5, weighted = false;
    const P = new Plane(f.stage, { extent: 3, center: [0.05, 0.05] });
    P.handle({ get: () => q, set: p => { q = p; upd(); }, color: COL.eig });
    P.render = () => {
      const g = P.ctx, step = 9;
      for (let px = 0; px < P.w; px += step) for (let py = 0; py < P.h; py += step) { const c = knnClassify(KNN_DATA, P.inv(px + step / 2, py + step / 2), k, weighted).cls; g.fillStyle = c ? 'rgba(255,107,107,.11)' : 'rgba(108,182,255,.11)'; g.fillRect(px, py, step, step); }
      const r = knnClassify(KNN_DATA, q, k, weighted), R = r.ds[r.ds.length - 1][0];
      P.path(Array.from({ length: 97 }, (_, i) => [q[0] + R * Math.cos(i * Math.PI / 48), q[1] + R * Math.sin(i * Math.PI / 48)]), 'rgba(232,236,248,.6)', { lw: 1.4, dash: [5, 4] });
      KNN_DATA.forEach(p => P.dot(p, 4, p[2] ? RED : BLUE));
      r.ds.forEach(([d]) => { const p = KNN_DATA.find(pp => Math.abs(Math.hypot(pp[0] - q[0], pp[1] - q[1]) - d) < 1e-12); if (p) P.dot(p, 8, p[2] ? RED : BLUE, { ring: true, lw: 1.6 }); });
      P.label(q, '★ x_q', COL.eig, { dx: 8, dy: -10, font: MONO });
    };
    const kS = slider(f.side, { label: '$k$', min: 1, max: 31, step: 1, value: k, fmt: v => String(v) }, v => { k = v; upd(); });
    const w = seg(f.side, [{ label: 'plain vote', value: false }, { label: 'weighted $w_i=1/(d+\\epsilon)$', value: true }], v => { weighted = v; upd(); }, false);
    const ro = readout(f.side);
    function upd() {
      const r = knnClassify(KNN_DATA, q, k, weighted), nr = r.ds.filter(d => d[1] === 1).length;
      ro(`<div class="line">${T(`k=${k}`)} nearest: ${nr} red, ${k - nr} blue</div>
        ${weighted ? `<div class="line">weights: red ${T(fmt(r.votes[1], 2))}, blue ${T(fmt(r.votes[0], 2))}</div>` : ''}
        <div class="flag" style="color:${r.cls ? RED : BLUE}">⇒ predict ${r.cls ? 'red' : 'blue'}</div>
        <div class="line" style="color:var(--muted)">the notes' flip: distances ${T('1,2,4')}, labels ${T('1,0,0')}: plain ${T('0')}, weighted ${T('1>0.5+0.25')}</div>`);
      P.draw();
    }
    upd();
    return { reset() { q = [0.1, 0.2]; k = 5; weighted = false; kS.set(5); w.set(false); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
export const KREG = (() => { const r = rng(111), fx = x => Math.sin(1.3 * x) + 0.25 * x, mk = n => Array.from({ length: n }, () => { const x = r() * 6; return [x, fx(x) + 0.35 * gauss(r)]; }); return { train: mk(60), val: mk(200) }; })();
export const knnReg = (P, x, k) => mean(P.map(p => [Math.abs(p[0] - x), p[1]]).sort((a, b) => a[0] - b[0]).slice(0, k).map(p => p[1]));
const knnreg = {
  title: 'k-NN regression and the minimum-error region',
  mount(f) {
    H(f, 'clamp(300px, 42vw, 360px)');
    const [hl, hr] = split(f, ['fit for this $k$'.replace('$k$', T('k')), `validation MSE against ${T('k')}`]);
    let k = 1;
    const curve = Array.from({ length: 40 }, (_, i) => [i + 1, mean(KREG.val.map(([x, y]) => (knnReg(KREG.train, x, i + 1) - y) ** 2))]);
    const best = curve.reduce((b, c) => (c[1] < b[1] ? c : b));
    const A = new Chart(hl, { x: [0, 6], y: [-2, 3.2], xlabel: 'x', pad: [22, 12, 26, 32] });
    const B = new Chart(hr, { x: [0, 41], y: [0, Math.max(...curve.map(c => c[1])) * 1.1], xlabel: 'k', pad: [22, 12, 26, 40] });
    A.render = () => { KREG.train.forEach(p => A.dot(p, 3, 'rgba(124,196,255,.8)')); A.fn(x => knnReg(KREG.train, x, k), RED, { lw: 2.2, n: 600 }); };
    B.render = () => { B.path(curve, RED, { lw: 2.2 }); B.dot(best, 5, COL.c1); B.label(best, `min at k = ${best[0]}`, COL.c1, { dy: 14, align: 'center', font: MONO }); B.dot(curve[k - 1], 5, COL.ink); };
    const kS = slider(f.side, { label: '$k$', min: 1, max: 40, step: 1, value: k, fmt: v => String(v) }, v => { k = v; upd(); });
    const ro = readout(f.side);
    function upd() {
      ro(`<div class="line">${T(`k=${k}`)}: validation MSE ${T(fmt(curve[k - 1][1], 4))}</div><div class="line">${k <= 3 ? 'small ' + T('k') + ': follows every point (high variance)' : k >= 25 ? 'large ' + T('k') + ': washes the shape out (high bias)' : 'between the two'}</div>
        <div class="line">minimum validation MSE at ${T(`k=${best[0]}`)} for this data</div>`);
      A.draw(); B.draw();
    }
    upd();
    return { reset() { k = 1; kS.set(1); upd(); }, best: () => best[0] };
  },
};

export default { 'lift': lift, 'polykernel': polykernel, 'kperceptron': kperceptron, 'rbf': rbf, 'gram': gram, 'krr': krr, 'knn': knn, 'knnreg': knnreg };
