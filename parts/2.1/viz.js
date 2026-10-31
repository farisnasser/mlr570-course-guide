// Part 2.1 Data Preprocessing — interactive visuals on the notes' own examples ({2,3,3,4,100}, ages {20,30,40,60} + 100,
// 5% positives, the 14-value binning sample with counts 9,3,1,1 / 4,4,3,3, PCA on a 2D cloud). Checked in scripts/check/2_1.py.
import { Plane, Chart, COL, FONT } from '../../src/viz/plane.js';
import * as L from '../../src/viz/la.js';
import { el, T, TD, fmt, matTex, vecTex, seg, slider, buttons, readout, legend } from '../../src/viz/ui.js';
import { rng, gauss, mean, varPop, varSample } from '../../src/viz/stats.js';
import { loadThree, Scene3 } from '../../src/viz/three-lite.js';

const H = (f, h) => { f.stage.style.height = h; };
const MONO = '600 11.5px "JetBrains Mono", monospace';
function split(f, titles) {
  const sp = el('div', 'split'); f.stage.appendChild(sp);
  return titles.map(t => { const d = el('div'); sp.appendChild(d); if (t) d.appendChild(el('div', 'stage-t', t)); return d; });
}
const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };

// ---------------------------------------------------------------------------------------------------------------
// 2.1.1 Scale distorts geometry: nearest neighbour under raw features vs min-max vs z-score
export const SCALE_DATA = (() => { const r = rng(11); return Array.from({ length: 16 }, () => [5000 + r() * 10000, 1 + r() * 18]); })();
const scale = {
  title: 'Scale can distort geometry: who is the nearest neighbour?',
  mount(f) {
    H(f, 'clamp(300px, 44vw, 390px)');
    const D = SCALE_DATA;
    let q = [6000, 2], mode = 'raw';
    const lo = [0, 1].map(j => Math.min(...D.map(p => p[j]))), hi = [0, 1].map(j => Math.max(...D.map(p => p[j])));
    const mu = [0, 1].map(j => mean(D.map(p => p[j]))), sd = [0, 1].map(j => Math.sqrt(varPop(D.map(p => p[j]))));
    const tr = p => mode === 'raw' ? p : mode === 'mm' ? p.map((v, j) => (v - lo[j]) / (hi[j] - lo[j])) : p.map((v, j) => (v - mu[j]) / sd[j]);
    const nearest = () => { let b = -1, bd = Infinity; D.forEach((p, i) => { const d = L.norm(L.sub(tr(p), tr(q))); if (d < bd) { bd = d; b = i; } }); return [b, bd]; };
    const C = new Chart(f.stage, { x: [4000, 16000], y: [0, 20], xlabel: 'Feature 1', ylabel: 'Feature 2', pad: [14, 14, 26, 40] });
    C.handle({ get: () => q, set: p => { q = p; upd(); }, color: COL.free });
    C.render = () => {
      const [b, bd] = nearest();
      // circle of the nearest distance in the current geometry, drawn back in raw coordinates
      const ring = Array.from({ length: 121 }, (_, i) => { const a = i * Math.PI / 60, t = L.add(tr(q), [bd * Math.cos(a), bd * Math.sin(a)]);
        return mode === 'raw' ? t : mode === 'mm' ? t.map((v, j) => lo[j] + v * (hi[j] - lo[j])) : t.map((v, j) => mu[j] + v * sd[j]); });
      C.path(ring, 'rgba(255,200,87,.6)', { lw: 1.5, dash: [5, 4] });
      D.forEach((p, i) => C.dot(p, i === b ? 6 : 4, i === b ? COL.eig : COL.span));
      C.seg(q, D[b], COL.eig, { lw: 1.5 }); C.dot(q, 5, COL.free);
    };
    const m = seg(f.side, [{ label: 'raw', value: 'raw' }, { label: 'min–max', value: 'mm' }, { label: 'z-score', value: 'z' }], v => { mode = v; upd(); }, mode);
    const ro = readout(f.side);
    function upd() {
      const [b, bd] = nearest(), d = L.sub(D[b], q), dt = L.sub(tr(D[b]), tr(q));
      ro(`<div class="line">nearest neighbour: Feature 1 ${T(fmt(D[b][0], 0))}, Feature 2 ${T(fmt(D[b][1], 1))}</div>
        <div class="line">differences: ${T(`${fmt(d[0], 0)}`)} in Feature 1, ${T(`${fmt(d[1], 1)}`)} in Feature 2</div>
        <div class="line">distance in this geometry ${T(`\\sqrt{${fmt(dt[0] ** 2, 3)}+${fmt(dt[1] ** 2, 3)}}=${fmt(bd, 3)}`)}</div>
        <div class="line" style="color:var(--muted)">the notes' case: ${T('\\sqrt{1000^2+5^2}\\approx1000.01')}</div>`);
      C.draw();
    }
    upd();
    return { reset() { q = [6000, 2]; mode = 'raw'; m.set('raw'); upd(); }, setMode(v) { mode = v; m.set(v); upd(); }, state: () => ({ mode, nearest: nearest()[0] }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 2.1.2 Mean vs median on {2,3,3,4,100}; rolling mean and rolling median with a window w
export const SERIES = (() => { const r = rng(5); return Array.from({ length: 120 }, (_, t) => 3 * Math.sin(t / 15) + 0.02 * t + gauss(r) * 0.8 + ([30, 31, 72, 95].includes(t) ? 8 : 0)); })();
export function rolling(x, w, fn) { return x.map((_, t) => (t < w - 1 ? null : fn(x.slice(t - w + 1, t + 1)))); }
const aggregate = {
  title: 'Mean vs median, and rolling windows',
  mount(f) {
    H(f, 'clamp(330px, 50vw, 420px)');
    const host = el('div'); host.style.cssText = 'position:absolute;inset:0;display:grid;grid-template-rows:0.8fr 1.2fr'; f.stage.appendChild(host);
    const top = el('div'), bot = el('div'); top.style.position = bot.style.position = 'relative'; host.append(top, bot);
    let S = [2, 3, 3, 4, 100], w = 9;
    const A = new Chart(top, { x: [-2, 110], y: [0, 2], yticks: [], pad: [24, 14, 22, 14], grid: false });
    const B = new Chart(bot, { x: [0, 119], y: [-6, 14], xlabel: 't', pad: [22, 14, 22, 30] });
    top.appendChild(el('div', 'stage-t', `${T('\\{2,3,3,4,100\\}')}: drag a point`));
    bot.appendChild(el('div', 'stage-t', `series ${T('x_t')} with window ${T('w')}`));
    S.forEach((_, i) => A.handle({ get: () => [S[i], 1], set: p => { S[i] = Math.round(p[0]); upd(); }, color: COL.ink }));
    A.render = () => {
      const m = mean(S), md = median(S);
      A.seg([m, 0.4], [m, 1.6], COL.nul, { lw: 2 }); A.label([m, 1.6], `mean ${fmt(m, 2)}`, COL.nul, { dy: -8, align: 'center', font: MONO });
      A.seg([md, 0.4], [md, 1.6], COL.c1, { lw: 2, dash: [4, 3] }); A.label([md, 0.4], `median ${fmt(md, 2)}`, COL.c1, { dy: 10, align: 'center', base: 'top', font: MONO });
    };
    B.render = () => {
      B.path(SERIES.map((v, t) => [t, v]), 'rgba(232,236,248,.35)', { lw: 1.2 });
      const rm = rolling(SERIES, w, mean), rd = rolling(SERIES, w, median);
      B.path(rm.map((v, t) => [t, v]).filter(p => p[1] != null), COL.nul, { lw: 2.2 });
      B.path(rd.map((v, t) => [t, v]).filter(p => p[1] != null), COL.c1, { lw: 2, dash: [5, 3] });
      B.rect(119 - w + 1 - 0.5, -6, 119.5, 14, 'rgba(108,182,255,.08)', 'rgba(108,182,255,.35)');
      B.label([119 - w / 2, 14], 'window w', COL.blue ?? '#6cb6ff', { dy: 10, align: 'center', base: 'top', font: MONO });
    };
    const wS = slider(f.side, { label: '$w$', min: 1, max: 31, step: 1, value: w, fmt: v => String(v) }, v => { w = v; upd(); });
    legend(f.side, [[COL.nul, 'rolling mean $\\bar x_t$'], [COL.c1, 'rolling median $\\tilde x_t$'], ['rgba(232,236,248,.5)', 'raw (noisy)']]);
    const ro = readout(f.side);
    function upd() {
      ro(`<div class="line">${T(`\\text{mean}=${S.reduce((a, b) => a + b, 0)}/5=${fmt(mean(S), 2)}`)}, ${T(`\\text{median}=${fmt(median(S), 2)}`)}</div>
        <div class="line">first ${T(`w-1=${w - 1}`)} outputs undefined (no future samples used)</div>`);
      A.draw(); B.draw();
    }
    upd();
    return { reset() { S = [2, 3, 3, 4, 100]; w = 9; wS.set(9); upd(); }, state: () => ({ mean: mean(S), median: median(S) }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 2.1.3 Random vs stratified validation splits on a dataset with 5% positives
const stratified = {
  title: 'Random vs stratified splits: the share of positives in validation',
  mount(f) {
    H(f, 'clamp(260px, 36vw, 320px)');
    const N = 400, POS = 20, VAL = 80;   // 5% positives, a 20% validation split
    let R = rng(21), rnd = [], str = [];
    const C = new Chart(f.stage, { x: [0, 0.14], y: [0, 3], xticks: [0, 0.025, 0.05, 0.075, 0.1, 0.125].map(v => [v, (v * 100).toFixed(1) + '%']), yticks: [[1, 'random'], [2, 'stratified']], pad: [14, 14, 26, 70], xlabel: '' });
    const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
    function split1() {
      const idx = shuffle([...Array(N).keys()]).slice(0, VAL); rnd.push(idx.filter(i => i < POS).length / VAL);
      const pos = shuffle([...Array(POS).keys()]).slice(0, Math.round(POS * VAL / N)); str.push(pos.length / VAL);
    }
    C.render = () => {
      C.seg([0.05, 0.3], [0.05, 2.7], COL.eig, { dash: [4, 4] });
      const jitter = rng(9);
      rnd.forEach(v => C.dot([v, 1 + (jitter() - 0.5) * 0.6], 3, 'rgba(255,107,107,.55)'));
      str.forEach(v => C.dot([v, 2 + (jitter() - 0.5) * 0.6], 3, 'rgba(95,224,176,.55)'));
    };
    buttons(f.side, [{ label: '1 split', on: () => { split1(); upd(); } }, { label: '200 splits', pri: true, on: () => { for (let i = 0; i < 200; i++) split1(); upd(); } }, { label: 'Clear', on: () => { R = rng(21); rnd = []; str = []; upd(); } }]);
    const ro = readout(f.side);
    function upd() {
      const sd = a => a.length > 1 ? Math.sqrt(varPop(a)) : 0, pct = v => (100 * v).toFixed(1) + '%';
      ro(`<div class="line">${N} samples, ${POS} positives (5%), validation ${VAL}</div>
        ${rnd.length ? `<div class="line">random: positives from ${pct(Math.min(...rnd))} to ${pct(Math.max(...rnd))} (std ${pct(sd(rnd))})</div><div class="line">stratified: always ${pct(str[0])}</div>` : ''}`);
      C.draw();
    }
    for (let i = 0; i < 60; i++) split1();
    upd();
    return { reset() { R = rng(21); rnd = []; str = []; for (let i = 0; i < 60; i++) split1(); upd(); }, state: () => ({ rnd, str }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 2.1.4 Min-max vs z-score on the ages example
const scalers = {
  title: 'Min–max and z-score on the same ages',
  mount(f) {
    H(f, 'clamp(260px, 36vw, 310px)');
    let ages = [20, 30, 40, 60], out = false;
    const all = () => out ? [...ages, 100] : ages;
    const host = el('div'); host.style.cssText = 'position:absolute;inset:0;display:grid;grid-template-rows:repeat(3,1fr)'; f.stage.appendChild(host);
    const rows = ['ages', 'min–max $x\'$', 'z-score $x\'$'].map(t => { const d = el('div'); d.style.position = 'relative'; d.appendChild(el('div', 'stage-t', t.replace(/\$(.+?)\$/g, (_, m) => T(m)))); host.appendChild(d); return d; });
    const A = new Chart(rows[0], { x: [0, 110], y: [0, 1], yticks: [], pad: [22, 16, 20, 16], grid: false });
    const B = new Chart(rows[1], { x: [-0.1, 1.1], y: [0, 1], yticks: [], xticks: [0, 0.25, 0.5, 0.75, 1], pad: [22, 16, 20, 16], grid: false });
    const Z = new Chart(rows[2], { x: [-2.5, 2.5], y: [0, 1], yticks: [], pad: [22, 16, 20, 16], grid: false });
    ages.forEach((_, i) => A.handle({ get: () => [ages[i], 0.45], set: p => { ages[i] = Math.round(p[0]); upd(); }, color: COL.ink }));
    const mm = () => { const a = all(), lo = Math.min(...a), hi = Math.max(...a); return a.map(v => (v - lo) / (hi - lo)); };
    const zs = () => { const a = all(), m = mean(a), s = Math.sqrt(varPop(a)); return a.map(v => (v - m) / s); };
    const draw = (C, vals) => vals.forEach((v, i) => { const thirty = all()[i] === 30; C.dot([v, 0.45], thirty ? 6 : 4.5, i === 4 ? COL.nul : thirty ? COL.eig : COL.span); if (thirty) C.label([v, 0.45], fmt(v, 3), COL.eig, { dy: -13, align: 'center', font: MONO }); });
    A.render = () => draw(A, all()); B.render = () => draw(B, mm()); Z.render = () => draw(Z, zs());
    const ob = buttons(f.side, [{ label: 'add the outlier 100', pri: true, on: () => { out = !out; ob[0].textContent = out ? 'remove the outlier' : 'add the outlier 100'; upd(); } }]);
    const ro = readout(f.side);
    function upd() {
      const a = all(), lo = Math.min(...a), hi = Math.max(...a), m = mean(a), s = Math.sqrt(varPop(a));
      ro(`<div class="line">${T(`x_{\\min}=${lo},\\ x_{\\max}=${hi}`)}: ${a.includes(30) ? T(`30\\mapsto\\frac{30-${lo}}{${hi}-${lo}}=${fmt((30 - lo) / (hi - lo), 3)}`) : ''}</div>
        <div class="line">${T(`\\mu=${fmt(m, 2)},\\ \\sigma=${fmt(s, 2)}`)}${a.includes(30) ? `: ${T(`30\\mapsto${fmt((30 - m) / s, 3)}`)}` : ''}</div>
        <div class="line" style="color:var(--muted)">${T('\\sigma')} here is computed on these values (dividing by ${T('n')})</div>`);
      A.draw(); B.draw(); Z.draw();
    }
    upd();
    return { reset() { ages = [20, 30, 40, 60]; out = false; ob[0].textContent = 'add the outlier 100'; upd(); }, state: () => ({ mm: mm(), z: zs() }), toggle() { ob[0].click(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 2.1.5 Label encoding vs one-hot geometry
const onehot = {
  title: 'Label encoding vs one-hot: fake order vs equal distances',
  async mount(f) {
    H(f, 'clamp(300px, 44vw, 380px)');
    const THREE = await loadThree();
    const [hl, hr] = split(f, ['label encoding', 'one-hot']);
    const P = new Chart(hl, { x: [-0.5, 2.5], y: [-1, 1], yticks: [], xticks: [0, 1, 2], pad: [30, 16, 26, 16], grid: false });
    const cols = ['#ff6b6b', '#5fe0b0', '#6cb6ff'], names = ['Red', 'Green', 'Blue'];
    P.render = () => {
      [0, 1, 2].forEach(i => { P.dot([i, 0], 8, cols[i]); P.label([i, 0], names[i], cols[i], { dy: -18, align: 'center', font: FONT.ui }); });
      P.arrow([0, -0.35], [1, -0.35], COL.muted, { lw: 1.5, head: 7 }); P.label([0.5, -0.35], '|0−1| = 1', COL.ink, { dy: 12, align: 'center', base: 'top', font: MONO });
      P.arrow([0, 0.45], [2, 0.45], COL.nul, { lw: 1.5, head: 7 }); P.label([1, 0.45], '|0−2| = 2', COL.nul, { dy: -12, align: 'center', font: MONO });
    };
    const S = new Scene3(hr, { cam: [2.6, -2.9, 2.2], target: [0.35, 0.35, 0.3], minDist: 2, maxDist: 10 });
    S.axes(1.3);
    const E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    E.forEach((e, i) => { const a = S.arrow(new THREE.Color(cols[i]).getHex(), { r: 0.018, head: 0.12 }); a.set([0, 0, 0], e); const lb = S.label(`${names[i]} = e${'₁₂₃'[i]}`); lb.p = e.map(v => v * 1.18); lb.e.style.color = cols[i]; });
    for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) { S.line([E[i], E[j]], 0xffc857, { opacity: 0.8 }); const lb = S.label('√2'); lb.p = E[i].map((v, k) => (v + E[j][k]) / 2 + 0.04); lb.e.style.color = COL.eig; }
    const ro = readout(f.side);
    ro(`<div class="line">${T('e_i^Te_j=0\\ (i\\neq j)')}</div><div class="line">${T('\\|e_i-e_j\\|_2=\\sqrt2\\approx1.414')} for every pair</div><div class="line">${T('\\mathbf w^Te_j=w_j')}: one coefficient per category</div>`);
    return { reset() { S.fly([2.6, -2.9, 2.2], [0.35, 0.35, 0.3], 500); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 2.1.6 Equal-width vs equal-frequency vs K-means bins (the notes' counts 9,3,1,1 and 4,4,3,3)
export const BIN_DATA = [1, 1.5, 2, 2.5, 3, 3.3, 3.8, 4.6, 5.5, 7.5, 9.5, 12, 16, 25];
export function binEdges(x, k, method) {
  const s = [...x].sort((a, b) => a - b), lo = s[0], hi = s[s.length - 1];
  if (method === 'width') return Array.from({ length: k + 1 }, (_, i) => lo + (hi - lo) * i / k);
  if (method === 'freq') {   // split the sorted values into k nearly equal groups (first groups get the extra one), cut halfway between groups
    const n = s.length, sizes = Array.from({ length: k }, (_, i) => Math.floor(n / k) + (i < n % k ? 1 : 0)); const e = [lo]; let c = 0;
    for (let i = 0; i < k - 1; i++) { c += sizes[i]; e.push((s[c - 1] + s[c]) / 2); } e.push(hi); return e;
  }
  // K-means in 1D (Lloyd's algorithm from quantile starts until nothing changes); bins cut halfway between sorted centres
  let cen = Array.from({ length: k }, (_, i) => s[Math.min(s.length - 1, Math.floor((i + 0.5) * s.length / k))]);
  for (let it = 0; it < 100; it++) {
    const g = cen.map(() => []); s.forEach(v => { let b = 0; cen.forEach((c, j) => { if (Math.abs(v - c) < Math.abs(v - cen[b])) b = j; }); g[b].push(v); });
    const nc = g.map((a, j) => a.length ? mean(a) : cen[j]); if (nc.every((c, j) => Math.abs(c - cen[j]) < 1e-12)) break; cen = nc;
  }
  cen.sort((a, b) => a - b); return [lo, ...cen.slice(1).map((c, j) => (c + cen[j]) / 2), hi];
}
export const binCounts = (x, e) => e.slice(1).map((r, i) => x.filter(v => (i === 0 ? v >= e[0] : v > e[i]) && v <= r).length);
const binning = {
  title: 'Three ways to cut the same values into bins',
  mount(f) {
    H(f, 'clamp(240px, 32vw, 290px)');
    let X = BIN_DATA.slice(), k = 4;
    const host = el('div'); host.style.cssText = 'position:absolute;inset:0;display:grid;grid-template-rows:repeat(3,1fr)'; f.stage.appendChild(host);
    const M = [['width', 'equal-width'], ['freq', 'equal-frequency'], ['kmeans', 'K-means']];
    const Cs = M.map(([, t]) => { const d = el('div'); d.style.position = 'relative'; d.appendChild(el('div', 'stage-t', t)); host.appendChild(d); return new Chart(d, { x: [-1, 27], y: [0, 1], yticks: [], pad: [22, 16, 18, 16], grid: false }); });
    Cs.forEach((C, r) => {
      X.forEach((_, i) => C.handle({ get: () => [X[i], 0.4], set: p => { X[i] = Math.max(0, Math.min(26, Math.round(p[0] * 10) / 10)); upd(); }, color: COL.span, ring: false }));
      C.render = () => {
        const e = binEdges(X, k, M[r][0]), cnt = binCounts(X, e);
        e.forEach(v => C.seg([v, 0.1], [v, 0.85], COL.nul, { lw: 2 }));
        cnt.forEach((c, i) => C.label([(e[i] + e[i + 1]) / 2, 0.85], String(c), COL.eig, { dy: -2, align: 'center', font: MONO }));
        X.forEach(v => C.dot([v, 0.4], 4, COL.span));
      };
    });
    const kS = slider(f.side, { label: '$k$', min: 2, max: 6, step: 1, value: k, fmt: v => String(v) }, v => { k = v; upd(); });
    const ro = readout(f.side);
    function upd() {
      ro(M.map(([m, t]) => { const e = binEdges(X, k, m); return `<div class="line">${t}: counts ${T(binCounts(X, e).join(',\\ '))}</div>`; }).join('') +
        `<div class="line">${T(`\\text{width}=\\frac{${fmt(Math.max(...X))}-${fmt(Math.min(...X))}}{${k}}=${fmt((Math.max(...X) - Math.min(...X)) / k, 2)}`)}</div>`);
      Cs.forEach(C => C.draw());
    }
    upd();
    return { reset() { X = BIN_DATA.slice(); k = 4; kS.set(4); upd(); }, state: () => Object.fromEntries(M.map(([m]) => [m, binCounts(X, binEdges(X, k, m))])) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 2.1.8 PCA: projected variance along a rotating axis peaks at the top eigenvector of Sigma
export const PCA_DATA = (() => { const r = rng(17); return Array.from({ length: 40 }, () => { const a = gauss(r) * 1.6, b = gauss(r) * 0.55; const c = Math.cos(0.6), s = Math.sin(0.6); return [0.4 + c * a - s * b, 0.3 + s * a + c * b]; }); })();
export function pcaStats(P) {
  const n = P.length, mu = [0, 1].map(j => mean(P.map(p => p[j]))), Xc = P.map(p => L.sub(p, mu));
  const S = [[0, 0], [0, 0]]; for (const x of Xc) for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) S[i][j] += x[i] * x[j] / (n - 1);
  return { mu, Xc, S, eig: L.eig2(S) };
}
const pca = {
  title: 'PCA: the axis that keeps the most variance',
  mount(f) {
    H(f, 'clamp(320px, 48vw, 420px)');
    let P = PCA_DATA.map(p => p.slice()), th = 0.15, stop = null;
    const C = new Plane(f.stage, { extent: 3.4 });
    const reh = () => { C.handles = []; P.forEach((_, i) => C.handle({ get: () => P[i], set: q => { P[i] = q; upd(); }, color: COL.span, ring: false })); C.handle({ get: () => { const { mu } = pcaStats(P); return L.add(mu, [2.8 * Math.cos(th), 2.8 * Math.sin(th)]); }, set: q => { const { mu } = pcaStats(P); th = Math.atan2(q[1] - mu[1], q[0] - mu[0]); upd(); }, color: COL.eig }); };
    C.render = () => {
      const { mu, Xc, eig } = pcaStats(P), u = [Math.cos(th), Math.sin(th)];
      C.lineDir(mu, u, COL.eig, { lw: 2 });
      if (eig.real) { C.lineDir(mu, eig.vecs[0], 'rgba(255,107,107,.45)', { lw: 1.4, dash: [6, 5] }); C.lineDir(mu, eig.vecs[1], 'rgba(47,183,166,.45)', { lw: 1.4, dash: [6, 5] }); }
      Xc.forEach((x, i) => { const t = L.dot(x, u), foot = L.add(mu, L.scale(u, t)); C.seg(P[i], foot, 'rgba(232,236,248,.22)', { lw: 1 }); C.dot(foot, 2.6, COL.eig); });
      P.forEach(p => C.dot(p, 4, COL.span));
      C.label(L.add(mu, L.scale(eig.vecs[0], 3.1)), 'PC1', '#ff8f8f', { font: FONT.ui }); C.label(L.add(mu, L.scale(eig.vecs[1], 2.2)), 'PC2', '#5fd4c4', { font: FONT.ui });
    };
    const plot = el('div'); plot.style.cssText = 'position:relative;height:120px;border:1px solid var(--line-soft);border-radius:8px'; f.side.appendChild(plot);
    const V = new Chart(plot, { x: [0, Math.PI], y: [0, 3.2], xticks: [[0, '0°'], [Math.PI / 2, '90°'], [Math.PI, '180°']], pad: [10, 8, 20, 30] });
    V.render = () => {
      const { S } = pcaStats(P), pv = a => { const u = [Math.cos(a), Math.sin(a)]; return L.dot(u, L.mv(S, u)); };
      V.o.y = [0, Math.max(...Array.from({ length: 60 }, (_, i) => pv(i * Math.PI / 60))) * 1.15];
      V.fn(pv, COL.eig, { lw: 2 }); const a = ((th % Math.PI) + Math.PI) % Math.PI; V.dot([a, pv(a)], 4.5, COL.ink);
    };
    buttons(f.side, [{ label: 'snap to PC1', pri: true, on: () => { const { eig } = pcaStats(P), t1 = Math.atan2(eig.vecs[0][1], eig.vecs[0][0]); const t0 = th; let d = t1 - t0; d = ((d + Math.PI / 2) % Math.PI + Math.PI) % Math.PI - Math.PI / 2; stop?.(); const s = performance.now(); const step = () => { const u = Math.min(1, (performance.now() - s) / 700); th = t0 + d * (1 - (1 - u) ** 3); upd(); if (u < 1) requestAnimationFrame(step); }; requestAnimationFrame(step); } }]);
    const ro = readout(f.side);
    function upd() {
      const { S, eig, Xc } = pcaStats(P), u = [Math.cos(th), Math.sin(th)], pv = varSample(Xc.map(x => L.dot(x, u)));
      const tot = eig.vals[0] + eig.vals[1];
      ro(`<div class="eqx">${TD(`\\Sigma=\\tfrac{1}{n-1}X_c^TX_c=${matTex(S, 3)}`)}</div>
        <div class="line">variance along the axis: ${T(fmt(pv, 3))}</div>
        <div class="line">${T(`\\lambda_1=${fmt(eig.vals[0], 3)},\\ \\lambda_2=${fmt(eig.vals[1], 3)}`)}</div>
        <div class="line">${T(`\\text{EVR}_1=${fmt(100 * eig.vals[0] / tot, 1)}\\%,\\ \\text{EVR}_2=${fmt(100 * eig.vals[1] / tot, 1)}\\%`)}</div>`);
      C.draw(); V.draw();
    }
    reh(); upd();
    return { reset() { P = PCA_DATA.map(p => p.slice()); th = 0.15; reh(); upd(); }, state: () => { const s = pcaStats(P); return { th, ...s }; }, setTheta(v) { th = v; upd(); } };
  },
};

export default { 'scale': scale, 'aggregate': aggregate, 'stratified': stratified, 'scalers': scalers, 'onehot': onehot, 'binning': binning, 'pca': pca };
