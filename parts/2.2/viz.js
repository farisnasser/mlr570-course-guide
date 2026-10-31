// Part 2.2 Similarity Measures — interactive visuals on the notes' examples (B = 2A, B = A^2, A = [1,2,3], C = 2A,
// the Jaccard sets / binary vectors) and a real distance simulation for the curse of dimensionality.
import { Plane, Chart, COL, FONT } from '../../src/viz/plane.js';
import * as L from '../../src/viz/la.js';
import { el, T, TD, fmt, seg, slider, buttons, readout, legend } from '../../src/viz/ui.js';
import { rng, mean, varPop } from '../../src/viz/stats.js';

const H = (f, h) => { f.stage.style.height = h; };
const MONO = '600 11.5px "JetBrains Mono", monospace';
export const pearson = (A, B) => { const ma = mean(A), mb = mean(B); const c = mean(A.map((a, i) => (a - ma) * (B[i] - mb))); return c / Math.sqrt(varPop(A) * varPop(B)); };

// ---------------------------------------------------------------------------------------------------------------
const pearsonViz = {
  title: 'Pearson correlation: linear association only',
  mount(f) {
    H(f, 'clamp(290px, 42vw, 370px)');
    const PRE = { line: [-2, -1, 0, 1, 2].map(a => [a + 3, 2 * (a + 3)]), sq: [-2, -1, 0, 1, 2].map(a => [a, a * a]) };
    let pts = PRE.line.map(p => p.slice()), which = 'line';
    const C = new Chart(f.stage, { x: [0, 6], y: [0, 11], xlabel: 'A', ylabel: 'B', pad: [14, 14, 26, 34] });
    const reh = () => { C.handles = []; pts.forEach((_, i) => C.handle({ get: () => pts[i], set: p => { pts[i] = p; upd(); }, color: COL.span })); };
    C.render = () => {
      const A = pts.map(p => p[0]), B = pts.map(p => p[1]), ma = mean(A), mb = mean(B), sl = mean(A.map((a, i) => (a - ma) * (B[i] - mb))) / varPop(A);
      C.fn(x => mb + sl * (x - ma), 'rgba(255,200,87,.6)', { lw: 1.6, dash: [6, 4] });
      if (which === 'sq') C.fn(x => x * x, 'rgba(255,107,107,.5)', { lw: 1.5 });
      pts.forEach(p => C.dot(p, 5, COL.span));
    };
    const s = seg(f.side, [{ label: '$B=2A$', value: 'line' }, { label: '$B=A^2$', value: 'sq' }], v => { which = v; pts = PRE[v].map(p => p.slice()); C.o.x = v === 'sq' ? [-3, 3] : [0, 6]; C.o.y = v === 'sq' ? [-1, 6] : [0, 11]; reh(); upd(); }, which);
    legend(f.side, [['rgba(255,200,87,.8)', 'best straight line (least squares)']]);
    const ro = readout(f.side);
    function upd() {
      const A = pts.map(p => p[0]), B = pts.map(p => p[1]), r = pearson(A, B);
      ro(`<div class="line">${T(`A=[${A.map(v => fmt(v, 1)).join(',')}]`)}</div><div class="line">${T(`B=[${B.map(v => fmt(v, 1)).join(',')}]`)}</div>
        <div class="line">${T(`\\operatorname{Cov}(A,B)=${fmt(mean(A.map((a, i) => (a - mean(A)) * (B[i] - mean(B)))), 3)}`)}</div>
        <div class="line">${T(`\\rho(A,B)=${isFinite(r) ? fmt(r, 3) : '\\text{undefined}'}`)}</div>`);
      C.draw();
    }
    reh(); upd();
    return { reset() { s.set('line'); which = 'line'; pts = PRE.line.map(p => p.slice()); C.o.x = [0, 6]; C.o.y = [0, 11]; reh(); upd(); }, state: () => pearson(pts.map(p => p[0]), pts.map(p => p[1])), set(v) { which = v; s.set(v); pts = PRE[v].map(p => p.slice()); C.o.x = v === 'sq' ? [-3, 3] : [0, 6]; C.o.y = v === 'sq' ? [-1, 6] : [0, 11]; reh(); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
const distances = {
  title: 'Euclidean straight line vs Manhattan staircase',
  mount(f) {
    H(f, 'clamp(290px, 42vw, 360px)');
    let A = [-2, -1.5], B = [2, 1.5], seed = 1;
    const P = new Plane(f.stage, { extent: 3 });
    P.handle({ get: () => A, set: p => { A = p.map(v => Math.round(v * 2) / 2); upd(); }, color: COL.ink });
    P.handle({ get: () => B, set: p => { B = p.map(v => Math.round(v * 2) / 2); upd(); }, color: COL.ink });
    const stair = r => { // a random monotone grid path from A to B in steps of 0.5
      const nx = Math.round(Math.abs(B[0] - A[0]) * 2), ny = Math.round(Math.abs(B[1] - A[1]) * 2), sx = Math.sign(B[0] - A[0]) * 0.5, sy = Math.sign(B[1] - A[1]) * 0.5;
      const moves = [...Array(nx).fill(0), ...Array(ny).fill(1)]; for (let i = moves.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [moves[i], moves[j]] = [moves[j], moves[i]]; }
      const pts = [A.slice()]; let c = A.slice(); for (const m of moves) { c = m ? [c[0], c[1] + sy] : [c[0] + sx, c[1]]; pts.push(c); } return pts;
    };
    P.render = () => {
      const r = rng(seed);
      for (let k = 0; k < 6; k++) P.path(stair(r), k ? 'rgba(255,107,107,.35)' : COL.nul, { lw: k ? 1.4 : 2.6 });
      P.seg(A, B, COL.span, { lw: 2.6 });
      P.label(A, 'A', COL.ink, { dx: -14, dy: 10, font: FONT.math }); P.label(B, 'B', COL.ink, { dx: 8, dy: -10, font: FONT.math });
    };
    buttons(f.side, [{ label: 'other staircases', on: () => { seed++; P.draw(); } }]);
    legend(f.side, [[COL.span, 'Euclidean: straight line'], [COL.nul, 'Manhattan: along the axes']]);
    const ro = readout(f.side);
    function upd() {
      const d = L.sub(A, B).map(Math.abs);
      ro(`<div class="line">${T(`\\sqrt{${fmt(d[0])}^2+${fmt(d[1])}^2}=${fmt(Math.hypot(...d), 3)}`)} (Euclidean)</div>
        <div class="line">${T(`${fmt(d[0])}+${fmt(d[1])}=${fmt(d[0] + d[1], 3)}`)} (Manhattan, every staircase)</div>
        <div class="line" style="color:var(--muted)">the notes' ${T('A=[1,2,3],\\ B=[4,5,6]')}: ${T(`\\sqrt{27}\\approx${fmt(Math.sqrt(27), 3)}`)} and ${T('9')}; differences ${T('[1,1,10]')}: ${T(`${fmt(Math.sqrt(102), 1)}`)} and ${T('12')}</div>`);
      P.draw();
    }
    upd();
    return { reset() { A = [-2, -1.5]; B = [2, 1.5]; seed = 1; upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// Curse of dimensionality: 1000 uniform points in [0,1]^d, one query; distances / average
export function distStats(d, seed = 1, n = 1000, metric = 'l2') {
  const r = rng(seed + d * 7919), q = Array.from({ length: d }, r), ds = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < d; j++) { const z = r() - q[j]; s += metric === 'l2' ? z * z : Math.abs(z); } ds[i] = metric === 'l2' ? Math.sqrt(s) : s; }
  let mn = Infinity, mx = -Infinity, sm = 0; for (const v of ds) { mn = Math.min(mn, v); mx = Math.max(mx, v); sm += v; }
  return { ds, min: mn, max: mx, avg: sm / n, contrast: (mx - mn) / mn };
}
const curse = {
  title: 'The curse of dimensionality: nearest and farthest squeeze together',
  mount(f) {
    H(f, 'clamp(300px, 44vw, 380px)');
    const host = el('div'); host.style.cssText = 'position:absolute;inset:0;display:grid;grid-template-rows:0.8fr 1.2fr'; f.stage.appendChild(host);
    const top = el('div'), bot = el('div'); top.style.position = bot.style.position = 'relative'; host.append(top, bot);
    top.appendChild(el('div', 'stage-t', 'every distance ÷ average'));
    bot.appendChild(el('div', 'stage-t', `${T('(d_{\\max}-d_{\\min})/d_{\\min}')} against ${T('d')} (log scale)`));
    let dim = 2, metric = 'l2';
    const A = new Chart(top, { x: [0, 2.2], y: [0, 1], yticks: [], xticks: [0, 0.5, 1, 1.5, 2], pad: [24, 14, 20, 30], grid: false });
    const D = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];
    const B = new Chart(bot, { x: [0, 3], y: [-1.5, 5], xticks: [0, 1, 2, 3].map(v => [v, String(10 ** v)]), yticks: [-1, 0, 1, 2, 3, 4, 5].map(v => [v, v === 0 ? '1' : `10^${v}`]), pad: [22, 14, 24, 40], xlabel: 'd' });
    const cache = {}; const st = (d, m) => (cache[d + m] ??= distStats(d, 1, 1000, m));
    A.render = () => {
      const s = st(dim, metric), r = rng(3);
      for (const v of s.ds) A.dot([v / s.avg, 0.3 + r() * 0.4], 1.6, 'rgba(124,196,255,.5)');
      A.seg([s.min / s.avg, 0.1], [s.min / s.avg, 0.9], COL.c1, { lw: 2.5 }); A.seg([s.max / s.avg, 0.1], [s.max / s.avg, 0.9], COL.nul, { lw: 2.5 }); A.seg([1, 0.1], [1, 0.9], COL.eig, { dash: [4, 3] });
      A.label([s.min / s.avg, 0.9], 'nearest', COL.c1, { dy: -8, align: 'center', font: MONO }); A.label([s.max / s.avg, 0.9], 'farthest', COL.nul, { dy: -8, align: 'center', font: MONO });
    };
    B.render = () => {
      for (const [m, c] of [['l2', COL.span], ['l1', COL.c2]]) B.path(D.map(d => [Math.log10(d), Math.log10(st(d, m).contrast)]), c, { lw: m === metric ? 2.6 : 1.4, alpha: m === metric ? 1 : 0.5 });
      B.dot([Math.log10(dim), Math.log10(st(dim, metric).contrast)], 5, COL.ink);
    };
    const dS = slider(f.side, { label: '$d$', min: 0, max: 10, step: 1, value: 1, fmt: i => String(D[i]) }, i => { dim = D[i]; upd(); });
    seg(f.side, [{ label: 'Euclidean', value: 'l2' }, { label: 'Manhattan', value: 'l1' }], v => { metric = v; upd(); }, metric);
    legend(f.side, [[COL.span, 'Euclidean'], [COL.c2, 'Manhattan']]);
    const ro = readout(f.side);
    function upd() {
      const s = st(dim, metric);
      ro(`<div class="line">${T(`d=${dim}`)}: ${T(`d_{\\min}=${fmt(s.min, 3)},\\ d_{\\max}=${fmt(s.max, 3)}`)}</div><div class="line">${T(`\\frac{d_{\\max}-d_{\\min}}{d_{\\min}}=${fmt(s.contrast, 3)}`)}</div>`);
      A.draw(); B.draw();
    }
    upd();
    return { reset() { dim = 2; dS.set(1); upd(); }, stats: (d, m = 'l2') => st(d, m) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
const cosine = {
  title: 'Cosine asks about the angle, Euclidean about the gap',
  mount(f) {
    H(f, 'clamp(290px, 42vw, 360px)');
    let A = [2.2, 0.9], B0 = [1, 2.1], k = 1;
    const P = new Plane(f.stage, { extent: 3.3, center: [1, 1] });
    const B = () => L.scale(B0, k);
    P.handle({ get: () => A, set: p => { A = p; upd(); }, color: COL.span });
    P.handle({ get: () => B(), set: p => { B0 = L.scale(p, 1 / k); upd(); }, color: COL.nul });
    P.render = () => {
      const a1 = Math.atan2(A[1], A[0]), a2 = Math.atan2(B()[1], B()[0]);
      P.path(Array.from({ length: 31 }, (_, i) => { const t = a1 + (a2 - a1) * i / 30; return [0.6 * Math.cos(t), 0.6 * Math.sin(t)]; }), COL.eig, { lw: 1.5 });
      P.seg(A, B(), 'rgba(232,236,248,.5)', { dash: [4, 4] });
      P.arrow([0, 0], A, COL.span); P.arrow([0, 0], B(), COL.nul);
      P.tipLabel(A, 'A', COL.span); P.tipLabel(B(), 'B', COL.nul);
      P.label([0.75 * Math.cos((a1 + a2) / 2), 0.75 * Math.sin((a1 + a2) / 2)], 'θ', COL.eig, { align: 'center' });
    };
    const kS = slider(f.side, { label: 'stretch $B$', min: 0.3, max: 2.5, step: 0.05, value: 1, fmt: v => '×' + v.toFixed(2) }, v => { k = v; upd(); });
    const ro = readout(f.side);
    function upd() {
      const b = B(), c = L.dot(A, b) / (L.norm(A) * L.norm(b));
      ro(`<div class="line">${T(`\\cos(A,B)=\\frac{A\\cdot B}{\\|A\\|\\,\\|B\\|}=\\frac{${fmt(L.dot(A, b), 3)}}{${fmt(L.norm(A), 3)}\\cdot${fmt(L.norm(b), 3)}}=${fmt(c, 3)}`)}</div>
        <div class="line">${T(`\\theta=${fmt(Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI, 1)}^\\circ`)}</div>
        <div class="line">Euclidean ${T(`\\|A-B\\|=${fmt(L.norm(L.sub(A, b)), 3)}`)}</div>
        <div class="line" style="color:var(--muted)">the notes' ${T('A=[1,2,3],\\ C=2A')}: ${T('\\cos=28/28=1')}, ${T(`\\|A-C\\|=\\sqrt{14}\\approx${fmt(Math.sqrt(14), 2)}`)}</div>`);
      P.draw();
    }
    upd();
    return { reset() { A = [2.2, 0.9]; B0 = [1, 2.1]; k = 1; kS.set(1); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
export const jac = (a, b) => { const inter = a.filter((v, i) => v && b[i]).length, uni = a.filter((v, i) => v || b[i]).length; return uni ? inter / uni : NaN; };
const jaccard = {
  title: 'Jaccard on binary vectors: shared absences never count',
  mount(f) {
    f.stage.style.minHeight = '0';
    let A = [1, 0, 1, 0, 1], B = [1, 1, 0, 0, 1];
    const box = el('div'); box.style.cssText = 'padding:16px;display:grid;gap:8px;justify-content:center'; f.stage.appendChild(box);
    const ro = readout(f.side);
    function upd() {
      const cs = 'width:44px;height:38px;border:1px solid var(--line);text-align:center;font:600 14px var(--mono);cursor:pointer;border-radius:6px';
      const cell = (v, i, row) => { const both = A[i] && B[i], none = !A[i] && !B[i]; return `<button type="button" data-r="${row}" data-i="${i}" style="${cs};background:${v ? (both ? 'rgba(47,183,166,.45)' : 'rgba(124,196,255,.22)') : 'transparent'};color:${none ? 'var(--faint)' : 'var(--ink)'}">${v}</button>`; };
      box.innerHTML = `<div style="display:grid;grid-template-columns:auto repeat(5,44px);gap:6px;align-items:center"><span></span>${[1, 2, 3, 4, 5].map(i => `<span style="text-align:center;font:500 11px var(--mono);color:var(--muted)">${i}</span>`).join('')}
        <span style="font:italic 600 15px var(--serif);padding-right:8px">A</span>${A.map((v, i) => cell(v, i, 'A')).join('')}<span style="font:italic 600 15px var(--serif);padding-right:8px">B</span>${B.map((v, i) => cell(v, i, 'B')).join('')}</div>`;
      box.querySelectorAll('button').forEach(b => b.onclick = () => { const arr = b.dataset.r === 'A' ? A : B; arr[+b.dataset.i] ^= 1; upd(); });
      const inter = A.map((v, i) => v && B[i] ? i + 1 : 0).filter(Boolean), uni = A.map((v, i) => v || B[i] ? i + 1 : 0).filter(Boolean), absent = A.map((v, i) => !v && !B[i] ? i + 1 : 0).filter(Boolean);
      ro(`<div class="line">shared: ${T(`\\{${inter.join(',')}\\}`)}, union: ${T(`\\{${uni.join(',')}\\}`)}</div>
        <div class="line">${T(`J=\\frac{${inter.length}}{${uni.length}}=${uni.length ? fmt(inter.length / uni.length, 3) : '\\text{undefined}'}`)}</div>
        ${absent.length ? `<div class="line">position${absent.length > 1 ? 's' : ''} ${absent.join(', ')}: 0 in both, ignored</div>` : ''}`);
    }
    upd();
    return { reset() { A = [1, 0, 1, 0, 1]; B = [1, 1, 0, 0, 1]; upd(); }, state: () => jac(A, B) };
  },
};

export default { 'pearson': pearsonViz, 'distances': distances, 'curse': curse, 'cosine': cosine, 'jaccard': jaccard };
