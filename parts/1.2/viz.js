// Part 1.2 Probability & Statistics — interactive visuals. Exact pdf/cdf and real sampling; the notes' examples are
// the defaults (recomputed in scripts/check/1_2.py).
import { Plane, Chart, COL, FONT } from '../../src/viz/plane.js';
import * as L from '../../src/viz/la.js';
import { el, T, TD, fmt, seg, slider, buttons, readout, legend } from '../../src/viz/ui.js';
import { normPdf, normCdf, DIST, rng, mean, varPop, varSample } from '../../src/viz/stats.js';
import { loadThree, Scene3 } from '../../src/viz/three-lite.js';

const H = (f, h) => { f.stage.style.height = h; };
const MONO = '600 11.5px "JetBrains Mono", monospace';
function split(f, titles) {
  const sp = el('div', 'split'); f.stage.appendChild(sp);
  return titles.map(t => { const d = el('div'); sp.appendChild(d); if (t) d.appendChild(el('div', 'stage-t', t)); return d; });
}
/** p as a fraction with a small denominator when it is one (1/4, 3/4, 2/3...), else a decimal */
function frac(p, d = 3) {
  for (const q of [2, 3, 4, 5, 8, 10, 16, 20, 80]) { const k = Math.round(p * q); if (Math.abs(p * q - k) < 1e-9) return k === 0 ? '0' : k === q ? '1' : `\\tfrac{${k / gcd(k, q)}}{${q / gcd(k, q)}}`; }
  return fmt(p, d);
}
const gcd = (a, b) => (b ? gcd(b, a % b) : a);

// ---------------------------------------------------------------------------------------------------------------
// 1.2.1 PMF <-> CDF on the notes' example p(0)=1/4, p(1)=1/2, p(2)=1/4
const pmfCdf = {
  title: 'PMF and CDF: probability at a value vs. accumulated so far',
  mount(f) {
    H(f, 'clamp(240px, 34vw, 300px)');
    const [hl, hr] = split(f, [`PMF ${T('p_X(x)')}`, `CDF ${T('F_X(x)')}`]);
    const xs = [0, 1, 2], P0 = [0.25, 0.5, 0.25];
    let p = P0.slice(), x = 1;
    const A = new Chart(hl, { x: [-0.6, 2.6], y: [0, 1.05], xticks: xs, yticks: [0, 0.25, 0.5, 0.75, 1], xlabel: 'x', pad: [26, 12, 26, 40] });
    const B = new Chart(hr, { x: [-0.6, 2.6], y: [0, 1.05], xticks: xs, yticks: [0, 0.25, 0.5, 0.75, 1], xlabel: 'x', pad: [26, 12, 26, 40] });
    const setP = (k, v) => {
      v = Math.max(0, Math.min(1, v)); const rest = p.reduce((s, q, i) => s + (i === k ? 0 : q), 0);
      p = p.map((q, i) => i === k ? v : rest > 1e-12 ? q / rest * (1 - v) : (1 - v) / 2); upd();
    };
    xs.forEach((xv, k) => A.handle({ get: () => [xv, p[k]], set: q => setP(k, q[1]), color: COL.nul, snap: 0.05 }));
    A.handle({ get: () => [x, 0], set: q => { x = q[0]; upd(); }, color: COL.free });
    B.handle({ get: () => [x, 0], set: q => { x = q[0]; upd(); }, color: COL.free });
    const F = t => xs.reduce((s, xv, k) => s + (xv <= t + 1e-12 ? p[k] : 0), 0);
    A.render = () => {
      xs.forEach((xv, k) => A.rect(xv - 0.18, 0, xv + 0.18, p[k], xv <= x + 1e-12 ? 'rgba(255,107,107,.75)' : 'rgba(255,107,107,.28)'));
      A.seg([x, 0], [x, 1.05], COL.free, { dash: [4, 4] });
    };
    B.render = () => {
      let acc = 0; const steps = [[-0.6, 0]];
      xs.forEach((xv, k) => { B.seg([steps[steps.length - 1][0], acc], [xv, acc], COL.span, { lw: 2.4 }); if (k > 0 || xv > -0.6) B.dot([xv, acc], 4, COL.span, { ring: true }); acc += p[k]; B.dot([xv, acc], 4.5, COL.span); steps.push([xv, acc]); });
      B.seg([2, acc], [2.6, acc], COL.span, { lw: 2.4 });
      B.seg([x, 0], [x, 1.05], COL.free, { dash: [4, 4] }); B.dot([x, F(x)], 5, COL.free);
      B.label([x, F(x)], `F(${fmt(x, 2)}) = ${fmt(F(x), 3)}`, COL.free, { dx: 8, dy: -12, font: MONO });
    };
    const ro = readout(f.side);
    function upd() {
      const terms = xs.filter(v => v <= x + 1e-12);
      ro(`<div class="line">${T(`p_X=\\left(${p.map(q => frac(q)).join(',\\ ')}\\right)`)}, ${T(`\\sum_x p_X(x)=${fmt(p.reduce((a, b) => a + b, 0), 6)}`)}</div>
        <div class="line">${T(`F_X(${fmt(x, 2)})=P(X\\le ${fmt(x, 2)})=${terms.length ? terms.map(v => `P(X=${v})`).join('+') : '0'}`)}</div>
        <div class="line">${T(`=${terms.length ? terms.map(v => frac(p[v])).join('+') + '=' : ''}${frac(F(x))}`)}</div>`);
      A.draw(); B.draw();
    }
    upd();
    return { reset() { p = P0.slice(); x = 1; upd(); }, state: () => ({ p, x, F: F(x) }), set(o) { if (o.p) p = o.p; if (o.x != null) x = o.x; upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.2.1 PDF area = probability = rise of the CDF
const pdfArea = {
  title: 'Probability is an area: $P(a\\le X\\le b)=F_X(b)-F_X(a)$',
  mount(f) {
    H(f, 'clamp(300px, 44vw, 380px)');
    const host = el('div'); host.style.cssText = 'position:absolute;inset:0;display:grid;grid-template-rows:1.25fr 1fr'; f.stage.appendChild(host);
    const top = el('div'), bot = el('div'); top.style.position = bot.style.position = 'relative'; host.append(top, bot);
    top.appendChild(el('div', 'stage-t', `PDF ${T('f_X(x)')}`)); bot.appendChild(el('div', 'stage-t', `CDF ${T('F_X(x)')}`));
    let kind = 'normal', P = { mu: 0, s: 1, l: 1, a: -2, b: 2 }, a = -1, b = 1;
    const dist = () => DIST[kind], pdf = x => dist().pdf(x, P), cdf = x => dist().cdf(x, P);
    const xr = () => kind === 'exponential' ? [-0.5, 5] : kind === 'uniform' ? [-3, 3] : [P.mu - 4 * P.s, P.mu + 4 * P.s];
    const C1 = new Chart(top, { x: xr(), y: [0, 0.5], pad: [24, 14, 22, 40], xlabel: 'x' });
    const C2 = new Chart(bot, { x: xr(), y: [0, 1.08], yticks: [0, 0.5, 1], pad: [20, 14, 22, 40], xlabel: 'x' });
    const ymax = () => Math.max(...Array.from({ length: 200 }, (_, i) => pdf(xr()[0] + (xr()[1] - xr()[0]) * i / 199))) * 1.18;
    const clampAB = () => { if (a > b) [a, b] = [b, a]; };
    C1.handle({ get: () => [a, 0], set: q => { a = q[0]; if (a > b) a = b; upd(); }, color: COL.c1 });
    C1.handle({ get: () => [b, 0], set: q => { b = q[0]; if (b < a) b = a; upd(); }, color: COL.c2 });
    C1.render = () => {
      const [x0, x1] = C1.o.x, N = 300, pts = Array.from({ length: N + 1 }, (_, i) => { const x = x0 + (x1 - x0) * i / N; return [x, pdf(x)]; });
      const inner = [[a, pdf(a)], ...pts.filter(([x]) => x > a && x < b), [b, pdf(b)]];
      C1.area(inner, 'rgba(255,107,107,.28)');
      C1.path(pts, COL.nul, { lw: 2.2 });
      C1.seg([a, 0], [a, pdf(a)], COL.c1, { dash: [4, 3] }); C1.seg([b, 0], [b, pdf(b)], COL.c2, { dash: [4, 3] });
      C1.label([a, 0], 'a', COL.c1, { dy: -16, align: 'center' }); C1.label([b, 0], 'b', COL.c2, { dy: -16, align: 'center' });
    };
    C2.render = () => {
      C2.fn(cdf, COL.span, { lw: 2.2, n: 400 });
      C2.seg([a, 0], [a, cdf(a)], COL.c1, { dash: [4, 3] }); C2.seg([b, 0], [b, cdf(b)], COL.c2, { dash: [4, 3] });
      C2.seg([a, cdf(a)], [b, cdf(a)], 'rgba(232,236,248,.4)', { dash: [3, 3] });
      C2.seg([b, cdf(a)], [b, cdf(b)], COL.nul, { lw: 3.5 });
      C2.dot([a, cdf(a)], 4, COL.c1); C2.dot([b, cdf(b)], 4, COL.c2);
      C2.label([b, (cdf(a) + cdf(b)) / 2], `${fmt(cdf(b) - cdf(a), 4)}`, COL.nul, { dx: 8, font: MONO });
    };
    const ks = seg(f.side, [{ label: 'Normal$(\\mu,\\sigma^2)$', value: 'normal' }, { label: 'Uniform$(-2,2)$', value: 'uniform' }, { label: 'Exponential$(\\lambda)$', value: 'exponential' }], v => { kind = v; a = v === 'exponential' ? 0.5 : -1; b = v === 'exponential' ? 2 : 1; box.hidden = v !== 'normal'; lb.hidden = v !== 'exponential'; reframe(); upd(); }, kind);
    const box = el('div'); box.style.cssText = 'display:grid;gap:6px'; f.side.appendChild(box);
    slider(box, { label: '$\\mu$', min: -2, max: 2, step: 0.1, value: 0, fmt: v => v.toFixed(1) }, v => { const d = v - P.mu; P.mu = v; a += d; b += d; reframe(); upd(); });
    slider(box, { label: '$\\sigma$', min: 0.4, max: 2, step: 0.05, value: 1, fmt: v => v.toFixed(2) }, v => { P.s = v; reframe(); upd(); });
    const bb = buttons(box, [1, 2, 3].map(k => ({ label: `$\\mu\\pm${k === 1 ? '' : k}\\sigma$`, on: () => { a = P.mu - k * P.s; b = P.mu + k * P.s; upd(); } })));
    const lb = el('div'); lb.hidden = true; f.side.appendChild(lb);
    slider(lb, { label: '$\\lambda$', min: 0.3, max: 3, step: 0.1, value: 1, fmt: v => v.toFixed(1) }, v => { P.l = v; reframe(); upd(); });
    const ro = readout(f.side);
    function reframe() { C1.o.x = C2.o.x = xr(); C1.o.y = [0, ymax()]; C1.o.yticks = undefined; }
    function upd() {
      clampAB();
      const pr = cdf(b) - cdf(a);
      ro(`<div class="eqx">${TD(`P(${fmt(a, 2)}\\le X\\le ${fmt(b, 2)})=\\int_{${fmt(a, 2)}}^{${fmt(b, 2)}}f_X(x)\\,dx`)}</div>
        <div class="line">${T(`=F_X(${fmt(b, 2)})-F_X(${fmt(a, 2)})=${fmt(cdf(b), 4)}-${fmt(cdf(a), 4)}=${fmt(pr, 4)}`)}</div>
        ${Math.abs(b - a) < 1e-9 ? `<div class="flag bad">${T('a=b')}: ${T('P(X=x)=0')}, while ${T(`f_X(${fmt(a, 2)})=${fmt(pdf(a), 3)}`)} is not a probability</div>` : ''}`);
      C1.draw(); C2.draw();
    }
    reframe(); upd();
    return { reset() { kind = 'normal'; ks.set('normal'); P = { mu: 0, s: 1, l: 1, a: -2, b: 2 }; a = -1; b = 1; box.hidden = false; lb.hidden = true; reframe(); upd(); }, state: () => ({ kind, a, b, P: cdf(b) - cdf(a) }), set(o) { Object.assign(P, o.P || {}); if (o.kind) kind = o.kind; if (o.a != null) a = o.a; if (o.b != null) b = o.b; reframe(); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.2.2 Mean, deviations, variance, std on A = {9,10,11}, B = {0,10,20}
const moments = {
  title: 'Same mean, different spread: deviations cancel, squares do not',
  mount(f) {
    H(f, 'clamp(300px, 40vw, 360px)');
    const host = el('div'); host.style.cssText = 'position:absolute;inset:0;display:grid;grid-template-rows:1.2fr 1fr'; f.stage.appendChild(host);
    const top = el('div'), bot = el('div'); top.style.position = bot.style.position = 'relative'; host.append(top, bot);
    bot.appendChild(el('div', 'stage-t', `squared deviations ${T('(x-\\mu)^2')}`));
    const A0 = [9, 10, 11], B0 = [0, 10, 20];
    let A = A0.slice(), B = B0.slice();
    const C = new Chart(top, { x: [-2, 22], y: [0, 3], yticks: [[1, 'A'], [2, 'B']], pad: [12, 14, 22, 30], grid: false });
    const Q = new Chart(bot, { x: [-0.5, 6.5], y: [0, 110], xticks: [[1, ''], [5, '']], pad: [24, 14, 10, 40] });
    const sets = [[A, 1, COL.nul, () => A], [B, 2, COL.span, () => B]];
    sets.forEach(([, row, color, get]) => get().forEach((_, i) => C.handle({ get: () => [get()[i], row], set: p => { get()[i] = Math.round(p[0] * 2) / 2; upd(); }, color })));
    C.render = () => {
      for (const [, row, color, get] of sets) {
        const v = get(), m = mean(v);
        C.seg([m, row - 0.45], [m, row + 0.45], color, { dash: [3, 3], lw: 1.5 });
        C.label([m, row + 0.55], `mean = ${fmt(m)}`, color, { align: 'center', font: MONO });
        v.forEach(x => { const d = x - m; if (Math.abs(d) > 1e-9) C.arrow([m, row - 0.22], [x, row - 0.22], d > 0 ? COL.c1 : COL.c2, { lw: 2, head: 7 }); });
      }
    };
    Q.render = () => {
      sets.forEach(([, , color, get], s) => {
        const v = get(), m = mean(v);
        v.forEach((x, i) => { const q = (x - m) ** 2, cx = s * 3.5 + i; Q.rect(cx - 0.35, 0, cx + 0.35, q, color === COL.nul ? 'rgba(255,93,115,.6)' : 'rgba(124,196,255,.6)'); Q.label([cx, q], fmt(q), COL.ink, { dy: -9, align: 'center', font: MONO }); });
        Q.label([s * 3.5 + 1, 0], s ? 'B' : 'A', color, { dy: 10, align: 'center', base: 'top', font: FONT.math });
      });
    };
    const ro = readout(f.side);
    function line(name, v) {
      const m = mean(v), dev = v.map(x => x - m), vr = varPop(v);
      return `<div class="line">${T(`${name}=\\{${v.map(x => fmt(x)).join(',\\ ')}\\},\\ \\mu=${fmt(m, 3)}`)}</div>
        <div class="line">${T(`E[X-\\mu]=\\tfrac{${dev.map(d => fmt(d, 2)).join('+').replace(/\+-/g, '-')}}{3}=${fmt(mean(dev), 6)}`)}</div>
        <div class="line">${T(`\\operatorname{Var}(${name})=${frac(vr)}${frac(vr).startsWith('\\') ? `\\approx${fmt(vr, 3)}` : ''},\\ \\sigma_${name}\\approx${fmt(Math.sqrt(vr), 2)}`)}</div>`;
    }
    function upd() { ro(line('A', A) + line('B', B)); Q.o.y = [0, Math.max(10, ...[...A, ...B].map(() => 0), ...sets.flatMap(([, , , g]) => g().map(x => (x - mean(g())) ** 2))) * 1.2]; C.draw(); Q.draw(); }
    upd();
    return { reset() { A.splice(0, 3, ...A0); B.splice(0, 3, ...B0); upd(); }, state: () => ({ A, B, varA: varPop(A), varB: varPop(B) }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.2.2 Sample mean and sample variance: real i.i.d. draws from the PMF example (mu = 1, sigma^2 = 0.5)
const sampling = {
  title: 'Sample mean and sample variance from real draws',
  mount(f) {
    H(f, 'clamp(300px, 42vw, 370px)');
    const [hl, hr] = split(f, [`${T('\\bar X_n')} over many samples`, `running average of ${T('S^2')}`]);
    const pv = [0.25, 0.5, 0.25], MU = 1, S2 = 0.5;
    let n = 4, R = rng(7), means = [], s2a = [], s2b = [], last = null;
    const draw1 = () => { const u = R(); return u < pv[0] ? 0 : u < pv[0] + pv[1] ? 1 : 2; };
    const H1 = new Chart(hl, { x: [0, 2], y: [0, 1], xlabel: 'x̄', pad: [26, 12, 26, 40] });
    const H2 = new Chart(hr, { x: [0, 1], y: [0, 0.9], xlabel: 'samples', pad: [26, 12, 26, 40] });
    H1.render = () => {
      if (!means.length) return;
      const step = 1 / n, bins = new Map();
      for (const m of means) { const k = Math.round(m / step); bins.set(k, (bins.get(k) || 0) + 1); }
      const mx = Math.max(...bins.values()); H1.o.y = [0, mx / means.length * 1.15];
      for (const [k, c] of bins) H1.rect(k * step - step * 0.4, 0, k * step + step * 0.4, c / means.length, 'rgba(178,140,255,.65)');
      H1.seg([MU, 0], [MU, H1.o.y[1]], COL.eig, { dash: [4, 4] }); H1.label([MU, H1.o.y[1]], 'μ = 1', COL.eig, { dy: 8, dx: 6, base: 'top', font: MONO });
    };
    H2.render = () => {
      const N = s2a.length; if (!N) return; H2.o.x = [0, Math.max(10, N)];
      H2.seg([0, S2], [H2.o.x[1], S2], COL.eig, { dash: [4, 4] }); H2.label([0, S2], 'σ² = 0.5', COL.eig, { dx: 6, dy: -9, font: MONO });
      const run = arr => { let s = 0; const k = Math.max(1, Math.ceil(arr.length / 400)); const out = []; arr.forEach((v, i) => { s += v; if (i % k === 0 || i === arr.length - 1) out.push([i + 1, s / (i + 1)]); }); return out; };
      H2.path(run(s2a), COL.c1, { lw: 2 }); H2.path(run(s2b), COL.c2, { lw: 2 });
    };
    const nS = slider(f.side, { label: '$n$', min: 2, max: 30, step: 1, value: n, fmt: v => String(v) }, v => { n = v; clear(); });
    buttons(f.side, [{ label: 'Draw 1 sample', on: () => add(1) }, { label: 'Draw 1,000', pri: true, on: () => add(1000) }, { label: 'Clear', on: () => clear() }]);
    legend(f.side, [[COL.c1, 'dividing by $n-1$'], [COL.c2, 'dividing by $n$']]);
    const ro = readout(f.side);
    function add(k) {
      for (let j = 0; j < k; j++) {
        const xs = Array.from({ length: n }, draw1); last = xs;
        means.push(mean(xs)); s2a.push(varSample(xs)); s2b.push(varPop(xs));
      }
      upd();
    }
    function clear() { R = rng(7); means = []; s2a = []; s2b = []; last = null; upd(); }
    function upd() {
      const N = means.length;
      ro(N ? `<div class="line">last sample ${T(`(${last.join(',')})`)}: ${T(`\\bar X=${fmt(mean(last), 3)},\\ S^2=${fmt(varSample(last), 3)}`)}</div>
        <div class="line">${N} samples of size ${T(`n=${n}`)}:</div>
        <div class="line">average of ${T(`\\bar X_n`)}: ${T(fmt(mean(means), 3))} ${T('(\\mu=1)')}</div>
        <div class="line">variance of ${T(`\\bar X_n`)}: ${T(fmt(varPop(means), 4))}, ${T(`\\sigma^2/n=${fmt(S2 / n, 4)}`)}</div>
        <div class="line">average ${T('S_n^2')} (÷${T('n-1')}): ${T(fmt(mean(s2a), 4))}; ÷${T('n')}: ${T(fmt(mean(s2b), 4))}</div>`
        : `<div class="line">population: ${T('p_X=(\\tfrac14,\\tfrac12,\\tfrac14)')} on ${T('0,1,2')}, ${T('\\mu=1,\\ \\sigma^2=0.5')}</div>`);
      H1.draw(); H2.draw();
    }
    upd();
    return { reset() { n = 4; nS.set(4); clear(); }, add, state: () => ({ n, means, s2a, s2b }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.2.3 Joint, marginal, conditional on the 80-student table
const students = {
  title: 'Joint, marginal and conditional on the 80-student table',
  mount(f) {
    f.body.classList.add('wide'); f.side.remove(); f.stage.style.minHeight = '0';
    const N = [[36, 14], [9, 21]], R = ['Passed', 'Did not pass'], Cn = ['Studied', 'Did not study'];
    const rowT = N.map(r => r[0] + r[1]), colT = [N[0][0] + N[1][0], N[0][1] + N[1][1]], tot = 80;
    let sel = { k: 'cond-col', i: 0, j: 0 };
    const wrap = el('div'); wrap.style.cssText = 'display:grid;grid-template-columns:minmax(0,auto) minmax(0,1fr);gap:18px;padding:14px;align-items:start'; f.stage.appendChild(wrap);
    const tbl = el('div'), out = el('div'); out.className = 'ro'; wrap.append(tbl, out);
    const mq = matchMedia('(max-width: 700px)'), lay = () => { wrap.style.gridTemplateColumns = mq.matches ? 'minmax(0,1fr)' : 'minmax(0,auto) minmax(0,1fr)'; }; lay(); mq.addEventListener('change', lay);
    const btns = el('div', 'seg'); btns.style.marginTop = '10px'; tbl.after(btns);
    const conds = [['cond-col', 0, 'given Studied'], ['cond-col', 1, 'given Did not study'], ['cond-row', 0, 'given Passed'], ['cond-row', 1, 'given Did not pass']];
    conds.forEach(([k, i, t]) => { const b = el('button', '', t); b.type = 'button'; b.onclick = () => { sel = { k, i }; upd(); }; btns.appendChild(b); });
    tbl.appendChild(btns);
    const cellStyle = 'padding:8px 12px;border:1px solid var(--line);text-align:center;cursor:pointer;font:500 14px var(--mono);min-width:64px';
    function upd() {
      const hi = (i, j) => (sel.k === 'joint' && sel.i === i && sel.j === j) || (sel.k === 'cond-row' && sel.i === i) || (sel.k === 'cond-col' && sel.j === undefined && sel.i === j) || (sel.k === 'cond-col' && sel.i === j);
      const bg = (on, c) => on ? `background:${c}` : '';
      tbl.querySelector('table')?.remove();
      const t = el('table'); t.style.cssText = 'border-collapse:collapse;color:var(--ink)';
      t.innerHTML = `<tr><td style="${cellStyle};border:0;cursor:default"></td>${Cn.map((c, j) => `<th style="${cellStyle};font:600 12.5px var(--display);color:var(--muted)">${c}</th>`).join('')}<th style="${cellStyle};font:600 12.5px var(--display);color:var(--muted)">Total</th></tr>` +
        R.map((r, i) => `<tr><th style="${cellStyle};font:600 12.5px var(--display);color:var(--muted);text-align:left">${r}</th>${N[i].map((v, j) => `<td data-k="joint" data-i="${i}" data-j="${j}" style="${cellStyle};${bg(hi(i, j), 'rgba(47,183,166,.3)')}">${v}</td>`).join('')}<td data-k="mrow" data-i="${i}" style="${cellStyle};color:var(--blue);${bg(sel.k === 'mrow' && sel.i === i, 'rgba(108,182,255,.2)')}">${rowT[i]}</td></tr>`).join('') +
        `<tr><th style="${cellStyle};font:600 12.5px var(--display);color:var(--muted);text-align:left">Total</th>${colT.map((v, j) => `<td data-k="mcol" data-j="${j}" style="${cellStyle};color:var(--blue);${bg(sel.k === 'mcol' && sel.j === j, 'rgba(108,182,255,.2)')}">${v}</td>`).join('')}<td style="${cellStyle};cursor:default">${tot}</td></tr>`;
      t.querySelectorAll('td[data-k]').forEach(td => td.onclick = () => { sel = { k: td.dataset.k, i: +td.dataset.i, j: +td.dataset.j }; upd(); });
      tbl.prepend(t);
      [...btns.children].forEach((b, n) => b.setAttribute('aria-pressed', String(sel.k === conds[n][0] && sel.i === conds[n][1])));
      let html = '';
      if (sel.k === 'joint') html = `<div class="line"><b style="color:#5fd4c4">Joint:</b> ${T(`P(\\text{${R[sel.i].toLowerCase()}}\\ \\&\\ \\text{${Cn[sel.j].toLowerCase()}})=\\dfrac{${N[sel.i][sel.j]}}{80}=${fmt(N[sel.i][sel.j] / 80, 4)}`)}</div><div class="line">it lives <b>inside</b> the cell</div>`;
      else if (sel.k === 'mrow') html = `<div class="line"><b style="color:var(--blue)">Marginal:</b> ${T(`P(\\text{${R[sel.i].toLowerCase()}})=\\dfrac{${N[sel.i][0]}}{80}+\\dfrac{${N[sel.i][1]}}{80}=\\dfrac{${rowT[sel.i]}}{80}=${fmt(rowT[sel.i] / 80, 4)}`)}</div><div class="line">sum out whether they studied: the row total</div>`;
      else if (sel.k === 'mcol') html = `<div class="line"><b style="color:var(--blue)">Marginal:</b> ${T(`P(\\text{${Cn[sel.j].toLowerCase()}})=\\dfrac{${N[0][sel.j]}}{80}+\\dfrac{${N[1][sel.j]}}{80}=\\dfrac{${colT[sel.j]}}{80}=${fmt(colT[sel.j] / 80, 4)}`)}</div><div class="line">sum out whether they passed: the column total</div>`;
      else {
        const byCol = sel.k === 'cond-col', vals = byCol ? [N[0][sel.i], N[1][sel.i]] : N[sel.i], den = byCol ? colT[sel.i] : rowT[sel.i];
        const names = byCol ? ['passed', 'failed'] : ['studied', 'did not study'], given = (byCol ? Cn : R)[sel.i].toLowerCase();
        html = `<div class="line"><b style="color:var(--red)">Conditional</b> on ${given}: divide each cell by ${T(String(den))}</div>` +
          vals.map((v, n) => `<div style="display:grid;grid-template-columns:150px minmax(0,1fr) 120px;gap:8px;align-items:center;font-size:13px"><span>${T(`P(\\text{${names[n]}}\\mid\\text{${given}})`)}</span><span style="height:12px;border-radius:3px;background:linear-gradient(90deg,var(--red) ${(v / den * 100).toFixed(1)}%,var(--line-soft) 0)"></span><span>${T(`\\tfrac{${v}}{${den}}=${fmt(v / den, 4)}`)}</span></div>`).join('') +
          `<div class="flag ok">${T(`\\tfrac{${vals[0]}}{${den}}+\\tfrac{${vals[1]}}{${den}}=1`)}: a new distribution</div>`;
      }
      out.innerHTML = html;
    }
    upd();
    return { reset() { sel = { k: 'cond-col', i: 0 }; upd(); }, select(s) { sel = s; upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.2.4 Bayes: 10,000 people
const bayes = {
  title: 'Bayes in action: where the positives come from',
  mount(f) {
    H(f, 'clamp(300px, 44vw, 380px)');
    const c = el('canvas'); c.style.cssText = 'position:absolute;inset:0;width:100%;height:100%'; f.stage.appendChild(c);
    let pD = 0.01, sens = 0.95, fpr = 0.05;
    const paint = () => {
      const r = f.stage.getBoundingClientRect(), d = Math.min(2, devicePixelRatio || 1); c.width = r.width * d; c.height = r.height * d;
      const g = c.getContext('2d'); g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, r.width, r.height);
      const N = 100, pad = 14, size = Math.min((r.width - 2 * pad) / N, (r.height - 2 * pad - 20) / N), x0 = (r.width - size * N) / 2, y0 = pad;
      const sick = Math.round(pD * 10000), sp = Math.round(sens * sick), hp = Math.round(fpr * (10000 - sick));
      // sick people fill the first columns (column-major), healthy positives fill from the other end
      for (let k = 0; k < 10000; k++) {
        const col = Math.floor(k / N), row = k % N; let fill;
        if (k < sick) fill = k < sp ? '#ff6b6b' : 'rgba(255,107,107,.3)';
        else fill = k >= 10000 - hp ? '#6cb6ff' : 'rgba(232,236,248,.07)';
        g.fillStyle = fill; g.fillRect(x0 + col * size, y0 + row * size, Math.max(1, size - 0.6), Math.max(1, size - 0.6));
      }
      g.font = '600 11.5px "JetBrains Mono", monospace'; g.fillStyle = COL.muted; g.textAlign = 'center';
      g.fillText('each square: one person; 10,000 in all', r.width / 2, r.height - 8);
    };
    new ResizeObserver(() => requestAnimationFrame(paint)).observe(f.stage);
    const sl = [
      slider(f.side, { label: '$P(D)$', min: 0.001, max: 0.3, step: 0.001, value: pD, fmt: v => v.toFixed(3) }, v => { pD = v; upd(); }),
      slider(f.side, { label: '$P(+\\mid D)$', min: 0.5, max: 1, step: 0.01, value: sens, fmt: v => v.toFixed(2) }, v => { sens = v; upd(); }),
      slider(f.side, { label: '$P(+\\mid\\bar D)$', min: 0, max: 0.3, step: 0.005, value: fpr, fmt: v => v.toFixed(3) }, v => { fpr = v; upd(); }),
    ];
    legend(f.side, [['#ff6b6b', 'sick, test $+$'], ['#6cb6ff', 'healthy, test $+$'], ['rgba(255,107,107,.45)', 'sick, test $-$']]);
    const ro = readout(f.side);
    function upd() {
      const pp = sens * pD + fpr * (1 - pD), post = sens * pD / pp, sick = Math.round(pD * 10000);
      ro(`<div class="eqx">${TD(`P(+)=\\underbrace{${fmt(sens)}(${fmt(pD, 3)})}_{\\text{sick}}+\\underbrace{${fmt(fpr, 3)}(${fmt(1 - pD, 3)})}_{\\text{healthy}}=${fmt(pp, 4)}`)}</div>
        <div class="eqx">${TD(`P(D\\mid+)=\\frac{${fmt(sens)}(${fmt(pD, 3)})}{${fmt(pp, 4)}}\\approx${fmt(post, 3)}`)}</div>
        <div class="line">of 10,000: ${Math.round(sens * sick)} sick positives and ${Math.round(fpr * (10000 - sick))} healthy positives</div>`);
      paint();
    }
    upd();
    return { reset() { pD = 0.01; sens = 0.95; fpr = 0.05; sl[0].set(pD); sl[1].set(sens); sl[2].set(fpr); upd(); }, state: () => { const pp = sens * pD + fpr * (1 - pD); return { pp, post: sens * pD / pp }; } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.2.4 / 1.2.5 Joint density sliced: bivariate normal with standard marginals and correlation rho
const STOPS = [[0x1b, 0x1f, 0x4b], [0x24, 0x4f, 0x86], [0x2a, 0x9c, 0x96], [0x96, 0xd0, 0x78], [0xf4, 0xe8, 0xa4]];
const ramp = t => { t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1); const i = Math.min(STOPS.length - 2, Math.floor(t)), u = t - i, a = STOPS[i], b = STOPS[i + 1]; return [0, 1, 2].map(k => (a[k] + (b[k] - a[k]) * u) / 255); };
export const fxy = (x, y, r) => Math.exp(-(x * x - 2 * r * x * y + y * y) / (2 * (1 - r * r))) / (2 * Math.PI * Math.sqrt(1 - r * r));
const jointDensity = {
  title: 'A joint density, its slices and its marginal',
  async mount(f) {
    H(f, 'clamp(320px, 50vw, 430px)');
    const THREE = await loadThree();
    const [hl, hr] = split(f, [`${T('f_{X,Y}(x,y)')}`, `slice at ${T('y')}`]);
    let r = 0.6, y0 = 0.8;
    const S = new Scene3(hl, { cam: [7.6, -8.8, 6.2], target: [0, 0, 0.4], minDist: 4 });
    const HZ = 9, R = 3, N = 70;
    const geo = new THREE.BufferGeometry(), pos = new Float32Array((N + 1) ** 2 * 3), col = new Float32Array((N + 1) ** 2 * 3), idx = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const a = i * (N + 1) + j; idx.push(a, a + N + 1, a + 1, a + N + 1, a + N + 2, a + 1); }
    geo.setIndex(idx); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    S.scene.add(new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide, shininess: 20, transparent: true, opacity: 0.9 })));
    S.axes(3.3);
    for (const [p, t] of [[[3.6, 0, 0], 'x'], [[0, 3.6, 0], 'y']]) { const l = S.label(t); l.p = p; l.e.style.color = COL.muted; }
    const cut = S.line([[0, 0, 0]], 0xff6b6b), plane = new THREE.Mesh(new THREE.PlaneGeometry(2 * R, 1.8), new THREE.MeshBasicMaterial({ color: 0xff6b6b, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }));
    S.scene.add(plane);
    const C = new Chart(hr, { x: [-3, 3], y: [0, 0.8], xlabel: 'x', pad: [26, 12, 26, 40] });
    const fY = y => normPdf(y), fX = x => normPdf(x);
    C.render = () => {
      C.fn(fX, 'rgba(232,236,248,.45)', { lw: 1.6, dash: [5, 4] });
      C.fn(x => fxy(x, y0, r), COL.nul, { lw: 2, dash: [2, 3] });
      C.fn(x => fxy(x, y0, r) / fY(y0), COL.nul, { lw: 2.6 });
    };
    const build = () => {
      for (let i = 0; i <= N; i++) for (let j = 0; j <= N; j++) {
        const x = -R + 2 * R * j / N, y = -R + 2 * R * i / N, z = fxy(x, y, r), k = (i * (N + 1) + j) * 3;
        pos[k] = x; pos[k + 1] = z * HZ / 1.6; pos[k + 2] = -y; const c = ramp(z / 0.3); col[k] = c[0]; col[k + 1] = c[1]; col[k + 2] = c[2];
      }
      geo.attributes.position.needsUpdate = true; geo.attributes.color.needsUpdate = true; geo.computeVertexNormals();
      S.setLine(cut, Array.from({ length: 121 }, (_, i) => { const x = -R + 2 * R * i / 120; return [x, y0, fxy(x, y0, r) * HZ / 1.6 + 0.01]; }));
      plane.position.set(0, 0.9, -y0);
      S.draw(); C.draw();
    };
    const rS = slider(f.side, { label: '$\\rho$', min: -0.9, max: 0.9, step: 0.05, value: r, fmt: v => v.toFixed(2) }, v => { r = v; upd(); });
    const yS = slider(f.side, { label: 'slice $y$', min: -2.5, max: 2.5, step: 0.05, value: y0, fmt: v => v.toFixed(2) }, v => { y0 = v; upd(); });
    legend(f.side, [[COL.nul, '$f_{X\\mid Y}(x\\mid y)=f_{X,Y}(x,y)/f_Y(y)$'], ['rgba(255,107,107,.6)', 'the raw cut $f_{X,Y}(x,y)$ (dotted)'], ['rgba(232,236,248,.6)', 'marginal $f_X(x)$ (dashed)']]);
    const ro = readout(f.side);
    function upd() {
      let area = 0; for (let i = 0; i < 2000; i++) { const x = -8 + 16 * (i + 0.5) / 2000; area += fxy(x, y0, r) / fY(y0) * 16 / 2000; }
      ro(`<div class="line">${T(`f_Y(${fmt(y0, 2)})=\\int f_{X,Y}(x,${fmt(y0, 2)})\\,dx=${fmt(fY(y0), 4)}`)}</div>
        <div class="line">${T(`\\int f_{X\\mid Y}(x\\mid ${fmt(y0, 2)})\\,dx=${fmt(area, 4)}`)}: a new distribution in ${T('x')}</div>
        ${Math.abs(r) < 1e-9 ? `<div class="flag ok">${T('\\rho=0')}: ${T('f_{X,Y}(x,y)=f_X(x)\\cdot f_Y(y)')}, every slice has the shape of ${T('f_X')}</div>` : `<div class="line">the slice shape depends on ${T('y')}: ${T('X')} and ${T('Y')} are not independent</div>`}`);
      build();
    }
    upd();
    return { reset() { r = 0.6; y0 = 0.8; rS.set(r); yS.set(y0); upd(); }, state: () => ({ r, y0 }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.2.5–1.2.7 A 2x2 joint table: marginals, independence test, E[XY], Cov, rho
export function tableStats(xv, yv, P) {
  const px = P.map(r => r[0] + r[1]), py = [P[0][0] + P[1][0], P[0][1] + P[1][1]];
  const EX = xv[0] * px[0] + xv[1] * px[1], EY = yv[0] * py[0] + yv[1] * py[1];
  let EXY = 0; for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) EXY += xv[i] * yv[j] * P[i][j];
  const VX = xv[0] ** 2 * px[0] + xv[1] ** 2 * px[1] - EX ** 2, VY = yv[0] ** 2 * py[0] + yv[1] ** 2 * py[1] - EY ** 2, cov = EXY - EX * EY;
  return { px, py, EX, EY, EXY, VX, VY, cov, rho: cov / Math.sqrt(VX * VY) };
}
const tableLab = {
  title: 'One table, every quantity: marginals, independence, $E[XY]$, Cov, $\\rho$',
  mount(f) {
    f.stage.style.minHeight = '0';
    const PRE = [
      { label: '$E[XY]$ example', xv: [1, 2], yv: [1, 2], P: [[0.2, 0.3], [0.1, 0.4]] },
      { label: 'independence example', xv: [0, 1], yv: [0, 1], P: [[0.15, 0.15], [0.35, 0.35]] },
    ];
    let k = 0, xv, yv, P;
    const load = i => { k = i; ({ xv, yv } = PRE[i]); P = PRE[i].P.map(r => r.slice()); };
    load(0);
    const box = el('div'); box.style.cssText = 'padding:14px;display:grid;gap:10px;justify-items:center'; f.stage.appendChild(box);
    const setCell = (i, j, v) => {
      v = Math.max(0, Math.min(1, v)); const rest = 1 - P[i][j], others = [[0, 0], [0, 1], [1, 0], [1, 1]].filter(([a, b]) => a !== i || b !== j);
      others.forEach(([a, b]) => { P[a][b] = rest > 1e-12 ? P[a][b] / rest * (1 - v) : (1 - v) / 3; }); P[i][j] = v; upd();
    };
    const pick = seg(f.side, PRE.map((p, i) => ({ label: p.label, value: i })), i => { load(i); upd(); }, 0);
    const ro = readout(f.side);
    function upd() {
      const s = tableStats(xv, yv, P), cs = 'padding:6px 10px;border:1px solid var(--line);text-align:center;font:500 13px var(--mono)';
      box.innerHTML = `<table style="border-collapse:collapse;color:var(--ink)"><tr><th style="${cs}">${T('X\\backslash Y')}</th>${yv.map(y => `<th style="${cs}">${T(`Y=${y}`)}</th>`).join('')}<th style="${cs};color:var(--blue)">${T('P(X)')}</th></tr>` +
        xv.map((x, i) => `<tr><th style="${cs}">${T(`X=${x}`)}</th>${yv.map((_, j) => { const ind = Math.abs(P[i][j] - s.px[i] * s.py[j]) < 5e-4; return `<td style="${cs}"><div>${fmt(P[i][j], 3)}</div><input type="range" min="0" max="1" step="0.01" value="${P[i][j]}" data-i="${i}" data-j="${j}" style="width:92px;accent-color:var(--eig)"><div style="font-size:11px;color:${ind ? 'var(--c1)' : 'var(--null)'}">${T(`P(X)P(Y)=${fmt(s.px[i] * s.py[j], 3)}`)} ${ind ? '✓' : '✗'}</div></td>`; }).join('')}<td style="${cs};color:var(--blue)">${fmt(s.px[i], 3)}</td></tr>`).join('') +
        `<tr><th style="${cs};color:var(--blue)">${T('P(Y)')}</th>${s.py.map(v => `<td style="${cs};color:var(--blue)">${fmt(v, 3)}</td>`).join('')}<td style="${cs}">1</td></tr></table>`;
      box.querySelectorAll('input').forEach(inp => inp.addEventListener('change', () => setCell(+inp.dataset.i, +inp.dataset.j, +inp.value)));
      const indep = [0, 1].every(i => [0, 1].every(j => Math.abs(P[i][j] - s.px[i] * s.py[j]) < 5e-4));
      ro(`<div class="flag ${indep ? 'ok' : 'bad'}">${indep ? 'every cell matches: independent' : 'a cell fails: not independent'}</div>
        <div class="line">${T(`E[X]=${fmt(s.EX, 4)},\\ E[Y]=${fmt(s.EY, 4)}`)}</div>
        <div class="line">${T(`E[XY]=${fmt(s.EXY, 4)}`)}</div>
        <div class="line">${T(`\\operatorname{Cov}=E[XY]-E[X]E[Y]=${fmt(s.cov, 4)}`)}</div>
        <div class="line">${T(`\\sigma_X=${fmt(Math.sqrt(s.VX), 3)},\\ \\sigma_Y=${fmt(Math.sqrt(s.VY), 3)}`)}</div>
        <div class="line">${T(`\\rho_{X,Y}=${isFinite(s.rho) ? fmt(s.rho, 3) : '\\text{undefined}'}`)}</div>`);
    }
    upd();
    return { reset() { load(0); pick.set(0); upd(); }, load(i) { load(i); pick.set(i); upd(); }, state: () => tableStats(xv, yv, P) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.2.7 Covariance as signed rectangles on a scatter
const covScatter = {
  title: 'Covariance: are both above (or both below) their means?',
  mount(f) {
    H(f, 'clamp(300px, 46vw, 400px)');
    const R0 = rng(3), base = Array.from({ length: 12 }, () => { const x = (R0() - 0.5) * 5; return [x, 0.7 * x + (R0() - 0.5) * 2.6]; });
    const PRE = { cloud: base, line: [-2, -1, 0, 1, 2].map(a => [a, 2 * a]), square: [-2, -1, 0, 1, 2].map(a => [a, a * a]) };
    let pts = base.map(p => p.slice());
    const C = new Chart(f.stage, { x: [-4.5, 4.5], y: [-4.5, 4.5], xlabel: 'X', ylabel: 'Y', pad: [14, 14, 26, 34] });
    const st = () => { const mx = mean(pts.map(p => p[0])), my = mean(pts.map(p => p[1])); const cov = mean(pts.map(([x, y]) => (x - mx) * (y - my))); const sx = Math.sqrt(varPop(pts.map(p => p[0]))), sy = Math.sqrt(varPop(pts.map(p => p[1]))); return { mx, my, cov, sx, sy, rho: cov / (sx * sy) }; };
    const reh = () => { C.handles = []; pts.forEach((_, i) => C.handle({ get: () => pts[i], set: q => { pts[i] = q; upd(); }, color: COL.ink, ring: false })); };
    C.render = () => {
      const { mx, my } = st();
      for (const [x, y] of pts) { const s = (x - mx) * (y - my); C.rect(mx, my, x, y, s >= 0 ? 'rgba(124,196,255,.13)' : 'rgba(255,122,89,.16)', s >= 0 ? 'rgba(124,196,255,.45)' : 'rgba(255,122,89,.5)'); }
      C.seg([mx, -4.5], [mx, 4.5], COL.eig, { dash: [4, 4] }); C.seg([-4.5, my], [4.5, my], COL.eig, { dash: [4, 4] });
      C.label([mx, 4.5], 'μX', COL.eig, { dx: 4, dy: 8, base: 'top', font: MONO }); C.label([4.5, my], 'μY', COL.eig, { dx: -22, dy: -8, font: MONO });
      for (const p of pts) C.dot(p, 5.5, COL.ink);
    };
    const pick = seg(f.side, [{ label: 'a cloud', value: 'cloud' }, { label: '$\\rho=1$ line', value: 'line' }, { label: '$B=A^2$ (§2.2.1)', value: 'square' }], v => { pts = PRE[v].map(p => p.slice()); C.o.y = v === 'square' ? [-1, 5] : [-4.5, 4.5]; reh(); upd(); }, 'cloud');
    const ro = readout(f.side);
    function upd() {
      const s = st();
      ro(`<div class="line">${T(`\\operatorname{Cov}(X,Y)=E[(X-\\mu_X)(Y-\\mu_Y)]=${fmt(s.cov, 3)}`)}</div><div class="line">${T(`\\sigma_X=${fmt(s.sx, 3)},\\ \\sigma_Y=${fmt(s.sy, 3)}`)}</div>
        <div class="line">${T(`\\rho_{X,Y}=${fmt(s.rho, 3)}`)}</div>
        <div class="line">${s.rho > 0.999 ? T('\\rho=1') + ': perfect linear correlation' : Math.abs(s.rho) < 1e-9 ? T('\\rho=0') + ': no linear relationship (even when ' + T('Y') + ' depends on ' + T('X') + ')' : s.rho > 0 ? T('\\rho>0') + ': increase together' : T('\\rho<0') + ': one up, other down'}</div>`);
      C.draw();
    }
    reh(); upd();
    return { reset() { pts = base.map(p => p.slice()); pick.set('cloud'); C.o.y = [-4.5, 4.5]; reh(); upd(); }, state: st, set(name) { pick.set(name); pts = PRE[name].map(p => p.slice()); reh(); upd(); } };
  },
};

export default {
  'pmf-cdf': pmfCdf,
  'pdf-area': pdfArea,
  'moments': moments,
  'sampling': sampling,
  'students': students,
  'bayes': bayes,
  'joint-density': jointDensity,
  'table-lab': tableLab,
  'cov-scatter': covScatter,
};
