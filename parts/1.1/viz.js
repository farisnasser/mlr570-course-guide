// Part 1.1 Linear Algebra — interactive visuals. Every number shown is computed live from the matrices here;
// the notes' worked examples are the defaults and presets (recomputed in scripts/check/1_1.py).
import { Plane, COL, FONT, tween } from '../../src/viz/plane.js';
import * as L from '../../src/viz/la.js';
import { el, T, TD, fmt, matTex, vecTex, seg, slider, buttons, readout, legend } from '../../src/viz/ui.js';
import { loadThree, Scene3 } from '../../src/viz/three-lite.js';

const S2 = Math.SQRT1_2;
const Q45 = [[S2, S2], [-S2, S2]];                 // the notes' Q = (1/sqrt2)[[1,1],[-1,1]]
const mt = M => `$\\left[\\begin{smallmatrix}${M.map(r => r.join('&')).join('\\\\')}\\end{smallmatrix}\\right]$`;
const pn = v => (v < 0 ? `(${fmt(v)})` : fmt(v));
const H = (f, h) => { f.stage.style.height = h; };
const STAGE_H = 'clamp(300px, 52vw, 430px)';
const deg = r => r * 180 / Math.PI;
const nice = (M, d = 2) => M.map(r => r.map(v => +fmt(v, d)));

/** fit the view so all points (and the unit square) are visible */
function fitExtent(pts, min = 2.6) { return Math.max(min, ...pts.flatMap(p => p.map(Math.abs))) * 1.18; }

// ---------------------------------------------------------------------------------------------------------------
// 1.1.1 Matrix as a linear transformation of the grid; det = area; det = 0 collapse
const gridTransform = {
  title: 'A matrix is a linear transformation of the grid',
  mount(f) {
    H(f, STAGE_H);
    const P = new Plane(f.stage, { extent: 3.2 });
    const PRESETS = [
      { label: '$I$', M: [[1, 0], [0, 1]] },
      { label: mt([[4, 7], [2, 6]]), M: [[4, 7], [2, 6]] },
      { label: mt([[2, 4], [1, 2]]), M: [[2, 4], [1, 2]] },
      { label: mt([[1, 2], [2, 4]]), M: [[1, 2], [2, 4]] },
      { label: mt([[2, 1], [1, 2]]), M: [[2, 1], [1, 2]] },
      { label: '$Q$', M: Q45 },
    ];
    let A = [[2, 1], [1, 2]].map(r => r.slice()), t = 1, stopT = null;
    const M = () => L.lerpM(L.I2(), A, t);
    const setExtent = (ext, ms = 500) => { const e0 = P.o.extent; tween(ms, u => { P.o.extent = e0 + (ext - e0) * u; P.draw(); }); };
    const fit = () => setExtent(fitExtent([[A[0][0], A[1][0]], [A[0][1], A[1][1]], [A[0][0] + A[0][1], A[1][0] + A[1][1]]]));
    const col = j => () => { const m = M(); return [m[0][j], m[1][j]]; };
    const setCol = j => p => { if (t < 1) { t = 1; tS.set(1); } A[0][j] = p[0]; A[1][j] = p[1]; pick.set(null); upd(); };
    P.handle({ get: col(0), set: setCol(0), color: COL.c1, snap: 0.5 });
    P.handle({ get: col(1), set: setCol(1), color: COL.c2, snap: 0.5 });

    P.render = (g) => {
      const m = M(), d = L.det2(m), a1 = [m[0][0], m[1][0]], a2 = [m[0][1], m[1][1]];
      const sing = Math.abs(d) < 1e-9;
      P.gridT(m, { color: 'rgba(124,196,255,.34)' });
      if (!sing) P.poly([[0, 0], a1, L.add(a1, a2), a2], d > 0 ? 'rgba(95,224,176,.16)' : 'rgba(255,122,89,.18)', d > 0 ? 'rgba(95,224,176,.7)' : 'rgba(255,122,89,.75)');
      if (sing && t === 1) {
        const c = L.norm(a1) > 1e-9 ? a1 : a2;
        if (L.norm(c) > 1e-9) { P.lineDir([0, 0], c, COL.span, { lw: 2.5 }); P.label(L.scale(L.unit(c), P.o.extent * 0.82), 'Col(A)', COL.span, { dy: -14, align: 'center' }); }
        // null space of a rank-1 A: perpendicular to its (proportional) rows
        const row = L.norm(A[0]) > 1e-9 ? A[0] : A[1], nul = L.norm(row) > 1e-9 ? L.unit([-row[1], row[0]]) : null;
        if (nul) {
          P.lineDir([0, 0], nul, COL.nul, { lw: 2, dash: [7, 6], alpha: .9 });
          const q = L.scale(nul[0] < 0 ? nul : L.scale(nul, -1), P.o.extent * 0.7);
          P.label(q, 'Null(A)', COL.nul, { dy: 14, align: 'center' });
        }
      }
      P.arrow([0, 0], a1, COL.c1, { lw: 3.2 });
      P.arrow([0, 0], a2, COL.c2, { lw: 3.2 });
      P.tipLabel(a1, t === 0 ? 'e₁' : `(${fmt(a1[0], 2)}, ${fmt(a1[1], 2)})`, COL.c1, { gap: 34, font: '600 12.5px "JetBrains Mono", monospace' });
      P.tipLabel(a2, t === 0 ? 'e₂' : `(${fmt(a2[0], 2)}, ${fmt(a2[1], 2)})`, COL.c2, { gap: 34, font: '600 12.5px "JetBrains Mono", monospace' });
    };

    const ctl = el('div', ''); f.side.appendChild(ctl);
    ctl.appendChild(el('h5', '', 'Matrices from the notes'));
    const pick = seg(ctl, PRESETS.map((p, i) => ({ label: p.label, value: i })), i => { A = PRESETS[i].M.map(r => r.slice()); play(); fit(); }, 4);
    const tS = slider(f.side, { label: 'apply $A$', min: 0, max: 1, step: 0.01, value: 1, fmt: v => v.toFixed(2) }, v => { stopT?.(); t = v; upd(); });
    buttons(f.side, [{ label: '▶ Play', pri: true, on: () => play() }]);
    const ro = readout(f.side);
    function play() { stopT?.(); stopT = tween(1400, u => { t = u; tS.set(u); upd(); }); }
    function upd() {
      const d = L.det2(A), sing = Math.abs(d) < 1e-9, inv = L.inv2(A), r = L.rank(A);
      ro(`<div class="eqx">${TD(`A=${matTex(nice(A))}`)}</div>
        <div class="line">${T(`\\det A=ad-bc=${pn(A[0][0])}\\cdot${pn(A[1][1])}-${pn(A[0][1])}\\cdot${pn(A[1][0])}=${fmt(d)}`)}</div>
        ${sing ? `<div class="flag bad">${T('\\det A=0')}: the plane collapses onto a line. A dimension was lost, so ${T('A')} is not invertible.</div>
          <div class="line">rank ${T(`=${r}`)}</div>`
        : `<div class="line">area of the square ${T('\\times')} ${T(`|\\det A|=${fmt(Math.abs(d))}`)}${d < 0 ? ` <span class="flag" style="color:var(--c2)">orientation flipped (${T('\\det A<0')})</span>` : ''}</div>
          <div class="eqx">${TD(`A^{-1}=\\frac{1}{${fmt(d)}}${matTex(nice([[A[1][1], -A[0][1]], [-A[1][0], A[0][0]]]))}`)}</div>`}`);
      P.draw();
    }
    upd();
    return { reset() { A = [[2, 1], [1, 2]]; pick.set(4); t = 1; tS.set(1); fit(); upd(); }, state: () => ({ A, t, det: L.det2(A) }), set(M) { A = M.map(r => r.slice()); t = 1; upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.1.1 AB = apply B first; order matters
const chain = {
  title: 'Order matters: $AB$ versus $BA$',
  mount(f) {
    H(f, 'clamp(260px, 40vw, 360px)');
    const sp = el('div', 'split'); f.stage.appendChild(sp);
    const hl = el('div'), hr = el('div'); sp.append(hl, hr);
    hl.appendChild(el('div', 'stage-t', `${T('AB')}: stretch ${T('B')} first, then rotate ${T('A')}`));
    hr.appendChild(el('div', 'stage-t', `${T('BA')}: rotate ${T('A')} first, then stretch ${T('B')}`));
    const Pl = new Plane(hl, { extent: 2.7 }), Pr = new Plane(hr, { extent: 2.7 });
    let s = 2, th = 45, u = 2, stopT = null;
    const B = () => [[s, 0], [0, 1]], A = () => L.rot(th * Math.PI / 180);
    const stageM = (first, second, k) => k <= 1 ? first(k) : L.mm(second(k - 1), first(1));
    const Bs = k => [[1 + (s - 1) * k, 0], [0, 1]], As = k => L.rot(th * Math.PI / 180 * k);
    const draw = (P, M, other) => {
      P.gridT(M, { color: 'rgba(124,196,255,.34)' });
      if (u >= 2) P.poly([[0, 0], [other[0][0], other[1][0]], [other[0][0] + other[0][1], other[1][0] + other[1][1]], [other[0][1], other[1][1]]], null, 'rgba(232,236,248,.45)', { dash: [5, 5] });
      const a1 = [M[0][0], M[1][0]], a2 = [M[0][1], M[1][1]];
      P.poly([[0, 0], a1, L.add(a1, a2), a2], 'rgba(95,224,176,.16)', 'rgba(95,224,176,.7)');
      P.arrow([0, 0], a1, COL.c1); P.arrow([0, 0], a2, COL.c2);
    };
    Pl.render = () => draw(Pl, stageM(Bs, As, u), L.mm(B(), A()));
    Pr.render = () => draw(Pr, stageM(As, Bs, u), L.mm(A(), B()));
    const sS = slider(f.side, { label: 'stretch $B$', min: 0.4, max: 2.5, step: 0.05, value: s, fmt: v => v.toFixed(2) }, v => { s = v; upd(); });
    const thS = slider(f.side, { label: 'rotate $A$', min: -180, max: 180, step: 5, value: th, fmt: v => v + '°' }, v => { th = v; upd(); });
    const uS = slider(f.side, { label: 'steps', min: 0, max: 2, step: 0.01, value: 2, fmt: v => v.toFixed(2) }, v => { stopT?.(); u = v; upd(); });
    buttons(f.side, [{ label: '▶ Play both', pri: true, on: () => { stopT?.(); stopT = tween(2600, k => { u = 2 * k; uS.set(u); upd(); }); } }]);
    legend(f.side, [['rgba(232,236,248,.6)', 'dashed: the other order']]);
    const ro = readout(f.side);
    function upd() {
      const AB = L.mm(A(), B()), BA = L.mm(B(), A()), same = Math.max(...L.msub(AB, BA).flat().map(Math.abs)) < 1e-9;
      ro(`<div class="eqx">${TD(`AB=${matTex(nice(AB))}`)}</div><div class="eqx">${TD(`BA=${matTex(nice(BA))}`)}</div>
        <div class="flag ${same ? 'ok' : 'bad'}">${same ? T('AB=BA') + ' for these two' : T('AB\\neq BA')}</div>`);
      Pl.draw(); Pr.draw();
    }
    upd();
    return { reset() { stopT?.(); s = 2; th = 45; u = 2; sS.set(2); thS.set(45); uS.set(2); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.1.2 Solving Ax = b: row picture and column picture, the notes' three cases
const solve = {
  title: 'Solving $A\\mathbf{x}=\\mathbf{b}$: three cases',
  mount(f) {
    H(f, 'clamp(300px, 44vw, 380px)');
    const CASES = [
      { name: '1) Unique', A: [[1, 1], [2, -1]], b: [3, 0], exL: 3.6, exR: 3.8 },
      { name: '2) No solution', A: [[1, 1], [2, 2]], b: [2, 5], exL: 3.6, exR: 6.2 },
      { name: '3) Infinitely many', A: [[1, 2], [2, 4]], b: [4, 8], exL: 4.6, exR: 9.4 },
    ];
    const sp = el('div', 'split'); f.stage.appendChild(sp);
    const hl = el('div'), hr = el('div'); sp.append(hl, hr);
    const tl = el('div', 'stage-t', 'the two lines'), tr = el('div', 'stage-t', `columns of ${T('A')} and ${T('\\mathbf b')}`);
    hl.appendChild(tl); hr.appendChild(tr);
    const Pl = new Plane(hl, { extent: 3.6 }), Pr = new Plane(hr, { extent: 3.8 });
    let k = 0, A = CASES[0].A, b = CASES[0].b.slice();
    const r1 = () => L.rank(A) === 1;
    const colDir = () => L.unit(L.norm([A[0][0], A[1][0]]) > 1e-9 ? [A[0][0], A[1][0]] : [A[0][1], A[1][1]]);
    Pr.handle({ get: () => b, set: p => {
      b = p;
      if (r1()) { const d = colDir(), pr = L.scale(d, L.dot(b, d)); if (L.norm(L.sub(b, pr)) < 0.25) b = pr; }
      upd();
    }, color: COL.free, snap: 0.5 });
    const status = () => {
      if (!r1()) return 'unique';
      const d = colDir(); return L.norm(L.sub(b, L.scale(d, L.dot(b, d)))) < 1e-9 ? 'many' : 'none';
    };
    Pl.render = () => {
      const st = status();
      for (const [i, c] of [[0, COL.c1], [1, COL.c2]]) {
        const [p, q] = A[i], rhs = b[i];
        if (Math.hypot(p, q) < 1e-9) continue;
        const pt = L.scale([p, q], rhs / (p * p + q * q));      // closest point of the line to the origin
        Pl.lineDir(pt, [-q, p], c, { lw: st === 'many' && i === 1 ? 2.5 : 2.5, dash: st === 'many' && i === 1 ? [10, 8] : null });
      }
      if (st === 'unique') {
        const x = L.mv(L.inv2(A), b); Pl.dot(x, 6, COL.ink); Pl.label(x, `(${fmt(x[0])}, ${fmt(x[1])})`, COL.ink, { dx: 10, dy: -12, font: '600 12px "JetBrains Mono", monospace' });
      }
    };
    Pr.render = () => {
      const a1 = [A[0][0], A[1][0]], a2 = [A[0][1], A[1][1]], st = status();
      if (!r1()) {
        const bx = Pr.box(); Pr.ctx.fillStyle = 'rgba(124,196,255,.07)'; Pr.ctx.fillRect(0, 0, Pr.w, Pr.h);
        Pr.label([bx.x0 + 0.15 * (bx.x1 - bx.x0), bx.y0 + 0.08 * (bx.y1 - bx.y0)], 'Col(A) = the whole plane', COL.span, { font: FONT.ui });
        const x = L.mv(L.inv2(A), b), p1 = L.scale(a1, x[0]);
        Pr.arrow([0, 0], p1, COL.c1, { lw: 2, alpha: .55, dash: [5, 4] }); Pr.arrow(p1, L.add(p1, L.scale(a2, x[1])), COL.c2, { lw: 2, alpha: .55, dash: [5, 4] });
        Pr.label(L.scale(p1, 0.5), `${fmt(x[0])}·a₁`, COL.c1, { dx: -8, align: 'right', font: '600 11px "JetBrains Mono", monospace' });
        Pr.label(L.add(p1, L.scale(a2, x[1] * 0.5)), `${fmt(x[1])}·a₂`, COL.c2, { dx: 8, font: '600 11px "JetBrains Mono", monospace' });
      } else {
        const d = colDir(); Pr.lineDir([0, 0], d, COL.span, { lw: 2.5 }); Pr.label(L.scale(d, Pr.o.extent * 0.75), 'Col(A)', COL.span, { dx: 14, font: FONT.ui });
        if (st === 'none') { const pr = L.scale(d, L.dot(b, d)); Pr.seg(b, pr, COL.nul, { dash: [4, 4], lw: 1.5 }); }
      }
      Pr.arrow([0, 0], a1, COL.c1, { lw: 2.6 }); Pr.arrow([0, 0], a2, COL.c2, { lw: 2.6 });
      Pr.tipLabel(a1, 'a₁', COL.c1); Pr.tipLabel(a2, 'a₂', COL.c2);
      Pr.arrow([0, 0], b, COL.free, { lw: 3 }); Pr.tipLabel(b, 'b', COL.free, { gap: 26 });
    };
    const cs = seg(f.side, CASES.map((c, i) => ({ label: c.name, value: i })), i => load(i), 0);
    const ro = readout(f.side);
    legend(f.side, [[COL.c1, 'row 1 / column 1'], [COL.c2, 'row 2 / column 2'], [COL.free, '$\\mathbf b$ (drag)']]);
    function load(i) { k = i; A = CASES[i].A; b = CASES[i].b.slice(); Pl.o.extent = CASES[i].exL; Pr.o.extent = CASES[i].exR; cs.set(i); upd(); }
    const eq = (r, rhs) => { const [p, q] = r; const term = (c, v, first) => c === 0 ? '' : `${first ? (c < 0 ? '-' : '') : (c < 0 ? '-' : '+')}${Math.abs(c) === 1 ? '' : fmt(Math.abs(c))}${v}`; return `${term(p, 'x', true)}${term(q, 'y', p === 0)}=${fmt(rhs)}`; };
    function upd() {
      const st = status();
      let msg;
      if (st === 'unique') { const x = L.mv(L.inv2(A), b); msg = `<div class="flag ok">Unique solution: ${T(`x=${fmt(x[0])},\\ y=${fmt(x[1])}`)}</div><div class="line">full rank, ${T('\\mathbf x=A^{-1}\\mathbf b')}</div>`; }
      else if (st === 'none') msg = `<div class="flag bad">No solution: parallel lines, ${T('\\mathbf b\\notin\\operatorname{Col}(A)')}</div>`;
      else {
        const [p, q] = A[0], rhs = b[0];
        const xs = Math.abs(p) > 1e-9 ? `\\mathbf x=\\left(${fmt(rhs / p)}${q / p === 0 ? '' : `${q / p > 0 ? '-' : '+'}${Math.abs(q / p) === 1 ? '' : fmt(Math.abs(q / p))}t`},\\ t\\right)` : `\\mathbf x=(t,\\ ${fmt(rhs / q)})`;
        msg = `<div class="flag eig">Infinitely many: same line, ${T('\\mathbf b\\in\\operatorname{Col}(A)')}</div><div class="line">${T(xs)} for any ${T('t')}</div>`;
      }
      ro(`<div class="line">${T(eq(A[0], b[0]))}, &nbsp;${T(eq(A[1], b[1]))}</div><div class="eqx">${TD(`A=${matTex(A)},\\quad\\mathbf b=${vecTex(b)}`)}</div><div class="line">rank ${T(`=${L.rank(A)}`)}</div>${msg}`);
      Pl.draw(); Pr.draw();
    }
    load(0);
    return { reset: () => load(k), load, state: () => ({ k, b, status: status() }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.1.3 Rank–nullity in 3D: the notes' A = [[1,2,-1],[2,4,-2]], null space = plane spanned by (1,0,1), (0,1,2)
const nullspace3d = {
  title: 'The null space is a blind spot: $A(x+z)=Ax$',
  async mount(f) {
    H(f, 'clamp(320px, 50vw, 430px)');
    const THREE = await loadThree();
    const sp = el('div', 'split'); f.stage.appendChild(sp);
    const hl = el('div'), hr = el('div'); sp.append(hl, hr);
    hl.appendChild(el('div', 'stage-t', `input space ${T('\\mathbb R^3')}`));
    hr.appendChild(el('div', 'stage-t', `output space ${T('\\mathbb R^2')}`));
    const A = [[1, 2, -1], [2, 4, -2]], n1 = [1, 0, 1], n2 = [0, 1, 2];
    const S = new Scene3(hl, { cam: [7.4, -8.6, 5.6], target: [0.2, 0.3, 0.4], minDist: 5 });
    S.axes(3.2);
    for (const [p, t] of [[[3.5, 0, 0], 'x₁'], [[0, 3.5, 0], 'x₂'], [[0, 0, 3.5], 'x₃']]) { const l = S.label(t); l.p = p; l.e.style.color = '#8e99bd'; }
    // the null plane (red) and its spanning vectors
    const nhat1 = L.unit(n1), nperp = L.unit(L.sub(n2, L.scale(nhat1, L.dot(n2, nhat1))));
    S.planeSpan(nhat1, nperp, 0xff5d73, 2.2, 0.18);
    const a1 = S.arrow(0xff5d73, { r: 0.022, head: 0.18 }); a1.set([0, 0, 0], n1);
    const a2 = S.arrow(0xff5d73, { r: 0.022, head: 0.18 }); a2.set([0, 0, 0], n2);
    const ln1 = S.label('(1,0,1)ᵀ'); ln1.p = L.scale(n1, 1.15); ln1.e.style.color = COL.nul;
    const ln2 = S.label('(0,1,2)ᵀ'); ln2.p = L.scale(n2, 1.1); ln2.e.style.color = COL.nul;
    const lN = S.label('Null(A)'); lN.p = L.add(L.scale(nhat1, -1.9), L.scale(nperp, 1.6)); lN.e.style.color = COL.nul;
    // the plane of everything that lands on the same output as x: x + Null(A)
    const fiber = S.planeSpan(nhat1, nperp, 0xb28cff, 2.2, 0.07);
    const ax = S.arrow(0xe8ecf8), az = S.arrow(0xff5d73, { r: 0.03 }), axz = S.arrow(0xb28cff);
    const lx = S.label('x'), lz = S.label('z'), lxz = S.label('x + z');
    lx.e.style.color = COL.ink; lz.e.style.color = COL.nul; lxz.e.style.color = COL.free;
    // output plane
    const P = new Plane(hr, { extent: 3.2 });
    let s = 1, p = 0.8, q = 0.6;
    const vecs = () => { const x = [s, 0, 0], z = L.add(L.scale(n1, p), L.scale(n2, q)); return { x, z, xz: L.add(x, z) }; };
    P.render = () => {
      P.lineDir([0, 0], [1, 2], COL.span, { lw: 2.5 }); P.label(L.scale(L.unit([1, 2]), 2.7), 'Col(A)', COL.span, { dx: 12, font: FONT.ui });
      const { x, xz } = vecs(), Ax = L.mv(A, x), Axz = L.mv(A, xz);
      P.dot([0, 0], 6, COL.nul); P.label([0, 0], 'Az = 0', COL.nul, { dx: 12, dy: 14, font: '600 12px "JetBrains Mono", monospace' });
      P.arrow([0, 0], Axz, COL.free, { lw: 6, alpha: .55 }); P.arrow([0, 0], Ax, COL.ink, { lw: 2.4 });
      P.label(Ax, 'Ax = A(x+z)', COL.ink, { dx: 12, dy: -6, font: '600 12px "JetBrains Mono", monospace' });
    };
    const place = () => {
      const { x, z, xz } = vecs();
      ax.set([0, 0, 0], x); az.set(x, xz); axz.set([0, 0, 0], xz);
      fiber.position.copy(new THREE.Vector3(x[0], x[2], -x[1]));
      lx.p = L.scale(x, 1.08).map((v, i) => v + (i === 2 ? -0.25 : 0)); lz.p = L.add(x, L.scale(z, 0.5)); lxz.p = L.scale(xz, 1.08);
      lz.on = L.norm(z) > 0.15;
      S.draw(); P.draw();
    };
    slider(f.side, { label: '$x=(s,0,0)^T$, $s$', min: -1.5, max: 1.5, step: 0.05, value: s, fmt: v => v.toFixed(2) }, v => { s = v; upd(); });
    f.side.appendChild(el('div', 'line', `${T('z=p\\,(1,0,1)^T+q\\,(0,1,2)^T')}`));
    const pS = slider(f.side, { label: '$p$', min: -1.5, max: 1.5, step: 0.05, value: p, fmt: v => v.toFixed(2) }, v => { p = v; upd(); });
    const qS = slider(f.side, { label: '$q$', min: -1.2, max: 1.2, step: 0.05, value: q, fmt: v => v.toFixed(2) }, v => { q = v; upd(); });
    const ro = readout(f.side);
    function upd() {
      const { x, z, xz } = vecs(), Ax = L.mv(A, x), Az = L.mv(A, z), Axz = L.mv(A, xz);
      ro(`<div class="eqx">${TD(`A=${matTex(A)}`)}</div>
        <div class="eqx">${TD(`x+z=${vecTex(xz)}`)}</div>
        <div class="line">${T(`Az=${vecTex(Az)}`)}, &nbsp; ${T(`A(x+z)=${vecTex(Axz)}=Ax`)}</div>
        <div class="line">rank ${T(`${L.rank(A)}`)} + nullity ${T(`${3 - L.rank(A)}`)} = ${T('3=n')}</div>`);
      place();
    }
    upd();
    return { reset() { s = 1; p = 0.8; q = 0.6; pS.set(p); qS.set(q); f.side.querySelector('input').value = 1; f.side.querySelector('output').textContent = '1.00'; upd(); }, state: () => vecs() };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.1.5 Eigenvectors stay on their span
const eigen = {
  title: 'Eigenvectors keep their direction: $Av=\\lambda v$',
  mount(f) {
    H(f, STAGE_H);
    const P = new Plane(f.stage, { extent: 3.4 });
    const PRESETS = [
      { label: mt([[2, 1], [1, 2]]), M: [[2, 1], [1, 2]] },
      { label: mt([[4, 0], [0, 3]]), M: [[4, 0], [0, 3]] },
      { label: mt([[1, 2], [2, 4]]), M: [[1, 2], [2, 4]] },
      { label: mt([[4, 7], [2, 6]]), M: [[4, 7], [2, 6]] },
      { label: '$Q$', M: Q45 },
    ];
    let A = PRESETS[0].M, t = 1, v = [Math.cos(2.2), Math.sin(2.2)], stopT = null;
    const M = () => L.lerpM(L.I2(), A, t);
    P.handle({ get: () => v, set: p => { const n = L.norm(p) || 1; v = L.scale(p, 1 / n);
      const e = L.eig2(A); if (e.real) for (const w of e.vecs) for (const sg of [1, -1]) { const ww = L.scale(w, sg); if (L.dot(v, ww) > Math.cos(4 * Math.PI / 180)) v = ww; }
      upd(); }, color: COL.free, limit: 99 });
    P.render = () => {
      const m = M(), e = L.eig2(A);
      P.gridT(m, { color: 'rgba(124,196,255,.2)', axes: false });
      P.path(Array.from({ length: 97 }, (_, i) => [Math.cos(i * Math.PI / 48), Math.sin(i * Math.PI / 48)]), 'rgba(232,236,248,.25)', { lw: 1, dash: [3, 4] });
      if (e.real) for (const w of e.vecs) P.lineDir([0, 0], w, COL.eig, { lw: 1.6, alpha: .55 });
      for (let i = 0; i < 16; i++) {
        const a = i * Math.PI / 8, u = [Math.cos(a), Math.sin(a)], Au = L.mv(m, u);
        const on = e.real && e.vecs.some(w => Math.abs(u[0] * w[1] - u[1] * w[0]) < 1e-9);
        P.arrow([0, 0], Au, on ? COL.eig : 'rgba(232,236,248,.38)', { lw: on ? 2.6 : 1.4, head: 8 });
      }
      const Av = L.mv(m, v);
      P.arrow([0, 0], v, COL.free, { lw: 2, dash: [5, 4], alpha: .85 });
      P.arrow([0, 0], Av, COL.free, { lw: 3.4 });
      P.tipLabel(v, 'v', COL.free); P.tipLabel(Av, t === 1 ? 'Av' : '', COL.free);
    };
    const ctl = el('div'); f.side.appendChild(ctl); ctl.appendChild(el('h5', '', 'Matrices from the notes'));
    const pick = seg(ctl, PRESETS.map((p, i) => ({ label: p.label, value: i })), i => { A = PRESETS[i].M; P.o.extent = i === 3 ? 6 : i === 2 ? 4.4 : 3.4; play(); }, 0);
    const tS = slider(f.side, { label: 'apply $A$', min: 0, max: 1, step: 0.01, value: 1, fmt: v => v.toFixed(2) }, x => { stopT?.(); t = x; upd(); });
    buttons(f.side, [{ label: '▶ Play', pri: true, on: () => play() }]);
    // characteristic polynomial chart
    f.side.appendChild(el('h5', '', `${T('\\det(A-\\lambda I)')} against ${T('\\lambda')}`));
    const plot = el('div'); plot.style.cssText = 'position:relative;height:130px;border:1px solid var(--line-soft);border-radius:8px'; f.side.appendChild(plot);
    const C = new Plane(plot, { extent: 1, bgGrid: false });
    C.render = () => {
      const tr = A[0][0] + A[1][1], dt = L.det2(A), pf = l => l * l - tr * l + dt, e = L.eig2(A);
      const lo = Math.min(-1, (e.real ? e.vals[1] : tr / 2) - 1.5), hi = Math.max(1, (e.real ? e.vals[0] : tr / 2) + 1.5);
      const xs = Array.from({ length: 121 }, (_, i) => lo + (hi - lo) * i / 120), ys = xs.map(pf);
      const yl = Math.min(...ys, -1), yh = Math.max(...ys.filter(y => y < 4 * Math.max(1, Math.abs(yl))), 1);
      const X = x => 8 + (x - lo) / (hi - lo) * (C.w - 16), Y = y => C.h - 14 - (y - yl) / (yh - yl) * (C.h - 24);
      const g = C.ctx; g.strokeStyle = 'rgba(232,236,248,.3)'; g.lineWidth = 1; g.beginPath(); g.moveTo(0, Y(0)); g.lineTo(C.w, Y(0)); g.stroke();
      g.strokeStyle = COL.span; g.lineWidth = 2; g.beginPath(); xs.forEach((x, i) => (i ? g.lineTo : g.moveTo).call(g, X(x), Y(Math.max(yl - 5, Math.min(yh + 5, ys[i]))))); g.stroke();
      g.font = '500 10.5px "JetBrains Mono", monospace'; g.fillStyle = COL.muted; g.textAlign = 'center';
      for (let k = Math.ceil(lo); k <= hi; k++) { g.fillText(k, X(k), C.h - 3); }
      if (e.real) e.vals.forEach((l, i) => { g.fillStyle = COL.eig; g.beginPath(); g.arc(X(l), Y(0), 4.5, 0, 7); g.fill(); g.fillText(`λ${'₁₂'[i]}=${fmt(l)}`, X(l) + (i ? -18 : 18), Y(0) - 9); });
      else { g.fillStyle = COL.nul; g.textAlign = 'left'; g.fillText('no real roots', 10, 14); }
    };
    const ro = readout(f.side);
    function play() { stopT?.(); stopT = tween(1400, u => { t = u; tS.set(u); upd(); }); }
    function upd() {
      const e = L.eig2(A), m = M(), Av = L.mv(A, v), cross = v[0] * Av[1] - v[1] * Av[0], isE = L.norm(Av) < 1e-9 || Math.abs(cross) < 1e-6 * Math.max(1, L.norm(Av));
      const lam = L.dot(Av, v);
      ro(`<div class="eqx">${TD(`A=${matTex(nice(A))}`)}</div>
        <div class="line">${T(`v=${vecTex(v)}`)} &nbsp; ${T(`Av=${vecTex(Av)}`)}</div>
        ${isE ? `<div class="flag eig">${T(`Av=\\lambda v`)} with ${T(`\\lambda=${fmt(lam)}`)}: ${T('v')} is an eigenvector</div>`
          : `<div class="line">${T('Av')} left the line of ${T('v')}</div>`}
        ${e.real ? `<div class="line">${e.vals.map((l, i) => T(`\\lambda_${i + 1}=${fmt(l)},\\ v_${i + 1}\\propto${vecTex(e.vecs[i].map(x => x / (Math.abs(e.vecs[i][0]) > 1e-9 ? Math.abs(e.vecs[i][0]) : 1)))}`)).join('<br>')}</div>`
          : `<div class="flag bad">no real eigenvalues: every direction turns</div>`}`);
      P.draw(); C.draw();
    }
    upd();
    return { reset() { A = PRESETS[0].M; pick.set(0); P.o.extent = 3.4; v = [Math.cos(2.2), Math.sin(2.2)]; t = 1; tS.set(1); upd(); }, setV(p) { v = L.unit(p); upd(); }, state: () => ({ A, v, Av: L.mv(A, v), eig: L.eig2(A) }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.1.6 A = P D P^-1 read right to left
const diagonalize = {
  title: 'Diagonalization: $A=PDP^{-1}$, read right to left',
  mount(f) {
    H(f, STAGE_H);
    const P_ = [[1, 1], [1, -1]], D = [[3, 0], [0, 1]], Pi = L.inv2(P_), A = [[2, 1], [1, 2]];
    const C = [L.I2(), Pi, L.mm(D, Pi), L.mm(P_, L.mm(D, Pi))];
    const P = new Plane(f.stage, { extent: 3.6 });
    let s = 0, stopT = null, k = 4;
    const cur = () => { const i = Math.min(2, Math.floor(s)), u = s - i; return L.lerpM(C[i], C[i + 1], u); };
    const x0 = [0.5, 1.5];
    P.render = () => {
      const m = cur();
      P.gridT(m, { color: 'rgba(124,196,255,.22)', axes: false });
      P.gridT(L.mm(m, P_), { color: 'rgba(255,200,87,.32)', axisColor: 'rgba(255,200,87,.85)', n: 6 });
      const v1 = L.mv(m, [1, 1]), v2 = L.mv(m, [1, -1]), x = L.mv(m, x0);
      P.arrow([0, 0], v1, COL.eig, { lw: 3 }); P.arrow([0, 0], v2, COL.span, { lw: 3 });
      P.tipLabel(v1, 'v₁', COL.eig); P.tipLabel(v2, 'v₂', COL.span);
      P.arrow([0, 0], x, COL.free, { lw: 2.6 }); P.tipLabel(x, s >= 3 ? 'Ax' : 'x', COL.free);
      if (s >= 3) { P.arrow([0, 0], L.mv(A, x0), COL.ink, { lw: 1.2, dash: [4, 4], alpha: .8 }); }
    };
    const NAMES = ['$x$', '$P^{-1}x$', '$DP^{-1}x$', '$PDP^{-1}x=Ax$'];
    const sS = slider(f.side, { label: 'stage', min: 0, max: 3, step: 0.01, value: 0, fmt: v => v.toFixed(2) }, v => { stopT?.(); s = v; upd(); });
    const go = (to, ms = 1100) => { stopT?.(); const s0 = s; stopT = tween(ms * Math.max(.4, Math.abs(to - s0)), u => { s = s0 + (to - s0) * u; sS.set(s); upd(); }); };
    buttons(f.side, [
      { label: '1) $P^{-1}$', on: () => { s = 0; go(1); } },
      { label: '2) $D$', on: () => { s = 1; go(2); } },
      { label: '3) $P$', on: () => { s = 2; go(3); } },
      { label: '▶ All', pri: true, on: () => { s = 0; go(3, 1300); } },
    ]);
    legend(f.side, [[COL.eig, 'eigenvector grid ($v_1$, $v_2$ as axes)'], ['rgba(124,196,255,.7)', 'regular grid']]);
    const ro = readout(f.side);
    f.side.appendChild(el('h5', '', `Powers: ${T('A^k=PD^kP^{-1}')}`));
    const kS = slider(f.side, { label: '$k$', min: 1, max: 10, step: 1, value: k, fmt: v => String(v) }, v => { k = v; upd(); });
    const ro2 = readout(f.side);
    function upd() {
      const stage = Math.min(3, Math.round(s * 100) / 100), i = Math.floor(stage + 1e-9);
      const msgs = ['the regular directions', `${T('P^{-1}')}: the eigenvector axes become the coordinate axes`, `${T('D')}: scale along the axes by ${T('3')} and ${T('1')}`, `${T('P')}: back to the regular directions, ${T('PDP^{-1}=A')}`];
      ro(`<div class="line">showing ${T(NAMES[Math.min(3, i)].slice(1, -1))}: ${msgs[Math.min(3, i)]}</div>
        <div class="eqx">${TD(`P=${matTex(P_)},\\quad D=${matTex(D)}`)}</div><div class="eqx">${TD(`P^{-1}=\\tfrac12${matTex([[1, 1], [1, -1]])}`)}</div>`);
      const Dk = [[3 ** k, 0], [0, 1]], Ak = L.mm(P_, L.mm(Dk, Pi));
      ro2(`<div class="eqx">${TD(`D^{${k}}=${matTex(Dk, 0)}`)}</div><div class="eqx">${TD(`A^{${k}}=${matTex(Ak, 0)}`)}</div>`);
      P.draw();
    }
    upd();
    return { reset() { stopT?.(); s = 0; sS.set(0); k = 4; kS.set(4); upd(); }, go, state: () => ({ s, M: cur() }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.1.7 One vector, three sizes; ellipse x^T A x = 1
const norms = {
  title: 'Shapes of equal norm',
  mount(f) {
    H(f, STAGE_H);
    const P = new Plane(f.stage, { extent: 8.2 });
    let mode = 'p', x = [3, -4], a11 = 1, a22 = 4, a12 = 0;
    const Am = () => [[a11, a12], [a12, a22]];
    P.handle({ get: () => x, set: p => { x = p; upd(); }, color: COL.free, snap: 0.5, visible: () => mode === 'p', limit: 7.5 });
    P.render = () => {
      if (mode === 'p') {
        const n1 = Math.abs(x[0]) + Math.abs(x[1]), n2 = L.norm(x), ni = Math.max(Math.abs(x[0]), Math.abs(x[1]));
        P.poly([[n1, 0], [0, n1], [-n1, 0], [0, -n1]], 'rgba(95,224,176,.06)', COL.c1, { lw: 2 });
        P.path(Array.from({ length: 129 }, (_, i) => [n2 * Math.cos(i * Math.PI / 64), n2 * Math.sin(i * Math.PI / 64)]), COL.span, { lw: 2 });
        P.poly([[ni, ni], [-ni, ni], [-ni, -ni], [ni, -ni]], null, COL.c2, { lw: 2 });
        P.path([[0, 0], [x[0], 0], x], COL.c1, { lw: 2, dash: [5, 4], alpha: .9 });
        P.seg([0, 0], x, COL.span, { lw: 2 });
        P.dot(x, 5, COL.free); P.label(x, `x = (${fmt(x[0])}, ${fmt(x[1])})`, COL.free, { dx: 14, dy: x[1] < 0 ? 14 : -14, font: '600 12px "JetBrains Mono", monospace' });
        P.label([n1, 0], `‖x‖₁ = ${fmt(n1)}`, COL.c1, { dy: -12, dx: -6, align: 'right', font: '600 12px "JetBrains Mono", monospace' });
        P.label([-n2 * S2, n2 * S2], `‖x‖₂ = ${fmt(n2)}`, COL.span, { dx: -8, dy: -8, align: 'right', font: '600 12px "JetBrains Mono", monospace' });
        P.label([ni, ni], `‖x‖∞ = ${fmt(ni)}`, COL.c2, { dx: -4, dy: -12, align: 'right', font: '600 12px "JetBrains Mono", monospace' });
      } else {
        const A = Am(), e = L.eig2(A);
        P.path(Array.from({ length: 129 }, (_, i) => [Math.cos(i * Math.PI / 64), Math.sin(i * Math.PI / 64)]), 'rgba(232,236,248,.3)', { lw: 1.2, dash: [4, 4] });
        if (!(a11 > 0 && a22 > 0 && a11 * a22 - a12 * a12 > 1e-9)) return;
        const [l1, l2] = e.vals, [w1, w2] = e.vecs, r1 = 1 / Math.sqrt(l1), r2 = 1 / Math.sqrt(l2);
        P.lineDir([0, 0], w1, COL.eig, { lw: 1.2, alpha: .5 }); P.lineDir([0, 0], w2, COL.eig, { lw: 1.2, alpha: .5 });
        P.path(Array.from({ length: 129 }, (_, i) => { const a = i * Math.PI / 64; return L.add(L.scale(w1, r1 * Math.cos(a)), L.scale(w2, r2 * Math.sin(a))); }), COL.span, { lw: 2.6, fill: 'rgba(124,196,255,.08)' });
        P.arrow([0, 0], L.scale(w1, r1), COL.eig, { lw: 2.4, head: 9 }); P.arrow([0, 0], L.scale(w2, r2), COL.eig, { lw: 2.4, head: 9 });
        P.label(L.scale(w1, r1), `1/√λ₁ = ${fmt(r1)}`, COL.eig, { dx: 8, dy: -10, font: '600 11.5px "JetBrains Mono", monospace' });
        P.label(L.scale(w2, r2), `1/√λ₂ = ${fmt(r2)}`, COL.eig, { dx: 8, dy: -10, font: '600 11.5px "JetBrains Mono", monospace' });
      }
    };
    const m = seg(f.side, [{ label: '$\\|x\\|_1,\\ \\|x\\|_2,\\ \\|x\\|_\\infty$', value: 'p' }, { label: 'ellipse $x^TAx=1$', value: 'e' }], v => { mode = v; P.o.extent = v === 'p' ? 8.2 : 1.45; P.o.step = v === 'p' ? 1 : 0.5; boxE.hidden = v === 'p'; upd(); }, 'p');
    const boxE = el('div'); boxE.style.display = 'grid'; boxE.style.gap = '6px'; boxE.hidden = true; f.side.appendChild(boxE);
    const s11 = slider(boxE, { label: '$a_{11}$', min: 0.25, max: 6, step: 0.25, value: a11, fmt: v => v.toFixed(2) }, v => { a11 = v; upd(); });
    const s22 = slider(boxE, { label: '$a_{22}$', min: 0.25, max: 6, step: 0.25, value: a22, fmt: v => v.toFixed(2) }, v => { a22 = v; upd(); });
    const s12 = slider(boxE, { label: '$a_{12}=a_{21}$', min: -3, max: 3, step: 0.25, value: a12, fmt: v => v.toFixed(2) }, v => { a12 = v; upd(); });
    const ro = readout(f.side);
    function upd() {
      if (mode === 'p') {
        const ax = x.map(Math.abs);
        ro(`<div class="line">${T(`\\|x\\|_1=${fmt(ax[0])}+${fmt(ax[1])}=${fmt(ax[0] + ax[1])}`)}</div>
          <div class="line">${T(`\\|x\\|_2=\\sqrt{${fmt(x[0] ** 2)}+${fmt(x[1] ** 2)}}=${fmt(L.norm(x), 3)}`)}</div>
          <div class="line">${T(`\\|x\\|_\\infty=\\max(${fmt(ax[0])},${fmt(ax[1])})=${fmt(Math.max(...ax))}`)}</div>`);
      } else {
        const A = Am(), pd = a11 > 0 && a22 > 0 && a11 * a22 - a12 * a12 > 1e-9, e = L.eig2(A);
        ro(`<div class="eqx">${TD(`A=${matTex(A)}`)}</div>` + (pd ? `<div class="line">${T(`\\lambda_1=${fmt(e.vals[0])},\\ \\lambda_2=${fmt(e.vals[1])}`)}</div>
          <div class="line">${a12 === 0 ? 'axes along the coordinate directions' : 'tilted: the axes are the eigenvectors of ' + T('A')}</div>`
          : `<div class="flag bad">${T('x^TAx=1')} is not an ellipse for this ${T('A')}</div>`));
      }
      P.draw();
    }
    upd();
    return { reset() { mode = 'p'; m.set('p'); x = [3, -4]; a11 = 1; a22 = 4; a12 = 0; s11.set(1); s22.set(4); s12.set(0); boxE.hidden = true; P.o.extent = 8.2; P.o.step = 1; upd(); }, setX(p) { x = p; upd(); }, setMode(v) { mode = v; P.o.extent = v === 'p' ? 8.2 : 1.45; boxE.hidden = v === 'p'; upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.1.7 Why L1 gives zeros: smallest loss contour touching the L1 diamond vs the L2 circle (exact solvers)
const HL = (() => { const R = L.rot(20 * Math.PI / 180); return L.mm(L.mm(R, [[1, 0], [0, 4]]), L.T(R)); })();
export function solveL1(c, Hm, t) {
  const loss = w => { const d = L.sub(w, c); return L.dot(d, L.mv(Hm, d)); };
  if (Math.abs(c[0]) + Math.abs(c[1]) <= t) return c.slice();
  const V = [[t, 0], [0, t], [-t, 0], [0, -t]]; let best = null;
  for (let k = 0; k < 4; k++) {
    const p0 = V[k], d = L.sub(V[(k + 1) % 4], p0), a = L.dot(d, L.mv(Hm, d)), b = 2 * L.dot(d, L.mv(Hm, L.sub(p0, c)));
    const s = Math.max(0, Math.min(1, -b / (2 * a))), w = L.add(p0, L.scale(d, s)), fw = loss(w);
    if (!best || fw < best[0]) best = [fw, w];
  }
  return best[1].map(v => Math.abs(v) < 1e-12 ? 0 : v);
}
export function solveL2(c, Hm, t) {
  if (L.norm(c) <= t) return c.slice();
  let lo = 0, hi = 1e6, w = c;
  for (let i = 0; i < 200; i++) { const mu = (lo + hi) / 2; w = L.mv(L.inv2(L.madd(Hm, [[mu, 0], [0, mu]])), L.mv(Hm, c)); if (L.norm(w) > t) lo = mu; else hi = mu; }
  return w;
}
const l1corner = {
  title: 'Why $L_1$ gives zeros and $L_2$ just shrinks',
  mount(f) {
    H(f, 'clamp(280px, 42vw, 360px)');
    const sp = el('div', 'split'); f.stage.appendChild(sp);
    const hl = el('div'), hr = el('div'); sp.append(hl, hr);
    hl.appendChild(el('div', 'stage-t', `${T('L_1')}: ${T('\\|w\\|_1\\le t')}`));
    hr.appendChild(el('div', 'stage-t', `${T('L_2')}: ${T('\\|w\\|_2\\le t')}`));
    const Pl = new Plane(hl, { extent: 1.75, center: [0.75, 0.25] }), Pr = new Plane(hr, { extent: 1.75, center: [0.75, 0.25] });
    let c = [2.2, 0.6], t = 1;
    const loss = w => { const d = L.sub(w, c); return L.dot(d, L.mv(HL, d)); };
    const e = L.eig2(HL);
    const contour = lev => Array.from({ length: 97 }, (_, i) => { const a = i * Math.PI / 48; return L.add(c, L.add(L.scale(e.vecs[0], Math.sqrt(lev / e.vals[0]) * Math.cos(a)), L.scale(e.vecs[1], Math.sqrt(lev / e.vals[1]) * Math.sin(a)))); });
    const draw = (P, which) => {
      const w = which === 1 ? solveL1(c, HL, t) : solveL2(c, HL, t), lw = loss(w);
      for (const k of [0.25, 0.5, 1, 1.5, 2, 3, 4]) { const lev = lw * k; if (lev > 1e-9) P.path(contour(lev), 'rgba(178,140,255,.28)', { lw: 1 }); }
      if (which === 1) P.poly([[t, 0], [0, t], [-t, 0], [0, -t]], 'rgba(95,224,176,.10)', COL.c1, { lw: 2.2 });
      else P.path(Array.from({ length: 129 }, (_, i) => [t * Math.cos(i * Math.PI / 64), t * Math.sin(i * Math.PI / 64)]), COL.span, { lw: 2.2, fill: 'rgba(124,196,255,.10)' });
      if (lw > 1e-12) P.path(contour(lw), COL.free, { lw: 2.4 });
      P.dot(c, 4, COL.free); P.label(c, 'min L(w)', COL.free, { dx: 8, dy: -10, font: '600 11px "JetBrains Mono", monospace' });
      const zero = w.some(v => v === 0);
      P.dot(w, 6.5, zero ? COL.eig : COL.ink);
      P.label(w, `(${fmt(w[0])}, ${fmt(w[1])})`, zero ? COL.eig : COL.ink, { dx: -10, dy: 16, align: 'right', font: '600 11.5px "JetBrains Mono", monospace' });
    };
    Pl.render = () => draw(Pl, 1); Pr.render = () => draw(Pr, 2);
    for (const P of [Pl, Pr]) P.handle({ get: () => c, set: p => { c = [Math.max(-1.5, Math.min(2.7, p[0])), Math.max(-1.6, Math.min(2, p[1]))]; upd(); }, color: COL.free, limit: 3 });
    slider(f.side, { label: 'budget $t$', min: 0.3, max: 1.6, step: 0.05, value: t, fmt: v => v.toFixed(2) }, v => { t = v; upd(); });
    const ro = readout(f.side);
    legend(f.side, [[COL.free, 'contours of $L(w)$'], [COL.c1, '$\\|w\\|_1=t$'], [COL.span, '$\\|w\\|_2=t$']]);
    function upd() {
      const w1 = solveL1(c, HL, t), w2 = solveL2(c, HL, t), z1 = w1.findIndex(v => v === 0);
      ro(`<div class="line">${T('L_1')}: ${T(`w=(${fmt(w1[0], 3)},\\ ${fmt(w1[1], 3)})`)}</div>
        ${z1 >= 0 ? `<div class="flag eig">a corner: ${T(`w_${z1 + 1}=0`)} exactly</div>` : '<div class="line">touches an edge, no weight is 0</div>'}
        <div class="line">${T('L_2')}: ${T(`w=(${fmt(w2[0], 3)},\\ ${fmt(w2[1], 3)})`)}</div>
        <div class="line">${w2.some(v => Math.abs(v) < 5e-4) ? 'a weight is (near) 0 only when the centre sits on an axis' : 'both weights shrink, neither is 0'}</div>`);
      Pl.draw(); Pr.draw();
    }
    upd();
    return { reset() { c = [2.2, 0.6]; t = 1; f.side.querySelector('input').value = 1; f.side.querySelector('output').textContent = '1.00'; upd(); }, state: () => ({ c, t, w1: solveL1(c, HL, t), w2: solveL2(c, HL, t) }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.1.7 Loss landscape L = w1^2 + w2^2 and its contour plot (bird's-eye view)
const STOPS = [[0x1b, 0x1f, 0x4b], [0x24, 0x4f, 0x86], [0x2a, 0x9c, 0x96], [0x96, 0xd0, 0x78], [0xf4, 0xe8, 0xa4]];
const ramp = t => { t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1); const i = Math.min(STOPS.length - 2, Math.floor(t)), u = t - i, a = STOPS[i], b = STOPS[i + 1]; return [0, 1, 2].map(k => (a[k] + (b[k] - a[k]) * u) / 255); };
const bowl = {
  title: 'A contour plot is a bird\u2019s-eye view of the loss landscape',
  async mount(f) {
    H(f, 'clamp(320px, 50vw, 440px)');
    f.body.classList.add('wide');
    f.side.remove();
    const THREE = await loadThree();
    const hz = 0.17;                                  // display height = 0.17 L (height rescaled for the picture only)
    const S = new Scene3(f.stage, { cam: [6.2, -6.4, 6.0], target: [0, 0, 0.9], minDist: 4, maxDist: 22 });
    const R = 3.25, NR = 60, NA = 96, pos = [], colr = [], idx = [];
    for (let i = 0; i <= NR; i++) for (let j = 0; j <= NA; j++) {
      const r = R * i / NR, a = 2 * Math.PI * j / NA, x = r * Math.cos(a), y = r * Math.sin(a), Lv = x * x + y * y;
      pos.push(x, Lv * hz, -y); colr.push(...ramp(Lv / (R * R)));
    }
    for (let i = 0; i < NR; i++) for (let j = 0; j < NA; j++) { const a = i * (NA + 1) + j, b = a + NA + 1; idx.push(a, b, a + 1, b, b + 1, a + 1); }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(colr, 3)); g.setIndex(idx); g.computeVertexNormals();
    const mesh = new THREE.Mesh(g, new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide, shininess: 25, transparent: true, opacity: 0.93 }));
    S.scene.add(mesh);
    const floor = -0.02;
    for (const Lv of [1, 4, 9]) {
      const r = Math.sqrt(Lv), ring = Array.from({ length: 129 }, (_, i) => [r * Math.cos(i * Math.PI / 64), r * Math.sin(i * Math.PI / 64)]);
      S.line(ring.map(([x, y]) => [x, y, Lv * hz + 0.01]), 0xff6b6b, { opacity: 0.95 });
      S.line(ring.map(([x, y]) => [x, y, floor]), 0xff6b6b, { opacity: 0.75 });
      const ang = { 1: -0.2, 4: -0.55, 9: -0.9 }[Lv], lb = S.label(`L=${Lv}`); lb.p = [r * Math.cos(ang), r * Math.sin(ang), Lv * hz + 0.12]; lb.e.style.color = '#ff8f8f'; lb.e.style.fontStyle = 'normal'; lb.e.style.font = '600 11.5px "JetBrains Mono", monospace';
    }
    S.line([[-3.6, 0, floor], [3.6, 0, floor]], 0x8e99bd, { opacity: .4 }); S.line([[0, -3.6, floor], [0, 3.6, floor]], 0x8e99bd, { opacity: .4 });
    for (const [p, t] of [[[3.9, 0, floor], 'w₁'], [[0, 3.9, floor], 'w₂']]) { const l = S.label(t); l.p = p; l.e.style.color = COL.muted; }
    const w0 = [3 * Math.cos(2.35), 3 * Math.sin(2.35)];
    const ball = S.ball(0xff7a59, 0.11), shadow = S.ball(0xff7a59, 0.06);
    const trail = S.line([[...w0, 9 * hz]], 0xff7a59), trailF = S.line([[...w0, floor]], 0xff7a59, { opacity: .8 });
    const start = S.label('model starts here'); start.p = [w0[0], w0[1], 9 * hz + 0.45]; start.e.style.cssText += ';color:#e8ecf8;font:600 11.5px Sora,sans-serif;font-style:normal';
    const ro = el('div', 'stage-t'); ro.style.top = 'auto'; ro.style.bottom = '8px'; f.stage.appendChild(ro);
    const acts = el('div', 'acts'); acts.style.cssText = 'position:absolute;top:8px;right:10px;z-index:3'; f.stage.appendChild(acts);
    let w = w0.slice(), path = [w0.slice()], run = 0;
    const set = () => {
      const Lv = w[0] ** 2 + w[1] ** 2;
      ball.set([w[0], w[1], Lv * hz + 0.1]); shadow.set([w[0], w[1], floor]);
      S.setLine(trail, path.map(([x, y]) => [x, y, (x * x + y * y) * hz + 0.03])); S.setLine(trailF, path.map(([x, y]) => [x, y, floor]));
      ro.innerHTML = `${T(`w=(${fmt(w[0])},\\ ${fmt(w[1])})`)} &nbsp; ${T(`L(w)=${fmt(Lv, 2)}`)}`;
      S.draw();
    };
    const roll = () => {
      cancelAnimationFrame(run); w = w0.slice(); path = [w.slice()];
      const step = () => {   // each frame: one small step straight downhill, w <- w - 0.02 * gradient L(w) (gradient = 2w)
        w = w.map(v => v - 0.02 * 2 * v); path.push(w.slice()); set();
        if (w[0] ** 2 + w[1] ** 2 > 2e-3) run = requestAnimationFrame(step);
      };
      run = requestAnimationFrame(step);
    };
    const [b1, b2] = ['▶ Roll the ball', 'View from above'].map((t, i) => { const b = el('button', i ? '' : 'pri', t); b.type = 'button'; acts.appendChild(b); return b; });
    let top = false;
    b1.onclick = roll;
    b2.onclick = () => { top = !top; b2.textContent = top ? '3D view' : 'View from above'; S.fly(top ? [0, -0.01, 11] : [6.2, -6.4, 6.0], top ? [0, 0, 0] : [0, 0, 0.9]); };
    set();
    return { reset() { cancelAnimationFrame(run); w = w0.slice(); path = [w.slice()]; top = false; b2.textContent = 'View from above'; S.fly([6.2, -6.4, 6.0], [0, 0, 0.9], 500); set(); }, roll, top: () => b2.click(), state: () => ({ w, L: w[0] ** 2 + w[1] ** 2 }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.1.8 SVD: rotate (V^T) -> stretch (Sigma) -> rotate again (U)
const svd2v = {
  title: 'SVD: rotate $\\to$ stretch/shrink $\\to$ rotate again',
  mount(f) {
    H(f, STAGE_H);
    const P = new Plane(f.stage, { extent: 3.6 });
    const PRESETS = [
      { label: mt([[2, 1], [1, 2]]), M: [[2, 1], [1, 2]] },
      { label: mt([[1, 2], [3, 4]]), M: [[1, 2], [3, 4]] },
      { label: mt([[1, 2], [2, 4]]), M: [[1, 2], [2, 4]] },
      { label: '$Q$', M: Q45 },
    ];
    let A = PRESETS[0].M.map(r => r.slice()), s = 3, stopT = null;
    const parts = () => {
      const d = L.svd2(A);
      const Vt = u => L.rot(-d.theta * u), Sg = u => [[1 + (d.s1 - 1) * u, 0], [0, 1 + (d.s2 - 1) * u]], U = u => L.rot(d.phi * u);
      return { d, M: s <= 1 ? Vt(s) : s <= 2 ? L.mm(Sg(s - 1), Vt(1)) : L.mm(U(s - 2), L.mm(Sg(1), Vt(1))) };
    };
    const colAt = j => () => (s < 3 ? [NaN, NaN] : [A[0][j], A[1][j]]);
    const setCol = j => p => { A[0][j] = p[0]; A[1][j] = p[1]; pick.set(null); upd(); };
    P.handle({ get: colAt(0), set: setCol(0), color: COL.c1, snap: 0.5, visible: () => s >= 3 });
    P.handle({ get: colAt(1), set: setCol(1), color: COL.c2, snap: 0.5, visible: () => s >= 3 });
    P.render = () => {
      const { d, M } = parts();
      P.gridT(M, { color: 'rgba(124,196,255,.2)', axes: false });
      P.path(Array.from({ length: 129 }, (_, i) => L.mv(M, [Math.cos(i * Math.PI / 64), Math.sin(i * Math.PI / 64)])), '#2fb7a6', { lw: 2.4, fill: 'rgba(47,183,166,.08)' });
      const a = L.mv(M, d.v1), b = L.mv(M, d.v2);
      P.arrow([0, 0], a, COL.eig, { lw: 3 }); P.arrow([0, 0], b, COL.span, { lw: 3 });
      const nm = s < 1e-9 ? ['v₁', 'v₂'] : s < 3 - 1e-9 ? ['', ''] : ['σ₁u₁', 'σ₂u₂'];
      P.tipLabel(a, nm[0], COL.eig); P.tipLabel(b, nm[1], COL.span);
      if (s >= 3) { P.arrow([0, 0], [A[0][0], A[1][0]], COL.c1, { lw: 1.6, alpha: .8 }); P.arrow([0, 0], [A[0][1], A[1][1]], COL.c2, { lw: 1.6, alpha: .8 }); }
    };
    const ctl = el('div'); f.side.appendChild(ctl); ctl.appendChild(el('h5', '', 'Matrices from the notes'));
    const pick = seg(ctl, PRESETS.map((p, i) => ({ label: p.label, value: i })), i => { A = PRESETS[i].M.map(r => r.slice()); P.o.extent = i === 1 ? 6 : i === 2 ? 5.6 : 3.6; s = 0; sS.set(0); go(3, 2400); }, 0);
    const sS = slider(f.side, { label: 'step', min: 0, max: 3, step: 0.01, value: s, fmt: v => v.toFixed(2) }, v => { stopT?.(); s = v; upd(); });
    const go = (to, ms = 900) => { stopT?.(); const s0 = s; stopT = tween(ms, u => { s = s0 + (to - s0) * u; sS.set(s); upd(); }); };
    buttons(f.side, [
      { label: '1) $V^T$', on: () => { s = 0; go(1); } }, { label: '2) $\\Sigma$', on: () => { s = 1; go(2); } },
      { label: '3) $U$', on: () => { s = 2; go(3); } }, { label: '▶ All', pri: true, on: () => { s = 0; go(3, 2400); } },
    ]);
    const ro = readout(f.side);
    function upd() {
      const { d } = parts(), Av1 = L.mv(A, d.v1), r = L.rank(A), nz = d.sigma.filter(x => x > 1e-9).length;
      ro(`<div class="eqx">${TD(`A=${matTex(nice(A))}`)}</div>
        <div class="line">${T(`\\sigma_1=${fmt(d.sigma[0], 3)},\\ \\sigma_2=${fmt(d.sigma[1], 3)}`)}</div>
        <div class="line">${T(`v_1=${vecTex(d.v1)}`)}, ${T(`Av_1=${vecTex(Av1)}=\\sigma_1u_1`)}</div>
        <div class="line">${T('V^T')} rotates by ${T(`${fmt(-deg(d.theta), 1)}^\\circ`)}, ${T('U')} by ${T(`${fmt(deg(d.phi), 1)}^\\circ`)}${d.s2 < -1e-12 ? `; ${T('\\det A<0')}: one axis is flipped (a reflection)` : ''}</div>
        <div class="line">nonzero singular values: ${T(String(nz))} = rank ${T(`${r}`)}</div>`);
      P.draw();
    }
    upd();
    return { reset() { A = PRESETS[0].M.map(r => r.slice()); pick.set(0); P.o.extent = 3.6; stopT?.(); s = 3; sS.set(3); upd(); }, go, state: () => ({ A, s, ...parts() }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.1.8 SVD as layers on the notes' 4x4 matrix; Eckart–Young
const A4 = [[4, 1, 2, 0], [1, 4, 0, 2], [2, 0, 4, 1], [0, 2, 1, 4]];
const svdLayers = {
  title: 'SVD as layers: $A_k=\\sum_{i\\le k}\\sigma_iu_iv_i^T$',
  mount(f) {
    const { s, U, V } = L.svd(A4);
    const layer = i => L.mscale(L.outer(U[i], V[i]), s[i]);
    const Ak = k => { let M = L.zeros(4, 4); for (let i = 0; i < k; i++) M = L.madd(M, layer(i)); return M; };
    f.stage.style.minHeight = '0';
    const grid = el('div', 'mtabs'); f.stage.appendChild(grid);
    const cell = v => { const a = Math.min(1, Math.abs(v) / 4), c = v >= 0 ? `rgba(124,196,255,${(0.08 + 0.55 * a).toFixed(3)})` : `rgba(255,122,89,${(0.08 + 0.55 * a).toFixed(3)})`; return `<td style="background:${c}">${fmt(v, 2)}</td>`; };
    const tab = M => `<div class="scroller"><table class="mtab">${M.map(r => `<tr>${r.map(cell).join('')}</tr>`).join('')}</table></div>`;
    let k = 1;
    const kS = slider(f.side, { label: '$k$', min: 1, max: 4, step: 1, value: k, fmt: v => String(v) }, v => { k = v; upd(); });
    const bars = el('div'); bars.style.cssText = 'display:flex;gap:8px;align-items:flex-end;height:96px;padding:4px 2px 0'; f.side.appendChild(bars);
    const ro = readout(f.side);
    function upd() {
      grid.innerHTML = `<div class="mt"><div>${T('A')}</div>${tab(A4)}</div>
        <div class="mt"><div>newest layer ${T(`\\sigma_${k}u_${k}v_${k}^T`)}, ${T(`\\sigma_${k}=${fmt(s[k - 1])}`)}</div>${tab(layer(k - 1))}</div>
        <div class="mt"><div>${T(`A_${k}`)} (layers ${T(`1..${k}`)})</div>${tab(Ak(k))}</div>
        <div class="mt"><div>still missing: ${T(`A-A_${k}`)}</div>${tab(L.msub(A4, Ak(k)))}</div>`;
      bars.innerHTML = s.map((x, i) => `<div style="flex:1;display:grid;justify-items:center;gap:3px"><div style="width:100%;height:${(x / s[0] * 64).toFixed(1)}px;border-radius:4px 4px 0 0;background:${i < k ? 'var(--eig)' : 'var(--line)'}"></div><span style="font:500 10.5px var(--mono);color:${i < k ? 'var(--ink)' : 'var(--faint)'}">σ${'₁₂₃₄'[i]}=${fmt(x)}</span></div>`).join('');
      const err = L.frob(L.msub(A4, Ak(k))), tail = s.slice(k).map(x => fmt(x * x)).join('+') || '0';
      ro(`<div class="line">${T(`\\|A-A_${k}\\|_F=${fmt(err, 3)}`)}</div>
        <div class="line">${T(`\\sqrt{${k < 4 ? s.slice(k).map((_, i) => `\\sigma_${k + i + 1}^2`).join('+') : '0'}}=\\sqrt{${tail}}=${fmt(Math.sqrt(s.slice(k).reduce((a, x) => a + x * x, 0)), 3)}`)}</div>
        <div class="flag ok">${k === 4 ? T('A_4=A') : 'equal: the error is exactly the discarded layers'}</div>`);
    }
    upd();
    return { reset() { k = 1; kS.set(1); upd(); }, setK(v) { k = v; kS.set(v); upd(); }, state: () => ({ s, k, A1: Ak(1), err: L.frob(L.msub(A4, Ak(k))) }) };
  },
};

export default {
  'grid-transform': gridTransform,
  'chain': chain,
  'solve': solve,
  'nullspace3d': nullspace3d,
  'eigen': eigen,
  'diagonalize': diagonalize,
  'norms': norms,
  'l1-corner': l1corner,
  'bowl': bowl,
  'svd2': svd2v,
  'svd-layers': svdLayers,
};
