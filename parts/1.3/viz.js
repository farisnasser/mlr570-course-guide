// Part 1.3 Calculus & Optimization — interactive visuals on the notes' own functions:
// f(x) = x^2, f(x,y) = 3x^2 y, (f, g) = (3x^2 y, 2x + y^8), y = sin(x^2), y = x + x^2, L(w) = w^T w.
import { Plane, Chart, COL, FONT } from '../../src/viz/plane.js';
import * as L from '../../src/viz/la.js';
import { el, T, TD, fmt, matTex, vecTex, seg, slider, buttons, readout, legend } from '../../src/viz/ui.js';
import { loadThree, Scene3 } from '../../src/viz/three-lite.js';

const H = (f, h) => { f.stage.style.height = h; };
const MONO = '600 11.5px "JetBrains Mono", monospace';
function split(f, titles) {
  const sp = el('div', 'split'); f.stage.appendChild(sp);
  return titles.map(t => { const d = el('div'); sp.appendChild(d); if (t) d.appendChild(el('div', 'stage-t', t)); return d; });
}
const STOPS = [[0x1b, 0x1f, 0x4b], [0x24, 0x4f, 0x86], [0x2a, 0x9c, 0x96], [0x96, 0xd0, 0x78], [0xf4, 0xe8, 0xa4]];
const ramp = t => { t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1); const i = Math.min(STOPS.length - 2, Math.floor(t)), u = t - i, a = STOPS[i], b = STOPS[i + 1]; return [0, 1, 2].map(k => (a[k] + (b[k] - a[k]) * u) / 255); };

/** a surface z = F(x,y) on [-R,R]^2 drawn with display height scale hz; returns {mesh, update(F)} */
function surface(S, THREE, R, N, hz, colorRange) {
  const geo = new THREE.BufferGeometry(), pos = new Float32Array((N + 1) ** 2 * 3), col = new Float32Array((N + 1) ** 2 * 3), idx = [];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const a = i * (N + 1) + j; idx.push(a, a + N + 1, a + 1, a + N + 1, a + N + 2, a + 1); }
  geo.setIndex(idx); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide, shininess: 22, transparent: true, opacity: 0.88 }));
  S.scene.add(mesh);
  return { mesh, update(F) {
    let lo = Infinity, hi = -Infinity; const zs = [];
    for (let i = 0; i <= N; i++) for (let j = 0; j <= N; j++) { const z = F(-R + 2 * R * j / N, -R + 2 * R * i / N); zs.push(z); lo = Math.min(lo, z); hi = Math.max(hi, z); }
    const [c0, c1] = colorRange ?? [lo, hi];
    for (let i = 0, k = 0; i <= N; i++) for (let j = 0; j <= N; j++, k++) {
      const x = -R + 2 * R * j / N, y = -R + 2 * R * i / N, z = zs[k];
      pos.set([x, z * hz, -y], k * 3); col.set(ramp((z - c0) / ((c1 - c0) || 1)), k * 3);
    }
    geo.attributes.position.needsUpdate = true; geo.attributes.color.needsUpdate = true; geo.computeVertexNormals();
  } };
}

// ---------------------------------------------------------------------------------------------------------------
// 1.3.1 Secant -> tangent on f(x) = x^2
const secant = {
  title: 'The secant turns into the tangent as $h\\to0$',
  mount(f) {
    H(f, 'clamp(290px, 42vw, 360px)');
    let x = 3, h = 1;
    const fx = t => t * t, C = new Chart(f.stage, { x: [-0.5, 5], y: [-2, 26], xlabel: 'x', ylabel: 'f(x)', pad: [14, 14, 26, 36] });
    C.handle({ get: () => [x, fx(x)], set: p => { x = Math.max(0, Math.min(4.2, Math.round(p[0] * 20) / 20)); upd(); }, color: COL.ink });
    C.handle({ get: () => [x + h, fx(x + h)], set: p => { h = Math.max(-2.5, Math.min(2, Math.round((p[0] - x) * 100) / 100)); if (Math.abs(h) < 0.01) h = 0.01; hS.set(h); upd(); }, color: COL.c2 });
    C.render = () => {
      C.fn(fx, COL.nul, { lw: 2.4 });
      const sl = (fx(x + h) - fx(x)) / h, d = 2 * x;
      C.fn(t => fx(x) + sl * (t - x), 'rgba(232,236,248,.75)', { lw: 1.6, dash: [6, 4] });
      C.fn(t => fx(x) + d * (t - x), COL.span, { lw: 2 });
      C.seg([x, 0], [x, fx(x)], 'rgba(232,236,248,.3)', { dash: [3, 3] }); C.seg([x + h, 0], [x + h, fx(x + h)], 'rgba(255,122,89,.4)', { dash: [3, 3] });
      C.dot([x, fx(x)], 5, COL.ink); C.dot([x + h, fx(x + h)], 5, COL.c2);
      C.label([x, 0], 'x', COL.ink, { dy: -10, dx: -8, font: FONT.math }); C.label([x + h, 0], 'x+h', COL.c2, { dy: -10, dx: 6, font: FONT.math });
    };
    const hS = slider(f.side, { label: '$h$', min: -2.5, max: 2, step: 0.01, value: h, fmt: v => v.toFixed(2) }, v => { h = Math.abs(v) < 0.01 ? 0.01 : v; upd(); });
    buttons(f.side, [{ label: 'shrink $h\\to0$', pri: true, on: () => { const h0 = h; let k = 0; const step = () => { k++; h = h0 * Math.pow(0.9, k); hS.set(h); upd(); if (Math.abs(h) > 0.001) requestAnimationFrame(step); }; requestAnimationFrame(step); } }]);
    legend(f.side, [['rgba(232,236,248,.75)', 'secant'], [COL.span, 'tangent: slope $f\'(x)$']]);
    const ro = readout(f.side);
    function upd() {
      const sl = (fx(x + h) - fx(x)) / h;
      ro(`<div class="line">${T(`x=${fmt(x, 2)},\\ h=${fmt(h, 4)}`)}</div>
        <div class="line">secant slope ${T(`\\frac{f(x+h)-f(x)}{h}=${fmt(sl, 4)}`)}</div>
        <div class="line">tangent slope ${T(`f'(x)=2x=${fmt(2 * x, 4)}`)}</div>
        <div class="line">${T(`\\Delta x=0.01`)}: ${T(`f'(x)\\,\\Delta x=${fmt(2 * x * 0.01, 4)}`)}, exact ${T(`f(x+0.01)-f(x)=${fmt(fx(x + 0.01) - fx(x), 4)}`)}</div>`);
      C.draw();
    }
    upd();
    return { reset() { x = 3; h = 1; hS.set(1); upd(); }, state: () => ({ x, h, slope: (fx(x + h) - fx(x)) / h }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.3.2 Partial derivatives as slices of f(x,y) = 3x^2 y; gradient perpendicular to contours
const partials = {
  title: 'Partial derivatives and the gradient of $f(x,y)=3x^2y$',
  async mount(f) {
    H(f, 'clamp(320px, 50vw, 430px)');
    const THREE = await loadThree();
    const [hl, hr] = split(f, [`${T('f(x,y)=3x^2y')}`, 'contours and $\\nabla f$'.replace('$\\nabla f$', T('\\nabla f'))]);
    const F = (x, y) => 3 * x * x * y, R = 1.5, hz = 0.085;
    let p = [1, 0.5];
    const S = new Scene3(hl, { cam: [5.4, -6.4, 3.9], target: [0, 0, 0], minDist: 3 });
    const sf = surface(S, THREE, R, 60, hz); sf.update(F);
    S.axes(1.9);
    for (const [q, t] of [[[2.1, 0, 0], 'x'], [[0, 2.1, 0], 'y']]) { const l = S.label(t); l.p = q; l.e.style.color = COL.muted; }
    const cx = S.line([[0, 0, 0]], 0xff6b6b), cy = S.line([[0, 0, 0]], 0x7cc4ff), tx = S.line([[0, 0, 0]], 0xff6b6b, { opacity: 0.9 }), ty = S.line([[0, 0, 0]], 0x7cc4ff, { opacity: 0.9 });
    const ball = S.ball(0xe8ecf8, 0.06);
    const C = new Plane(hr, { extent: 1.6 });
    C.handle({ get: () => p, set: q => { p = q.map(v => Math.max(-R, Math.min(R, Math.round(v * 20) / 20))); upd(); }, color: COL.ink });
    C.render = () => {
      // contours of f on a grid (marching squares, levels at multiples of 0.5)
      const g = C.ctx, n = 120, lv = []; for (let k = -10; k <= 10; k++) if (k) lv.push(k * 0.5);
      const val = (i, j) => F(-R + 2 * R * i / n, -R + 2 * R * j / n);
      g.lineWidth = 1;
      for (const L0 of lv) {
        g.strokeStyle = L0 > 0 ? 'rgba(124,196,255,.45)' : 'rgba(255,122,89,.45)'; g.beginPath();
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          const c = [val(i, j), val(i + 1, j), val(i + 1, j + 1), val(i, j + 1)], P = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], pts = [];
          for (let e = 0; e < 4; e++) { const a = c[e] - L0, b = c[(e + 1) % 4] - L0; if ((a < 0) !== (b < 0)) { const t = a / (a - b), A = P[e], B = P[(e + 1) % 4]; pts.push([-R + 2 * R * (A[0] + t * (B[0] - A[0])) / n, -R + 2 * R * (A[1] + t * (B[1] - A[1])) / n]); } }
          if (pts.length >= 2) { g.moveTo(...C.P(pts[0])); g.lineTo(...C.P(pts[1])); }
        }
        g.stroke();
      }
      const gr = [6 * p[0] * p[1], 3 * p[0] * p[0]], n2 = L.norm(gr);
      if (n2 > 1e-9) {   // direction is exact; the drawn length is capped so the arrow stays on screen
        const d = L.scale(gr, (0.2 + 0.45 * Math.min(1, n2 / 6)) / n2);
        C.arrow(p, L.add(p, d), COL.eig, { lw: 3 }); C.arrow(p, L.sub(p, d), COL.c1, { lw: 2, dash: [4, 3] });
        C.label(L.add(p, d), '∇f', COL.eig, { dx: 8, font: FONT.math }); C.label(L.sub(p, d), '−∇f', COL.c1, { dx: 8, font: FONT.math });
      }
      C.dot(p, 4, COL.ink);
    };
    legend(f.side, [['#ff6b6b', '$y$ fixed: slope $\\partial f/\\partial x=6xy$'], [COL.span, '$x$ fixed: slope $\\partial f/\\partial y=3x^2$'], [COL.eig, '$\\nabla f$']]);
    const ro = readout(f.side);
    function upd() {
      const [x, y] = p, fxv = 6 * x * y, fyv = 3 * x * x, N = 60;
      S.setLine(cx, Array.from({ length: N + 1 }, (_, i) => { const t = -R + 2 * R * i / N; return [t, y, F(t, y) * hz + 0.01]; }));
      S.setLine(cy, Array.from({ length: N + 1 }, (_, i) => { const t = -R + 2 * R * i / N; return [x, t, F(x, t) * hz + 0.01]; }));
      S.setLine(tx, [-0.45, 0.45].map(d => [x + d, y, (F(x, y) + fxv * d) * hz + 0.02]));
      S.setLine(ty, [-0.45, 0.45].map(d => [x, y + d, (F(x, y) + fyv * d) * hz + 0.02]));
      ball.set([x, y, F(x, y) * hz + 0.03]);
      ro(`<div class="line">${T(`(x,y)=(${fmt(x)},\\ ${fmt(y)}),\\ f=${fmt(F(x, y), 3)}`)}</div>
        <div class="line">${T(`\\frac{\\partial f}{\\partial x}=6xy=${fmt(fxv, 3)}`)}</div>
        <div class="line">${T(`\\frac{\\partial f}{\\partial y}=3x^2=${fmt(fyv, 3)}`)}</div>
        <div class="line">${T(`\\nabla f=[\\,${fmt(fxv, 3)},\\ ${fmt(fyv, 3)}\\,]`)}</div>`);
      S.draw(); C.draw();
    }
    upd();
    return { reset() { p = [1, 0.5]; upd(); }, state: () => ({ p, grad: [6 * p[0] * p[1], 3 * p[0] ** 2] }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.3.3 The Jacobian of (3x^2 y, 2x + y^8) as the local linear map
const jacobian = {
  title: 'The Jacobian is the best linear stand-in near a point',
  mount(f) {
    H(f, 'clamp(300px, 46vw, 400px)');
    const [hl, hr] = split(f, [`input ${T('(x,y)')}`, `output ${T('(f,g)')}`]);
    const map = ([x, y]) => [3 * x * x * y, 2 * x + y ** 8], Jac = ([x, y]) => [[6 * x * y, 3 * x * x], [2, 8 * y ** 7]];
    let p = [1, 0.8], e = 0.3;
    const A = new Plane(hl, { extent: 1.6, center: [0.6, 0.4] }), B = new Plane(hr, { extent: 3, center: [2.4, 2] });
    A.handle({ get: () => p, set: q => { p = q.map(v => Math.round(v * 20) / 20); p[1] = Math.max(-1.05, Math.min(1.05, p[1])); upd(); }, color: COL.ink, limit: 2 });
    const square = () => { const pts = []; for (let k = 0; k <= 80; k++) { const t = k / 20, s = t % 1, side = Math.floor(t) % 4; const c = [[-1, -1], [1, -1], [1, 1], [-1, 1]]; const a = c[side], b = c[(side + 1) % 4]; pts.push([p[0] + e * (a[0] + (b[0] - a[0]) * s), p[1] + e * (a[1] + (b[1] - a[1]) * s)]); } return pts; };
    A.render = () => {
      for (let k = -4; k <= 4; k++) { A.seg([p[0] + k * e / 4, p[1] - e], [p[0] + k * e / 4, p[1] + e], 'rgba(124,196,255,.25)', { lw: 1 }); A.seg([p[0] - e, p[1] + k * e / 4], [p[0] + e, p[1] + k * e / 4], 'rgba(124,196,255,.25)', { lw: 1 }); }
      A.path(square(), COL.span, { lw: 2, close: true }); A.dot(p, 4, COL.ink);
      A.arrow(p, [p[0] + e, p[1]], COL.c1, { lw: 2, head: 8 }); A.arrow(p, [p[0], p[1] + e], COL.c2, { lw: 2, head: 8 });
    };
    B.render = () => {
      const q = map(p), J = Jac(p);
      for (let k = -4; k <= 4; k++) {
        B.path(Array.from({ length: 21 }, (_, i) => map([p[0] + k * e / 4, p[1] - e + 2 * e * i / 20])), 'rgba(124,196,255,.25)', { lw: 1 });
        B.path(Array.from({ length: 21 }, (_, i) => map([p[0] - e + 2 * e * i / 20, p[1] + k * e / 4])), 'rgba(124,196,255,.25)', { lw: 1 });
      }
      B.path(square().map(map), COL.span, { lw: 2, close: true });
      const lin = s => L.add(q, L.mv(J, L.sub(s, p)));
      B.path(square().map(lin), COL.eig, { lw: 2, dash: [6, 4], close: true });
      B.dot(q, 4, COL.ink);
      B.arrow(q, L.add(q, L.scale([J[0][0], J[1][0]], e)), COL.c1, { lw: 2, head: 8 }); B.arrow(q, L.add(q, L.scale([J[0][1], J[1][1]], e)), COL.c2, { lw: 2, head: 8 });
    };
    const eS = slider(f.side, { label: 'square size', min: 0.02, max: 0.5, step: 0.01, value: e, fmt: v => v.toFixed(2) }, v => { e = v; upd(); });
    legend(f.side, [[COL.span, 'true image under $(f,g)$'], [COL.eig, 'image under $J$ (dashed)'], [COL.c1, '$J$ column 1'], [COL.c2, '$J$ column 2']]);
    const ro = readout(f.side);
    function upd() {
      const J = Jac(p);
      // the largest gap between the true image and the linear one, along the square's boundary
      let gap = 0; const q = map(p); for (const s of square()) gap = Math.max(gap, L.norm(L.sub(map(s), L.add(q, L.mv(J, L.sub(s, p))))));
      ro(`<div class="line">${T(`(x,y)=(${fmt(p[0])},\\ ${fmt(p[1])})`)}</div>
        <div class="eqx">${TD(`J=\\begin{bmatrix}6xy&3x^2\\\\2&8y^7\\end{bmatrix}=${matTex(J, 3)}`)}</div>
        <div class="line">largest gap between the two shapes: ${T(fmt(gap, 4))} (square half-size ${T(fmt(e, 2))})</div>`);
      A.draw(); B.draw();
    }
    upd();
    return { reset() { p = [1, 0.8]; e = 0.3; eS.set(e); upd(); }, state: () => ({ p, J: Jac(p) }) };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.3.4 Chain rule / total derivative on a computation graph
const chain = {
  title: 'Chain rule on a computation graph: multiply along a path, add the paths',
  mount(f) {
    f.stage.style.minHeight = '0';
    let mode = 'sin', x = 1.2;
    const box = el('div'); box.style.cssText = 'padding:12px'; f.stage.appendChild(box);
    const node = (X, Y, label, val, c) => `<g transform="translate(${X},${Y})"><rect x="-62" y="-22" width="124" height="44" rx="9" fill="#12172c" stroke="${c}" stroke-width="1.6"/><text y="-3" text-anchor="middle" fill="#e8ecf8" font-size="13" font-style="italic" font-family="Source Serif 4, serif">${label}</text><text y="14" text-anchor="middle" fill="#8e99bd" font-size="11" font-family="JetBrains Mono, monospace">${val}</text></g>`;
    const edge = (x1, y1, x2, y2, c, fwd, bwd, curve = 0) => {
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 + curve;
      return `<path d="M${x1},${y1} Q${mx},${my} ${x2},${y2}" fill="none" stroke="${c}" stroke-width="1.8" marker-end="url(#cg-ar)"/><text x="${mx}" y="${my - 8 + (curve > 0 ? 18 : 0)}" text-anchor="middle" font-size="11.5" font-family="JetBrains Mono, monospace" fill="${c}">${fwd}</text><text x="${mx}" y="${my + 20 + (curve > 0 ? 14 : 0)}" text-anchor="middle" font-size="10.5" font-family="JetBrains Mono, monospace" fill="#ffc857">${bwd}</text>`;
    };
    seg(f.side, [{ label: '$y=\\sin(x^2)$', value: 'sin' }, { label: '$y=x+x^2$', value: 'sum' }], v => { mode = v; upd(); }, mode);
    const xS = slider(f.side, { label: '$x$', min: -2, max: 2, step: 0.05, value: x, fmt: v => v.toFixed(2) }, v => { x = v; upd(); });
    legend(f.side, [['#e8ecf8', 'value (forward)'], [COL.eig, 'local derivative (backward)']]);
    const ro = readout(f.side);
    function upd() {
      const u = x * x;
      let svg, dy, yv, fn, msg;
      if (mode === 'sin') {
        yv = Math.sin(u); dy = 2 * x * Math.cos(u); fn = t => Math.sin(t * t);
        svg = `<svg viewBox="0 0 600 120" style="width:100%;max-width:640px;display:block;margin:0 auto" role="img" aria-label="Graph x to u = x squared to y = sin u">
          <defs><marker id="cg-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#8e99bd"/></marker></defs>
          ${edge(132, 55, 238, 55, '#8e99bd', '', `du/dx=2x=${fmt(2 * x, 3)}`)}${edge(362, 55, 468, 55, '#8e99bd', '', `dy/du=cos u=${fmt(Math.cos(u), 3)}`)}
          ${node(70, 55, 'x', fmt(x, 3), '#b28cff')}${node(300, 55, 'u = x²', fmt(u, 3), '#7cc4ff')}${node(530, 55, 'y = sin u', fmt(yv, 3), '#5fe0b0')}</svg>`;
        msg = (`<div class="line">${T(`\\frac{dy}{dx}=\\frac{dy}{du}\\frac{du}{dx}=${fmt(Math.cos(u), 4)}\\times${fmt(2 * x, 4)}=${fmt(dy, 4)}`)}</div>
          <div class="line">${T(`2x\\cos(x^2)=${fmt(dy, 4)}`)}</div>`);
      } else {
        yv = x + u; dy = 1 + 2 * x; fn = t => t + t * t;
        svg = `<svg viewBox="0 0 600 150" style="width:100%;max-width:640px;display:block;margin:0 auto" role="img" aria-label="Graph x to y directly and through u = x squared">
          <defs><marker id="cg-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#8e99bd"/></marker></defs>
          ${edge(132, 40, 468, 40, '#ff6b6b', 'direct', '∂y/∂x = 1')}
          ${edge(120, 62, 240, 108, '#8e99bd', '', `du/dx=2x=${fmt(2 * x, 3)}`, 10)}${edge(362, 108, 480, 62, '#8e99bd', '', '∂y/∂u = 1', 10)}
          ${node(70, 40, 'x', fmt(x, 3), '#b28cff')}${node(300, 112, 'u = x²', fmt(u, 3), '#7cc4ff')}${node(530, 40, 'y = x + u', fmt(yv, 3), '#5fe0b0')}</svg>`;
        msg = (`<div class="line">direct:${T('\\frac{\\partial y}{\\partial x}=1')}</div><div class="line">through ${T('u')}: ${T(`\\frac{\\partial y}{\\partial u}\\frac{du}{dx}=(1)(${fmt(2 * x, 3)})=${fmt(2 * x, 3)}`)}</div>
          <div class="line">${T(`\\frac{dy}{dx}=1+2x=${fmt(dy, 4)}`)}</div>`);
      }
      box.innerHTML = svg;
      const hN = 1e-6, fd = (fn(x + hN) - fn(x - hN)) / (2 * hN);
      ro(msg + `<div class="line" style="color:var(--muted)">nudge check: ${T(`\\frac{y(x+10^{-6})-y(x-10^{-6})}{2\\cdot10^{-6}}=${fmt(fd, 4)}`)}</div>`);
    }
    upd();
    return { reset() { mode = 'sin'; x = 1.2; xS.set(x); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.3.6 Hessian: min / max / saddle for f = 1/2 x^T H x
const hessian = {
  title: 'The Hessian decides: minimum, maximum or saddle',
  async mount(f) {
    H(f, 'clamp(320px, 48vw, 420px)');
    const THREE = await loadThree();
    let a = 1, b = 0, c = -1;
    const S = new Scene3(f.stage, { cam: [5.6, -6.6, 4.8], target: [0, 0, 0], minDist: 3 });
    const R = 1.6, hz = 0.55, sf = surface(S, THREE, R, 60, hz, [-2.6, 2.6]);
    S.axes(2);
    const e1 = S.line([[0, 0, 0]], 0x7cc4ff), e2 = S.line([[0, 0, 0]], 0x7cc4ff);
    S.ball(0xe8ecf8, 0.06).set([0, 0, 0.02]);
    const sl = [
      slider(f.side, { label: '$H_{11}$', min: -2, max: 2, step: 0.1, value: a, fmt: v => v.toFixed(1) }, v => { a = v; upd(); }),
      slider(f.side, { label: '$H_{12}=H_{21}$', min: -2, max: 2, step: 0.1, value: b, fmt: v => v.toFixed(1) }, v => { b = v; upd(); }),
      slider(f.side, { label: '$H_{22}$', min: -2, max: 2, step: 0.1, value: c, fmt: v => v.toFixed(1) }, v => { c = v; upd(); }),
    ];
    const pre = seg(f.side, [{ label: '$H\\succ0$', value: 'min' }, { label: '$H\\prec0$', value: 'max' }, { label: 'mixed', value: 'saddle' }], v => { [a, b, c] = { min: [1.2, 0.4, 0.8], max: [-1.2, 0.4, -0.8], saddle: [1, 0, -1] }[v]; sl[0].set(a); sl[1].set(b); sl[2].set(c); upd(); }, 'saddle');
    const ro = readout(f.side);
    function upd() {
      const Hm = [[a, b], [b, c]], F = (x, y) => 0.5 * (a * x * x + 2 * b * x * y + c * y * y), e = L.eig2(Hm);
      sf.update(F);
      [e1, e2].forEach((ln, i) => {
        const v = e.vecs[i], col = e.vals[i] > 1e-9 ? 0x7cc4ff : e.vals[i] < -1e-9 ? 0xff5d73 : 0xffc857;
        ln.material.color.setHex(col);
        S.setLine(ln, Array.from({ length: 41 }, (_, k) => { const t = -R + 2 * R * k / 40; return [v[0] * t, v[1] * t, F(v[0] * t, v[1] * t) * hz + 0.02]; }));
      });
      const kind = e.vals[1] > 1e-9 ? ['local minimum', 'ok', 'H\\succ0'] : e.vals[0] < -1e-9 ? ['local maximum', 'bad', 'H\\prec0'] : e.vals[0] > 1e-9 && e.vals[1] < -1e-9 ? ['saddle point', 'eig', '\\text{mixed}'] : ['flat direction: the test is inconclusive', '', '\\text{a zero eigenvalue}'];
      ro(`<div class="eqx">${TD(`H=${matTex(Hm, 1)}`)}</div><div class="line">eigenvalues ${T(`${fmt(e.vals[0], 3)},\\ ${fmt(e.vals[1], 3)}`)}</div><div class="flag ${kind[1]}">${T(kind[2])}: ${kind[0]}</div>`);
      S.draw();
    }
    upd();
    return { reset() { [a, b, c] = [1, 0, -1]; sl[0].set(a); sl[1].set(b); sl[2].set(c); pre.set('saddle'); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.3.6 Convexity: chord above the graph; graph above every tangent
const convexity = {
  title: 'Convexity: the chord stays above, the tangent stays below',
  mount(f) {
    H(f, 'clamp(290px, 42vw, 370px)');
    const FN = { convex: { f: x => x * x, d: x => 2 * x, y: [-0.8, 5], lab: 'f(x) = x²' }, non: { f: x => 0.25 * x ** 4 - x * x + 0.3 * x, d: x => x ** 3 - 2 * x + 0.3, y: [-1.6, 2.4], lab: 'f(x) = x⁴/4 − x² + 0.3x' } };
    let kind = 'convex', xa = -1.5, xb = 1.2, lam = 0.5;
    const C = new Chart(f.stage, { x: [-2.2, 2.2], y: FN.convex.y, xlabel: 'x', pad: [14, 14, 26, 36] });
    const fn = () => FN[kind];
    C.handle({ get: () => [xa, fn().f(xa)], set: p => { xa = p[0]; upd(); }, color: COL.c1 });
    C.handle({ get: () => [xb, fn().f(xb)], set: p => { xb = p[0]; upd(); }, color: COL.c2 });
    C.render = () => {
      const F = fn(), m = lam * xa + (1 - lam) * xb, ch = lam * F.f(xa) + (1 - lam) * F.f(xb);
      C.fn(F.f, COL.span, { lw: 2.4 });
      C.fn(x => F.f(xa) + F.d(xa) * (x - xa), 'rgba(255,200,87,.7)', { lw: 1.5, dash: [5, 4] });
      C.seg([xa, F.f(xa)], [xb, F.f(xb)], COL.nul, { lw: 2 });
      C.seg([m, F.f(m)], [m, ch], ch >= F.f(m) - 1e-12 ? COL.c1 : COL.nul, { lw: 3 });
      C.dot([m, ch], 4.5, COL.nul); C.dot([m, F.f(m)], 4.5, COL.span);
      C.label([-2.1, F.y[1]], F.lab, COL.span, { dy: 12, font: FONT.math });
    };
    seg(f.side, [{ label: 'convex $x^2$', value: 'convex' }, { label: 'non-convex', value: 'non' }], v => { kind = v; C.o.y = FN[v].y; xa = v === 'non' ? -1.4 : -1.5; xb = v === 'non' ? 1.4 : 1.2; upd(); }, kind);
    const lS = slider(f.side, { label: '$\\lambda$', min: 0, max: 1, step: 0.01, value: lam, fmt: v => v.toFixed(2) }, v => { lam = v; upd(); });
    legend(f.side, [[COL.nul, 'chord'], ['rgba(255,200,87,.8)', 'tangent at $\\mathbf{x}$']]);
    const ro = readout(f.side);
    function upd() {
      const F = fn(), m = lam * xa + (1 - lam) * xb, ch = lam * F.f(xa) + (1 - lam) * F.f(xb), ok = F.f(m) <= ch + 1e-12;
      // first-order test over a grid of y
      let worst = Infinity; for (let i = 0; i <= 200; i++) { const y = -2.2 + 4.4 * i / 200; worst = Math.min(worst, F.f(y) - (F.f(xa) + F.d(xa) * (y - xa))); }
      ro(`<div class="line">${T(`f(\\lambda x+(1-\\lambda)y)=${fmt(F.f(m), 3)}`)}</div><div class="line">${T(`\\lambda f(x)+(1-\\lambda)f(y)=${fmt(ch, 3)}`)}</div>
        <div class="flag ${ok ? 'ok' : 'bad'}">${ok ? 'chord above the graph here' : 'the chord dips below the graph: not convex'}</div>
        <div class="flag ${worst >= -1e-12 ? 'ok' : 'bad'}">${worst >= -1e-12 ? 'graph above the tangent at ' + T('x') + ' everywhere' : 'graph falls below the tangent at ' + T('x')}</div>`);
      C.draw();
    }
    upd();
    return { reset() { kind = 'convex'; C.o.y = FN.convex.y; xa = -1.5; xb = 1.2; lam = 0.5; lS.set(lam); upd(); } };
  },
};

// ---------------------------------------------------------------------------------------------------------------
// 1.3.6 Gradient descent on L(w) = w^T w: w_{t+1} = (1 - 2 eta) w_t
const gd = {
  title: 'Gradient descent on $L(\\mathbf{w})=\\mathbf{w}^T\\mathbf{w}$ and the step size $\\eta$',
  mount(f) {
    H(f, 'clamp(300px, 44vw, 380px)');
    const [hl, hr] = split(f, [`one weight: ${T('L(w)=w^2')}`, `two weights: contours of ${T('L')}`]);
    let eta = 0.15, steps = 8;
    const w0 = [2.2, 1.4];
    const A = new Chart(hl, { x: [-3.5, 3.5], y: [-0.5, 12], xlabel: 'w', pad: [14, 12, 26, 32] });
    const B = new Plane(hr, { extent: 3.4 });
    const iter = (w, k) => Array.from({ length: k + 1 }, (_, t) => w.map(v => v * (1 - 2 * eta) ** t));
    A.render = () => {
      A.fn(w => w * w, COL.nul, { lw: 2.2 });
      const it = iter([w0[0]], steps).map(([w]) => w).filter(w => Math.abs(w) < 3.5);
      for (let t = 0; t + 1 < it.length; t++) A.arrow([it[t], it[t] ** 2], [it[t + 1], it[t + 1] ** 2], COL.span, { lw: 1.8, head: 8 });
      it.forEach((w, t) => A.dot([w, w * w], t ? 3.5 : 5, t ? COL.span : COL.ink));
    };
    B.render = () => {
      for (const r of [0.5, 1, 1.5, 2, 2.5, 3]) B.path(Array.from({ length: 97 }, (_, i) => [r * Math.cos(i * Math.PI / 48), r * Math.sin(i * Math.PI / 48)]), 'rgba(255,107,107,.32)', { lw: 1 });
      const it = iter(w0, steps).filter(w => L.norm(w) < 4.5);
      for (let t = 0; t + 1 < it.length; t++) B.arrow(it[t], it[t + 1], COL.span, { lw: 2, head: 9 });
      it.forEach((w, t) => B.dot(w, t ? 3.5 : 5, t ? COL.span : COL.ink));
    };
    const eS = slider(f.side, { label: '$\\eta$', min: 0.02, max: 1.15, step: 0.01, value: eta, fmt: v => v.toFixed(2) }, v => { eta = v; upd(); });
    const sS = slider(f.side, { label: 'steps', min: 1, max: 25, step: 1, value: steps, fmt: v => String(v) }, v => { steps = v; upd(); });
    buttons(f.side, [0.05, 0.25, 0.75, 1.05].map(v => ({ label: `$\\eta=${v}$`, on: () => { eta = v; eS.set(v); upd(); } })));
    const ro = readout(f.side);
    function upd() {
      const k = 1 - 2 * eta, w = iter(w0, steps)[steps];
      const regime = Math.abs(k) < 1 ? (k >= 0 ? ['shrinks toward 0 without changing sign', 'ok'] : ['flips sign each step but still shrinks', 'eig']) : Math.abs(k) === 1 ? ['bounces with the same size', 'eig'] : ['flips sign and grows: overshooting', 'bad'];
      ro(`<div class="line">${T(`\\mathbf w_{t+1}=(1-2\\eta)\\,\\mathbf w_t=${fmt(k, 3)}\\,\\mathbf w_t`)}</div>
        <div class="flag ${regime[1]}">${T(`|1-2\\eta|=${fmt(Math.abs(k), 3)}`)}: ${regime[0]}</div>
        <div class="line">after ${steps} steps ${T(`\\mathbf w=(${fmt(w[0], 4)},\\ ${fmt(w[1], 4)})`)}, ${T(`L=${fmt(L.dot(w, w), 4)}`)}</div>`);
      A.draw(); B.draw();
    }
    upd();
    return { reset() { eta = 0.15; steps = 8; eS.set(eta); sS.set(steps); upd(); }, state: () => ({ eta, k: 1 - 2 * eta }) };
  },
};

export default { 'secant': secant, 'partials': partials, 'jacobian': jacobian, 'chain': chain, 'hessian': hessian, 'convexity': convexity, 'gd': gd };
