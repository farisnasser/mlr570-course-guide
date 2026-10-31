// plane — a 2D canvas stage for the visuals: world <-> screen mapping, transformed grids, arrows, labels and
// draggable handles (mouse, pen and touch; a touch only blocks page scrolling when it starts on a handle).
// Renders on demand: call plane.draw() after changing state; the paint callback is plane.render(ctx, plane).

const DPR = () => Math.min(2, window.devicePixelRatio || 1);
export const COL = {
  c1: '#5fe0b0', c2: '#ff7a59', span: '#7cc4ff', nul: '#ff5d73', eig: '#ffc857', free: '#b28cff',
  ink: '#e8ecf8', muted: '#8e99bd', faint: '#5d6890', grid: '#7cc4ff', bg: '#0f1324',
};
export const FONT = { math: 'italic 600 15px "Source Serif 4", Georgia, serif', small: '500 11px "JetBrains Mono", monospace', ui: '600 12px Sora, system-ui, sans-serif' };

export class Plane {
  constructor(host, o = {}) {
    this.o = Object.assign({ extent: 4, center: [0, 0], bgGrid: true, step: 1 }, o);
    this.host = host;
    this.c = document.createElement('canvas');
    this.c.className = 'pan';
    this.c.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block';
    host.appendChild(this.c);
    this.ctx = this.c.getContext('2d');
    this.handles = []; this.drag = null; this.render = null; this.onDrag = null;
    this.w = 0; this.h = 0;
    this._raf = 0;
    this.ro = new ResizeObserver(() => requestAnimationFrame(() => this.resize())); this.ro.observe(host);
    this._bind();
    this.resize();
  }
  resize() {
    const r = this.host.getBoundingClientRect(), d = DPR();
    if (!r.width || !r.height) return;
    this.w = r.width; this.h = r.height;
    this.c.width = Math.round(r.width * d); this.c.height = Math.round(r.height * d);
    this.ctx.setTransform(d, 0, 0, d, 0, 0);
    this.paint();
  }
  get s() { return Math.min(this.w, this.h) / (2 * this.o.extent); }
  X(x) { return this.w / 2 + (x - this.o.center[0]) * this.s; }
  Y(y) { return this.h / 2 - (y - this.o.center[1]) * this.s; }
  P(p) { return [this.X(p[0]), this.Y(p[1])]; }
  inv(px, py) { return [(px - this.w / 2) / this.s + this.o.center[0], -(py - this.h / 2) / this.s + this.o.center[1]]; }
  /** visible world box */
  box() { const [x0, y1] = this.inv(0, 0), [x1, y0] = this.inv(this.w, this.h); return { x0, x1, y0, y1 }; }

  draw() { if (!this._raf) this._raf = requestAnimationFrame(() => { this._raf = 0; this.paint(); }); }
  paint() {
    if (!this.w) return;
    const g = this.ctx; g.clearRect(0, 0, this.w, this.h);
    if (this.o.bgGrid) this.bgGrid();
    this.render?.(g, this);
    for (const h of this.handles) if (h.visible?.() !== false && h.ring !== false) this.handleRing(h);
  }

  // ---------- primitives ----------
  bgGrid() {
    const g = this.ctx, { x0, x1, y0, y1 } = this.box(), st = this.o.step;
    g.lineWidth = 1; g.strokeStyle = 'rgba(124,196,255,.07)'; g.beginPath();
    for (let x = Math.ceil(x0 / st) * st; x <= x1; x += st) { g.moveTo(this.X(x), 0); g.lineTo(this.X(x), this.h); }
    for (let y = Math.ceil(y0 / st) * st; y <= y1; y += st) { g.moveTo(0, this.Y(y)); g.lineTo(this.w, this.Y(y)); }
    g.stroke();
    g.strokeStyle = 'rgba(232,236,248,.22)'; g.beginPath();
    g.moveTo(this.X(0), 0); g.lineTo(this.X(0), this.h); g.moveTo(0, this.Y(0)); g.lineTo(this.w, this.Y(0)); g.stroke();
  }
  /** grid lines x = k, y = k mapped through M (2x2 rows), clipped to |k| <= n */
  gridT(M, o = {}) {
    const g = this.ctx, n = o.n ?? Math.ceil(this.o.extent * 2.4), st = o.step ?? 1, E = n * st;
    const map = (x, y) => [M[0][0] * x + M[0][1] * y, M[1][0] * x + M[1][1] * y];
    g.lineWidth = o.lw ?? 1; g.strokeStyle = o.color ?? 'rgba(124,196,255,.38)';
    g.beginPath();
    for (let k = -n; k <= n; k++) {
      const v = k * st;
      let a = map(v, -E), b = map(v, E); g.moveTo(...this.P(a)); g.lineTo(...this.P(b));
      a = map(-E, v); b = map(E, v); g.moveTo(...this.P(a)); g.lineTo(...this.P(b));
    }
    g.stroke();
    if (o.axes !== false) {
      g.lineWidth = (o.lw ?? 1) + 0.8; g.strokeStyle = o.axisColor ?? 'rgba(232,236,248,.55)';
      g.beginPath();
      g.moveTo(...this.P(map(0, -E))); g.lineTo(...this.P(map(0, E)));
      g.moveTo(...this.P(map(-E, 0))); g.lineTo(...this.P(map(E, 0)));
      g.stroke();
    }
  }
  /** infinite line through point p along direction d */
  lineDir(p, d, color, o = {}) {
    const n = Math.hypot(...d); if (n < 1e-12) return;
    const L = this.o.extent * 6 / n, g = this.ctx;
    g.save(); g.strokeStyle = color; g.lineWidth = o.lw ?? 2; g.globalAlpha = o.alpha ?? 1; if (o.dash) g.setLineDash(o.dash);
    g.beginPath(); g.moveTo(this.X(p[0] - d[0] * L), this.Y(p[1] - d[1] * L)); g.lineTo(this.X(p[0] + d[0] * L), this.Y(p[1] + d[1] * L)); g.stroke(); g.restore();
  }
  seg(a, b, color, o = {}) {
    const g = this.ctx; g.save(); g.strokeStyle = color; g.lineWidth = o.lw ?? 1.5; g.globalAlpha = o.alpha ?? 1; if (o.dash) g.setLineDash(o.dash);
    g.beginPath(); g.moveTo(...this.P(a)); g.lineTo(...this.P(b)); g.stroke(); g.restore();
  }
  arrow(a, b, color, o = {}) {
    const g = this.ctx, [x0, y0] = this.P(a), [x1, y1] = this.P(b), dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy);
    const lw = o.lw ?? 3, hd = Math.min(o.head ?? 12, L * 0.45);
    g.save(); g.globalAlpha = o.alpha ?? 1; g.strokeStyle = g.fillStyle = color; g.lineWidth = lw; g.lineCap = 'round';
    if (o.dash) g.setLineDash(o.dash);
    if (L > 0.5) {
      const ux = dx / L, uy = dy / L;
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1 - ux * hd * 0.8, y1 - uy * hd * 0.8); g.stroke();
      g.setLineDash([]);
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - ux * hd - uy * hd * 0.5, y1 - uy * hd + ux * hd * 0.5);
      g.lineTo(x1 - ux * hd + uy * hd * 0.5, y1 - uy * hd - ux * hd * 0.5); g.closePath(); g.fill();
    } else { g.beginPath(); g.arc(x0, y0, lw * 1.2, 0, 7); g.fill(); }
    g.restore();
  }
  poly(pts, fill, stroke, o = {}) {
    const g = this.ctx; g.save(); g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, ...this.P(p))); g.closePath();
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = o.lw ?? 1.5; if (o.dash) g.setLineDash(o.dash); g.stroke(); }
    g.restore();
  }
  path(pts, color, o = {}) {
    const g = this.ctx; g.save(); g.strokeStyle = color; g.lineWidth = o.lw ?? 2; g.globalAlpha = o.alpha ?? 1; g.lineJoin = 'round';
    if (o.dash) g.setLineDash(o.dash);
    g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, ...this.P(p))); if (o.close) g.closePath();
    if (o.fill) { g.fillStyle = o.fill; g.fill(); }
    g.stroke(); g.restore();
  }
  dot(p, r, color, o = {}) {
    const g = this.ctx; g.save(); g.globalAlpha = o.alpha ?? 1; g.beginPath(); g.arc(...this.P(p), r, 0, 7);
    if (o.ring) { g.strokeStyle = color; g.lineWidth = o.lw ?? 2; g.stroke(); } else { g.fillStyle = color; g.fill(); }
    g.restore();
  }
  label(p, text, color, o = {}) {
    const g = this.ctx; let [x, y] = this.P(p); x += o.dx ?? 0; y += o.dy ?? 0;
    g.save(); g.font = o.font ?? FONT.math; g.textAlign = o.align ?? 'left'; g.textBaseline = o.base ?? 'middle';
    g.lineWidth = 4; g.strokeStyle = 'rgba(11,14,26,.9)'; g.lineJoin = 'round';
    g.strokeText(text, x, y); g.fillStyle = color; g.fillText(text, x, y); g.restore();
  }
  /** label placed just beyond the tip of vector v (pushed outward along v) */
  tipLabel(v, text, color, o = {}) {
    const n = Math.hypot(...v) || 1, k = (o.gap ?? 16) / this.s;
    this.label([v[0] + v[0] / n * k, v[1] + v[1] / n * k], text, color, { align: 'center', ...o });
  }
  handleRing(h) {
    const [x, y] = this.P(h.get()), g = this.ctx, act = this.drag === h || this.hover === h;
    g.save(); g.beginPath(); g.arc(x, y, act ? 11 : 9, 0, 7); g.strokeStyle = h.color; g.lineWidth = 2; g.globalAlpha = act ? 1 : .8; g.stroke();
    g.beginPath(); g.arc(x, y, 3.2, 0, 7); g.fillStyle = h.color; g.fill(); g.restore();
  }

  // ---------- interaction ----------
  /** h: {get:()=>[x,y], set:([x,y])=>void, color, snap?:number, visible?:()=>bool, ring?:false} */
  handle(h) { this.handles.push(h); return h; }
  _hit(px, py, slop) {
    let best = null, bd = slop;
    for (const h of this.handles) {
      if (h.visible?.() === false) continue;
      const [x, y] = this.P(h.get()), d = Math.hypot(px - x, py - y);
      if (d < bd) { bd = d; best = h; }
    }
    return best;
  }
  _pos(e) { const r = this.c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  _bind() {
    const c = this.c;
    c.addEventListener('touchstart', e => {
      const t = e.touches[0], r = c.getBoundingClientRect();
      if (this._hit(t.clientX - r.left, t.clientY - r.top, 30)) e.preventDefault();
    }, { passive: false });
    c.addEventListener('pointerdown', e => {
      const [px, py] = this._pos(e), h = this._hit(px, py, e.pointerType === 'touch' ? 30 : 18);
      if (!h) return;
      this.drag = h; c.setPointerCapture(e.pointerId); e.preventDefault(); this._move(px, py);
    });
    c.addEventListener('pointermove', e => {
      const [px, py] = this._pos(e);
      if (this.drag) { this._move(px, py); return; }
      const h = this._hit(px, py, 18); if (h !== this.hover) { this.hover = h; c.style.cursor = h ? 'grab' : ''; this.draw(); }
    });
    const up = () => { if (this.drag) { this.drag = null; this.onRelease?.(); this.draw(); } };
    c.addEventListener('pointerup', up); c.addEventListener('pointercancel', up);
    c.addEventListener('pointerleave', () => { if (this.hover && !this.drag) { this.hover = null; this.draw(); } });
  }
  _move(px, py) {
    let p = this.inv(px, py);
    const lim = this.drag.limit ?? this.o.extent * 1.6; p = p.map(v => Math.max(-lim, Math.min(lim, v)));
    const sn = this.drag.snap; if (sn) p = p.map(v => Math.abs(v / sn - Math.round(v / sn)) < 0.22 ? Math.round(v / sn) * sn : v);
    this.drag.set(p); this.onDrag?.(this.drag); this.draw();
  }
  destroy() { this.ro.disconnect(); cancelAnimationFrame(this._raf); }
}

/** requestAnimationFrame tween: calls f(t) for t in [0,1] over ms, eased; returns a cancel fn. Instant when reduced motion. */
export const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches;
export function tween(ms, f, done) {
  if (REDUCE || ms <= 0) { f(1); done?.(); return () => {}; }
  let t0 = null, id = 0, stop = false;
  const ease = t => t < .5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
  const step = ts => { if (stop) return; t0 ??= ts; const t = Math.min(1, (ts - t0) / ms); f(ease(t)); if (t < 1) id = requestAnimationFrame(step); else done?.(); };
  id = requestAnimationFrame(step);
  return () => { stop = true; cancelAnimationFrame(id); };
}
