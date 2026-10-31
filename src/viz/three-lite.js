// three-lite — the small three.js layer for the 3D visuals (three@0.128.0 + OrbitControls as globals, loaded lazily).
// Math convention: world (x, y, z) with z up. three.js is y-up, so a math point (x, y, z) is placed at (x, z, -y).

const URLS = ['https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js',
  'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'];
let loading = null;
function script(src) { return new Promise((ok, bad) => { const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = bad; document.head.appendChild(s); }); }
export function loadThree() {
  loading ??= (async () => { if (!window.THREE) await script(URLS[0]); if (!window.THREE.OrbitControls) await script(URLS[1]); return window.THREE; })();
  return loading;
}

export const W = (p) => new window.THREE.Vector3(p[0], p[2], -p[1]);

export class Scene3 {
  constructor(host, o = {}) {
    const THREE = window.THREE;
    this.host = host; this.o = Object.assign({ cam: [5, 4, 6], target: [0, 0, 0], fov: 40 }, o);
    this.gl = document.createElement('div'); this.gl.className = 'gl'; host.appendChild(this.gl);
    this.lbls = document.createElement('div'); this.lbls.className = 'lbls'; host.appendChild(this.lbls);
    this.r = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.gl.appendChild(this.r.domElement);
    this.scene = new THREE.Scene();
    this.cam = new THREE.PerspectiveCamera(this.o.fov, 1, 0.05, 200);
    this.cam.position.copy(W(this.o.cam));
    this.ctl = new THREE.OrbitControls(this.cam, this.r.domElement);
    this.ctl.target.copy(W(this.o.target)); this.ctl.enableDamping = true; this.ctl.dampingFactor = 0.12; this.ctl.enablePan = false;
    this.ctl.minDistance = this.o.minDist ?? 3; this.ctl.maxDistance = this.o.maxDist ?? 30;
    this.ctl.addEventListener('change', () => this.draw());
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const dl = new THREE.DirectionalLight(0xffffff, 0.55); dl.position.set(4, 9, 6); this.scene.add(dl);
    this.labels = []; this.onFrame = null; this._raf = 0; this._anim = false;
    this.ro = new ResizeObserver(() => requestAnimationFrame(() => this.resize())); this.ro.observe(host); this.resize();
  }
  resize() {
    const { width: w, height: h } = this.host.getBoundingClientRect(); if (!w || !h) return;
    this.w = w; this.h = h; this.r.setSize(w, h); this.cam.aspect = w / h; this.cam.updateProjectionMatrix(); this.draw();
  }
  draw() { if (!this._raf) this._raf = requestAnimationFrame(() => { this._raf = 0; this.frame(); }); }
  frame() {
    const moving = this.ctl.update();
    this.onFrame?.();
    this.r.render(this.scene, this.cam);
    this.placeLabels();
    if (moving || this._anim) this.draw();
  }
  /** keep drawing every frame while on (for animations) */
  animate(on) { this._anim = on; if (on) this.draw(); }
  label(html, cls = '') { const e = document.createElement('div'); e.className = 'lb ' + cls; e.innerHTML = html; this.lbls.appendChild(e); const L = { e, p: [0, 0, 0], on: true }; this.labels.push(L); return L; }
  placeLabels() {
    const THREE = window.THREE, v = new THREE.Vector3();
    for (const L of this.labels) {
      if (!L.on) { L.e.style.display = 'none'; continue; }
      v.copy(W(L.p)).project(this.cam);
      if (v.z > 1) { L.e.style.display = 'none'; continue; }
      L.e.style.display = '';
      const x = (v.x + 1) / 2 * this.w, y = (1 - v.y) / 2 * this.h;
      L.e.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%) translate(${L.dx ?? 0}px, ${L.dy ?? 0}px)`;
    }
  }
  /** fly the camera to a math-space position (and target) */
  fly(pos, target, ms = 900) {
    const THREE = window.THREE, p0 = this.cam.position.clone(), t0 = this.ctl.target.clone(), p1 = W(pos), t1 = W(target ?? this.o.target);
    const start = performance.now(), ease = t => t < .5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
    const step = () => {
      const t = Math.min(1, (performance.now() - start) / ms), e = ease(t);
      this.cam.position.lerpVectors(p0, p1, e); this.ctl.target.lerpVectors(t0, t1, e); this.draw();
      if (t < 1) requestAnimationFrame(step);
    };
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { this.cam.position.copy(p1); this.ctl.target.copy(t1); this.draw(); }
    else requestAnimationFrame(step);
  }

  // ---------- objects (math coordinates) ----------
  line(pts, color, o = {}) {
    const THREE = window.THREE, g = new THREE.BufferGeometry().setFromPoints(pts.map(W));
    const m = o.dash ? new THREE.LineDashedMaterial({ color, dashSize: o.dash[0], gapSize: o.dash[1], transparent: true, opacity: o.opacity ?? 1 })
      : new THREE.LineBasicMaterial({ color, transparent: true, opacity: o.opacity ?? 1 });
    const L = new THREE.Line(g, m); if (o.dash) L.computeLineDistances(); this.scene.add(L); return L;
  }
  setLine(L, pts) { L.geometry.setFromPoints(pts.map(W)); if (L.material.isLineDashedMaterial) L.computeLineDistances(); }
  arrow(color, o = {}) {
    const THREE = window.THREE, grp = new THREE.Group();
    const mat = new THREE.MeshPhongMaterial({ color, shininess: 40 });
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(o.r ?? 0.035, o.r ?? 0.035, 1, 12), mat);
    const head = new THREE.Mesh(new THREE.ConeGeometry((o.r ?? 0.035) * 3, o.head ?? 0.26, 18), mat);
    grp.add(shaft, head); this.scene.add(grp);
    const A = { grp, shaft, head, set: (a, b) => {
      const va = W(a), vb = W(b), d = vb.clone().sub(va), L = d.length();
      grp.visible = L > 1e-6; if (!grp.visible) return;
      const hl = Math.min(o.head ?? 0.26, L * 0.5), dir = d.clone().normalize();
      shaft.scale.set(1, Math.max(1e-4, L - hl), 1); shaft.position.copy(va).addScaledVector(dir, (L - hl) / 2);
      head.position.copy(va).addScaledVector(dir, L - hl / 2);
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      shaft.quaternion.copy(q); head.quaternion.copy(q);
    } };
    return A;
  }
  ball(color, r = 0.08) {
    const THREE = window.THREE, m = new THREE.Mesh(new THREE.SphereGeometry(r, 20, 14), new THREE.MeshPhongMaterial({ color, shininess: 60 }));
    this.scene.add(m); return { m, set: p => m.position.copy(W(p)) };
  }
  /** a flat quad spanned by u, v (math vectors) through the origin, half-size s */
  planeSpan(u, v, color, s = 2.5, opacity = 0.18) {
    const THREE = window.THREE, g = new THREE.BufferGeometry();
    const P = (a, b) => W([u[0] * a + v[0] * b, u[1] * a + v[1] * b, u[2] * a + v[2] * b]);
    const c = [P(-s, -s), P(s, -s), P(s, s), P(-s, s)];
    g.setFromPoints([c[0], c[1], c[2], c[0], c[2], c[3]]); g.computeVertexNormals();
    const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }));
    this.scene.add(m); return m;
  }
  /** axes through the origin, length a */
  axes(a = 3, color = 0x8e99bd) { for (const d of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) this.line([d.map(x => -x * a), d.map(x => x * a)], color, { opacity: 0.45 }); }
  destroy() { this.ro.disconnect(); this.r.dispose(); }
}
