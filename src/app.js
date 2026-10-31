// Part page: part.html?p=1.1 — chapter/part header, the notes (verbatim), the PDF's figures, the interactive visuals,
// the journey rail with per-section progress, and prev/next.
import { renderNotes, setXref, inline } from './notes.js';
import * as prog from './progress.js';

const LOCAL = /^(localhost|127\.0\.0\.1)$/.test(location.hostname);
const G = window.__guide = { errors: [], viz: {}, mounted: [] };
if (LOCAL) {
  addEventListener('error', e => { if (!/ResizeObserver loop/.test(e.message)) G.errors.push(String(e.message || e)); });
  addEventListener('unhandledrejection', e => G.errors.push(String(e.reason?.stack || e.reason)));
}

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function katexReady() { for (let i = 0; i < 200 && !window.katex; i++) await new Promise(r => setTimeout(r, 25)); }

function flatParts(course) { return course.chapters.flatMap(c => c.parts.map(p => ({ ...p, ch: c }))); }

function rail(course, cur) {
  const parts = flatParts(course);
  const html = course.chapters.map(c => `<h3>Chapter ${c.n} · ${esc(c.title)}</h3><ol>${c.parts.map(p => {
    const f = prog.partFrac(p), isCur = p.id === cur.id;
    const cls = ['rp', isCur ? 'cur' : '', !p.ready ? 'soon' : '', f >= 1 ? 'done' : f > 0 ? 'partial' : ''].join(' ');
    const secs = isCur ? `<div class="secs">${p.sections.map(s => `<a href="#${prog.secId(s.n)}" data-s="${prog.secId(s.n)}" class="${prog.isRead(prog.secId(s.n)) ? 'read' : ''}"><span class="sn">${s.n}</span><span>${esc(s.title)}</span></a>`).join('')}</div>` : '';
    return `<li class="${cls}" style="--p:${Math.round(f * 100)}%"><span class="dot"></span><a href="${p.ready ? `part.html?p=${p.id}` : '#'}"${p.ready ? '' : ' aria-disabled="true" tabindex="-1"'}><span class="pn">${p.id}</span><span>${esc(p.title)}${p.ready ? '' : ' <small>· soon</small>'}</span></a>${secs}</li>`;
  }).join('')}</ol>`).join('');
  $('#rail').innerHTML = html + `<p class="rail-foot">${parts.filter(p => p.ready).length} of ${parts.length} parts on the site. Progress is saved in this browser.<br><a href="index.html">Course map</a> · <a href="reference/MLR570_Course_Notes.pdf#page=${cur.page}" target="_blank" rel="noopener">PDF p.&nbsp;${cur.page}</a></p>`;
}

function topProgress(course) {
  const parts = flatParts(course), tot = parts.reduce((s, p) => s + p.sections.length, 0);
  const read = parts.reduce((s, p) => s + Math.round(prog.partFrac(p) * p.sections.length), 0);
  $('#top-bar').style.width = (100 * read / tot).toFixed(1) + '%';
  $('#top-pct').textContent = `${read}/${tot} sections`;
}

async function inlineFigs(root, pid) {
  await Promise.all([...root.querySelectorAll('figure.pdffig')].map(async f => {
    try {
      const r = await fetch(`parts/${pid}/fig/${f.dataset.fig}.svg`);
      if (!r.ok) throw new Error(r.status);
      f.innerHTML = await r.text();
      const svg = f.querySelector('svg'), w = +svg.getAttribute('width');
      svg.removeAttribute('width'); svg.removeAttribute('height');
      svg.style.width = (w * 1.42).toFixed(0) + 'px';
    } catch (e) { G.errors.push(`figure ${f.dataset.fig}: ${e}`); f.textContent = `[figure ${f.dataset.fig} missing]`; }
  }));
}

function frame(fig, def) {
  const cap = fig.dataset.cap;
  fig.innerHTML = `<div class="viz-h"><span class="viz-tag">Interactive</span><b>${inline(def.title)}</b><button class="reset" type="button">Reset</button></div>
    <div class="viz-body"><div class="viz-stage"></div><div class="viz-side"></div></div>${cap ? `<figcaption>${inline(cap)}</figcaption>` : ''}`;
  const body = fig.querySelector('.viz-body'), stage = fig.querySelector('.viz-stage'), side = fig.querySelector('.viz-side');
  if (def.height) stage.style.height = typeof def.height === 'function' ? def.height() : def.height;
  return { fig, body, stage, side, reset: fig.querySelector('.reset') };
}

async function mountVisuals(root, pid) {
  const figs = [...root.querySelectorAll('figure.viz')]; if (!figs.length) return;
  let mod;
  try { mod = (await import(`../parts/${pid}/viz.js`)).default; } catch (e) { G.errors.push('viz.js: ' + (e.stack || e)); return; }
  const mount = async fig => {
    const id = fig.dataset.viz, def = mod[id];
    if (!def) { G.errors.push('no visual ' + id); fig.textContent = `[visual ${id} missing]`; return; }
    const f = frame(fig, def);
    try {
      const api = (await def.mount(f)) || {};
      G.viz[id] = api; G.mounted.push(id);
      f.reset.addEventListener('click', () => api.reset?.());
      if (!api.reset) f.reset.hidden = true;
    } catch (e) { G.errors.push(`${id}: ${e.stack || e}`); fig.querySelector('.viz-stage').textContent = 'This visual failed to load.'; }
  };
  // mount each visual shortly before it scrolls into view (several use WebGL)
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { io.unobserve(e.target); mount(e.target); } }), { rootMargin: '600px 0px' });
  figs.forEach(f => io.observe(f));
  G.mountAll = () => Promise.all(figs.filter(f => !f.querySelector('.viz-h')).map(f => { io.unobserve(f); return mount(f); }));
}

function spy(course, part) {
  const links = new Map([...document.querySelectorAll('.secs a')].map(a => [a.dataset.s, a]));
  const heads = [...document.querySelectorAll('.nt-h')];
  let curId = null;
  const onScroll = () => {
    let id = heads[0]?.id;
    for (const h of heads) if (h.getBoundingClientRect().top < 120) id = h.id;
    if (id !== curId) {
      curId = id; links.forEach((a, k) => a.classList.toggle('on', k === id));
      const a = links.get(id); if (a && window.innerWidth >= 1100) a.scrollIntoView({ block: 'nearest' });
      prog.setLast(part.id, id);
    }
    const doc = document.documentElement, f = doc.scrollTop / Math.max(1, doc.scrollHeight - innerHeight);
    $('#read-prog').style.width = (100 * Math.min(1, f)).toFixed(1) + '%';
  };
  addEventListener('scroll', onScroll, { passive: true }); onScroll();
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    const id = e.target.dataset.sec;
    if (prog.markRead(id)) { links.get(id)?.classList.add('read'); topProgress(course); rail(course, part); spyRebind(); }
  }), { rootMargin: '0px 0px -10% 0px' });
  document.querySelectorAll('.sec-end').forEach(s => io.observe(s));
  function spyRebind() { const fresh = new Map([...document.querySelectorAll('.secs a')].map(a => [a.dataset.s, a])); fresh.forEach((a, k) => { links.set(k, a); a.classList.toggle('on', k === curId); }); }
}

async function main() {
  const pid = new URLSearchParams(location.search).get('p') || '1.1';
  const [course, notes] = await Promise.all([
    fetch('data/course.json').then(r => r.json()),
    fetch(`parts/${pid}/notes.md`).then(r => r.ok ? r.text() : null),
  ]);
  await katexReady();
  const parts = flatParts(course), i = parts.findIndex(p => p.id === pid), part = parts[i];
  if (!part || notes == null) { $('#notes').innerHTML = `<p>Part ${esc(pid)} is not on the site yet. <a href="index.html">Back to the course map</a>.</p>`; return; }
  document.title = `${part.id} ${part.title} · MLR 570 Course Guide`;
  $('#crumb').innerHTML = `Chapter ${part.ch.n} · <b>Part ${part.id} ${esc(part.title)}</b>`;

  // section / part references in the text link to their place in the guide
  const secPage = new Map(parts.flatMap(p => p.sections.map(s => [s.n, p])));
  setXref(n => {
    const p = secPage.get(n) || parts.find(q => q.id === n);
    if (!p || !p.ready) return null;
    const hash = p.id === n ? '' : '#' + prog.secId(n);
    return p.id === pid ? hash || '#' : `part.html?p=${p.id}${hash}`;
  });
  const first = part.ch.parts[0].id === part.id;
  const nViz = (notes.match(/^:::viz /gm) || []).length;
  $('#head').innerHTML = (first ? `<div class="ch-ban"><div class="k">Chapter ${part.ch.n}</div><h1>${part.ch.n} · ${esc(part.ch.title)}</h1><p>${inline(part.ch.intro)}</p></div>` : '') +
    `<div class="part-h"><div class="k">Part ${part.id}</div><h2>${esc(part.title)}</h2><div class="meta"><span>PDF pp. ${part.page}–${part.end}</span><span>${part.sections.length} sections</span>${nViz ? `<span>${nViz} interactive visuals</span>` : ''}<a href="reference/MLR570_Course_Notes.pdf#page=${part.page}" target="_blank" rel="noopener">open the PDF</a></div></div>`;

  const ctx = { secs: [], viz: [], page: n => part.sections.find(s => s.n === n)?.page };
  const root = $('#notes');
  root.innerHTML = renderNotes(notes, ctx);
  G.sections = ctx.secs;

  const prev = parts[i - 1], next = parts[i + 1];
  const card = (p, cls, k) => p ? (p.ready ? `<a class="${cls}" href="part.html?p=${p.id}"><small>${k}</small><b>${p.id} ${esc(p.title)}</b></a>` : `<a class="${cls} soon" aria-disabled="true"><small>${k} · soon</small><b>${p.id} ${esc(p.title)}</b></a>`) : '<span></span>';
  $('#pnav').innerHTML = card(prev, 'prev', '← Previous part') + card(next, 'next', 'Next part →');
  const doneBtn = $('#done-btn');
  const syncDone = () => { doneBtn.textContent = prog.isDone(part.id) ? '✓ Part complete' : 'Mark part complete'; doneBtn.classList.toggle('pri', !prog.isDone(part.id)); };
  doneBtn.addEventListener('click', () => { prog.setDone(part.id, !prog.isDone(part.id)); syncDone(); rail(course, part); topProgress(course); });
  syncDone();
  $('#done').hidden = false;

  rail(course, part); topProgress(course);
  await inlineFigs(root, pid);
  if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  spy(course, part);
  mountVisuals(root, pid);
  G.ready = true;
}

// mobile rail drawer
document.addEventListener('click', e => {
  if (e.target.closest('#rail-btn')) { document.body.classList.add('rail-open'); $('#scrim').hidden = false; }
  else if (e.target.closest('#scrim') || (e.target.closest('#rail a') && innerWidth < 1100)) { document.body.classList.remove('rail-open'); $('#scrim').hidden = true; }
});
main().catch(e => { G.errors.push(String(e.stack || e)); $('#notes').textContent = 'Could not load this part: ' + e; });
