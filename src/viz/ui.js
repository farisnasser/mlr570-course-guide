// Controls and readouts shared by the visuals. Labels may contain $..$ math (rendered with KaTeX).
import { inline, tex } from '../notes.js';

export function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
export const T = s => tex(s, false);
export const TD = s => tex(s, true);

/** number formatting: trims trailing zeros, never shows -0 */
export function fmt(x, d = 2) {
  if (!isFinite(x)) return x > 0 ? '\\infty' : '-\\infty';
  let s = (Math.abs(x) < 0.5 * 10 ** -d ? 0 : x).toFixed(d);
  if (s.includes('.')) s = s.replace(/0+$/, '').replace(/\.$/, '');
  return s === '-0' ? '0' : s;
}
export const matTex = (M, d = 2) => `\\begin{bmatrix}${M.map(r => r.map(v => fmt(v, d)).join('&')).join('\\\\')}\\end{bmatrix}`;
export const vecTex = (v, d = 2) => `\\begin{bmatrix}${v.map(x => fmt(x, d)).join('\\\\')}\\end{bmatrix}`;

/** segmented buttons: items [{label, value}], returns {set(value)} */
export function seg(host, items, onPick, value) {
  const box = el('div', 'seg'); host.appendChild(box);
  const btns = items.map(it => {
    const b = el('button', '', inline(it.label)); b.type = 'button';
    b.addEventListener('click', () => { set(it.value); onPick(it.value); });
    box.appendChild(b); return [it.value, b];
  });
  const set = v => btns.forEach(([val, b]) => b.setAttribute('aria-pressed', String(val === v)));
  set(value);
  return { set, el: box };
}

/** range slider with a math label and a live value */
export function slider(host, o, onInput) {
  const row = el('label', 'sl');
  row.innerHTML = `<span>${inline(o.label)}</span><input type="range" min="${o.min}" max="${o.max}" step="${o.step}" value="${o.value}"><output></output>`;
  host.appendChild(row);
  const inp = row.querySelector('input'), out = row.querySelector('output');
  const show = () => { out.textContent = o.fmt ? o.fmt(+inp.value) : inp.value; };
  inp.addEventListener('input', () => { show(); onInput(+inp.value); });
  show();
  return { set(v) { inp.value = v; show(); }, get: () => +inp.value, el: row, input: inp };
}

export function buttons(host, items) {
  const box = el('div', 'acts'); host.appendChild(box);
  return items.map(it => { const b = el('button', it.pri ? 'pri' : '', inline(it.label)); b.type = 'button'; b.addEventListener('click', it.on); box.appendChild(b); return b; });
}

/** a readout block that only re-renders when its HTML changes */
export function readout(host, title) {
  if (title) host.appendChild(el('h5', '', title));
  const box = el('div', 'ro'); host.appendChild(box);
  let last = '';
  return html => { if (html !== last) { box.innerHTML = html; last = html; } };
}

export function legend(host, items) {
  const box = el('div', 'legend', items.map(([c, t]) => `<span><i style="background:${c}"></i>${inline(t)}</span>`).join(''));
  host.appendChild(box); return box;
}
