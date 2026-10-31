// Reading progress, kept in this browser only (localStorage). A section counts as read once its end scrolls into view.
const KEY = 'mlr570-guide:v1';
let S;
function load() {
  if (S) return S;
  try { S = JSON.parse(localStorage.getItem(KEY)) || {}; } catch { S = {}; }
  S.read ??= {}; S.done ??= {};
  return S;
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch { /* private mode: progress lives for this visit */ } }

export const isRead = id => !!load().read[id];
export function markRead(id) { if (!load().read[id]) { S.read[id] = Date.now(); save(); return true; } return false; }
export const isDone = pid => !!load().done[pid];
export function setDone(pid, on) { load(); if (on) S.done[pid] = Date.now(); else delete S.done[pid]; save(); }
export function setLast(pid, sec) { load().last = { pid, sec, t: Date.now() }; save(); }
export const last = () => load().last;
export const secId = n => 's-' + n.replace(/\./g, '-');

/** fraction of a part's sections read (a part marked complete counts as 1) */
export function partFrac(part) {
  if (isDone(part.id)) return 1;
  const n = part.sections.length; if (!n) return 0;
  return part.sections.filter(s => isRead(secId(s.n))).length / n;
}
