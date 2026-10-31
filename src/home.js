// Course map: chapters and parts as a path, with progress and "continue where you left off".
import * as prog from './progress.js';

const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function main() {
  const course = await fetch('data/course.json').then(r => r.json());
  const parts = course.chapters.flatMap(c => c.parts);
  const tot = parts.reduce((s, p) => s + p.sections.length, 0);
  const read = parts.reduce((s, p) => s + Math.round(prog.partFrac(p) * p.sections.length), 0);
  document.getElementById('top-bar').style.width = (100 * read / tot).toFixed(1) + '%';
  document.getElementById('top-pct').textContent = `${read}/${tot} sections`;

  const last = prog.last(), lastPart = last && parts.find(p => p.id === last.pid && p.ready);
  const firstOpen = parts.find(p => p.ready && prog.partFrac(p) < 1) || parts.find(p => p.ready);
  const go = lastPart ? `part.html?p=${lastPart.id}#${last.sec}` : `part.html?p=${firstOpen.id}`;
  const goLabel = lastPart ? `Continue ${lastPart.id} ${esc(lastPart.title)}` : `Start with ${firstOpen.id} ${esc(firstOpen.title)}`;
  document.getElementById('stat').innerHTML = `<div><div class="big">${Math.round(100 * read / tot)}%</div><div class="lab">${read} of ${tot} sections read</div></div>
    <div><div class="big">${parts.filter(p => prog.partFrac(p) >= 1).length}<span style="color:var(--faint)">/${parts.length}</span></div><div class="lab">parts complete</div></div>
    <div><div class="big">${parts.reduce((s, p) => s + (p.visuals || 0), 0)}</div><div class="lab">interactive visuals</div></div>
    <a class="btn pri" href="${go}">${goLabel} →</a>`;

  document.getElementById('journey').innerHTML = course.chapters.map(c => `<section class="chap">
    <div class="ck">Chapter ${c.n}</div><h2>${c.n} · ${esc(c.title)}</h2><p>${esc(c.intro)}</p>
    <div class="pcards">${c.parts.map(p => {
      const f = prog.partFrac(p), tag = p.ready ? 'a' : 'div';
      return `<${tag} class="pcard${p.ready ? '' : ' soon'}"${p.ready ? ` href="part.html?p=${p.id}"` : ''}>
        <div class="pk"><span>PART ${p.id}</span>${p.ready ? `<span class="ring" style="--p:${Math.round(f * 100)}" title="${Math.round(f * 100)}% read"></span>` : '<span>soon</span>'}</div>
        <h3>${esc(p.title)}</h3>
        <ul>${p.sections.map(s => `<li class="${prog.isRead(prog.secId(s.n)) ? 'read' : ''}"><span>${s.n}</span>${esc(s.title)}</li>`).join('')}</ul>
        <div class="pm"><span>pp. ${p.page}–${p.end}</span><span>${p.visuals ? `${p.visuals} visuals` : ''}</span></div>
      </${tag}>`;
    }).join('')}</div></section>`).join('');
}
main();
