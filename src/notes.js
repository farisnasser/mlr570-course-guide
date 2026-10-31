// notes.md -> HTML. One file per part (parts/<id>/notes.md); the notes are transcribed verbatim from the PDF.
//
// Syntax (adapted from the Survey Atlas notes renderer):
//   ## 1.1.1 Title          numbered section (anchor s-1-1-1, rail entry, PDF page chip via course.json)
//   ### Title               run-in sub-heading (the PDF's bold paragraph heads)
//   $$ ... $$               display math (may span lines)                      $..$  inline math
//   **bold**  *italic*  `code`  [text](url)      §1.2.7-style references become links automatically
//   - item / 1. item        lists; continuation lines (incl. $$ blocks) indented 3 spaces
//   | a | b |               tables (header, --- row). First line `|{stack}` / `|{cells}` / `|{ruled}` sets a class.
//   :::def Title            red box   (key definitions / results; title runs into the first paragraph)
//   :::ml Title             blue box  (why it matters in ML)
//   :::add Title            amber box (additions / clarifications not in the original notes)
//   :::ex Title             grey example box with a header bar
//   :::model Title          teal-headed box (the PDF's "Mental model" summaries)
//   :::plain                bordered box (e.g. "The flow")
//   :::cap                  a figure caption from the PDF (small, centred)
//   :::big Title            red box, centred large text ("Big picture")
//   :::cols [even|figleft]  side-by-side columns; :::col ... ::: inside (stack on phones)
//   :::fig <id>             the PDF's own figure, parts/<id>/fig/<id>.svg (exact vector extract)
//   :::viz <id> | caption   an interactive visual from parts/<id>/viz.js (body ignored)
//   > text                  source / footnote line
// Fences nest; every opener needs its own closing :::.
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export const tex = (t, display) => {
  try { return window.katex.renderToString(t, { displayMode: !!display, throwOnError: false, strict: false, trust: false }); }
  catch { return esc(t); }
};

let XREF = n => null;   // set by the page: section/part number -> href
export const setXref = f => { XREF = f; };

export function inline(s) {
  const math = [];
  s = s.replace(/\$([^$\n]+?)\$/g, (_, t) => `\u0000${math.push(t) - 1}\u0000`);
  s = esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*/g, '$1<i>$2</i>')
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/§(\d+\.\d+(?:\.\d+)?)((?:[–-]\d+\.\d+(?:\.\d+)?)?)/g, (m, a) => {
      const h = XREF(a); return h ? `<a class="xref" href="${h}">${m}</a>` : m;
    });
  s = s.replace(/&lt;br&gt;/g, '<br>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => tex(math[+i], false));
}

function takeFence(lines, i) {
  const body = []; let depth = 1;
  for (i++; i < lines.length; i++) {
    const l = lines[i].trim();
    if (/^:::\S/.test(l)) depth++;
    else if (l === ':::' && --depth === 0) return [body, i + 1];
    body.push(lines[i]);
  }
  return [body, i];
}

function table(rows) {
  let cls = '';
  if (/^\|\{(\w+(?:\s+\w+)*)\}\s*$/.test(rows[0].trim())) { cls = rows[0].trim().slice(2, -1); rows = rows.slice(1); }
  const cells = r => r.trim().replace(/^\||\|$/g, '').split(/(?<!\\)\|/).map(c => c.trim());
  const head = cells(rows[0]), aligns = cells(rows[1]).map(c => c.endsWith(':') ? (c.startsWith(':') ? 'center' : 'right') : 'left');
  const bare = head.every(c => !c);
  const label = j => esc((head[j] || '').replace(/\$|\*\*/g, ''));
  const td = (tag, c, j) => `<${tag}${aligns[j] !== 'left' ? ` style="text-align:${aligns[j]}"` : ''}${tag === 'td' && !bare ? ` data-label="${label(j)}"` : ''}>${inline(c)}</${tag}>`;
  return `<div class="nt-table ${cls}${bare ? ' bare' : ''}"><table>${bare ? '' : `<thead><tr>${head.map((c, j) => td('th', c, j)).join('')}</tr></thead>`}<tbody>${
    rows.slice(2).map(r => `<tr>${cells(r).map((c, j) => td('td', c, j)).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function list(lines, ordered) {
  const items = []; const marker = ordered ? /^\d+\.\s+/ : /^[-*]\s+/;
  for (const l of lines) {
    if (marker.test(l)) items.push([l.replace(marker, '')]);
    else items[items.length - 1].push(l.replace(/^ {1,3}/, ''));
  }
  const tag = ordered ? 'ol' : 'ul';
  return `<${tag}>${items.map(it => `<li>${blocks(it)}</li>`).join('')}</${tag}>`;
}

const BOX = { def: 'bx-def', ml: 'bx-ml', add: 'bx-add', ex: 'bx-ex', model: 'bx-ex bx-model', plain: 'bx-plain', big: 'bx-def bx-big' };

function box(kind, title, body, ctx) {
  let inner = blocks(body, ctx);
  if (kind === 'ex' || kind === 'model') return `<div class="bx ${BOX[kind]}">${title ? `<div class="bx-hd">${inline(title)}</div>` : ''}<div class="bx-b">${inner}</div></div>`;
  let t = '';
  if (title) {
    // red and "big" boxes run their title into the first paragraph, like the PDF; blue / amber put it on its own line
    if ((kind === 'def' || kind === 'big') && inner.startsWith('<p>')) inner = `<p><span class="bx-t inl">${inline(title)}</span>` + inner.slice(3);
    else t = `<div class="bx-t">${inline(title)}</div>`;
  }
  return `<div class="bx ${BOX[kind]}"><div class="bx-b">${t}${inner}</div></div>`;
}

export function blocks(lines, ctx = {}) {
  const out = []; let i = 0;
  while (i < lines.length) {
    const raw = lines[i], l = raw.trim();
    if (!l) { i++; continue; }
    let m;
    if ((m = l.match(/^(#{2,3})\s+(.*)$/))) {
      if (m[1] === '##') {
        const mm = m[2].match(/^(\d+(?:\.\d+)+)\s+(.*)$/);
        const num = mm ? mm[1] : '', title = mm ? mm[2] : m[2], id = 's-' + num.replace(/\./g, '-');
        ctx.secs?.push({ id, num, title });
        if (ctx.secs && ctx.secs.length > 1) out.push(`<div class="sec-end" data-sec="${ctx.secs[ctx.secs.length - 2].id}"></div>`);
        const pg = ctx.page?.(num);
        out.push(`<h3 class="nt-h" id="${id}"><span class="num">${num}</span><span>${inline(title)}</span>${pg ? `<span class="pg">p.&nbsp;${pg}</span>` : ''}</h3>`);
      } else out.push(`<h4 class="nt-h4">${inline(m[2])}</h4>`);
      i++; continue;
    }
    if (l.startsWith('$$')) {
      let t = l.slice(2);
      if (t.endsWith('$$') && t.length >= 2) { out.push(`<div class="nt-eq">${tex(t.slice(0, -2), true)}</div>`); i++; continue; }
      const buf = [t];
      for (i++; i < lines.length && !lines[i].trim().endsWith('$$'); i++) buf.push(lines[i]);
      buf.push((lines[i] || '').trim().slice(0, -2)); i++;
      out.push(`<div class="nt-eq">${tex(buf.join('\n'), true)}</div>`); continue;
    }
    if ((m = l.match(/^:::(\w+)\s*(.*)$/))) {
      const [body, next] = takeFence(lines, i); i = next;
      const [kind, arg] = [m[1], m[2].trim()];
      if (kind === 'cols') out.push(`<div class="nt-cols ${esc(arg)}">${blocks(body, ctx)}</div>`);
      else if (kind === 'col') out.push(`<div class="nt-col">${blocks(body, ctx)}</div>`);
      else if (kind === 'cap') out.push(`<div class="nt-cap">${blocks(body, ctx)}</div>`);
      else if (kind === 'fig') out.push(`<figure class="pdffig" data-fig="${esc(arg)}"></figure>`);
      else if (kind === 'viz') {
        const k = arg.search(/\s\|\s/), id = k < 0 ? arg : arg.slice(0, k).trim(), cap = k < 0 ? '' : arg.slice(k + 3).trim();
        ctx.viz?.push(id);
        out.push(`<figure class="viz" data-viz="${esc(id)}" data-cap="${esc(cap || '')}"></figure>`);
      }
      else if (BOX[kind]) out.push(box(kind, arg, body, ctx));
      else out.push(blocks(body, ctx));
      continue;
    }
    if (l.startsWith('|')) {
      const rows = []; for (; i < lines.length && lines[i].trim().startsWith('|'); i++) rows.push(lines[i]);
      out.push(table(rows)); continue;
    }
    const ord = /^\d+\.\s+/.test(l), un = /^[-*]\s+/.test(l);
    if (ord || un) {
      const buf = [];
      for (; i < lines.length; i++) {
        const x = lines[i];
        if (!x.trim()) { if (i + 1 < lines.length && /^ {3}|^\d+\.\s|^[-*]\s/.test(lines[i + 1])) { buf.push(''); continue; } break; }
        if (buf.length && !/^ {3}/.test(x) && !(ord ? /^\d+\.\s+/ : /^[-*]\s+/).test(x.trim())) break;
        buf.push(x);
      }
      out.push(list(buf, ord)); continue;
    }
    if (l.startsWith('>')) {
      const buf = []; for (; i < lines.length && lines[i].trim().startsWith('>'); i++) buf.push(lines[i].trim().replace(/^>\s?/, ''));
      out.push(`<p class="nt-src">${inline(buf.join(' '))}</p>`); continue;
    }
    const buf = [];
    for (; i < lines.length; i++) {
      const x = lines[i].trim();
      if (!x || /^(#{2,3}\s|\$\$|:::|\||>|[-*]\s|\d+\.\s)/.test(x)) break;
      // a line ending in two spaces or a backslash is a hard break (the PDF's line-separated statements)
      buf.push(lines[i].endsWith('\\') ? x.slice(0, -1) + '\u0001' : x);
    }
    out.push(`<p>${inline(buf.join(' ')).replace(/\u0001 ?/g, '<br>')}</p>`);
  }
  return out.join('\n');
}

export function renderNotes(text, ctx) {
  const html = blocks(text.split(/\r?\n/), ctx);
  return ctx.secs?.length ? html + `<div class="sec-end" data-sec="${ctx.secs[ctx.secs.length - 1].id}"></div>` : html;
}
