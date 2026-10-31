---
name: add-notes
description: Faris updated the MLR570 notes PDF (new parts or sections, edits) — bring the Course Guide in line: rebuild the course outline, transcribe new or changed parts verbatim into parts/<id>/notes.md, extract the PDF's figures, build the part's interactive visuals and its check script, audit, commit. Use when asked to add new notes, a new chapter/part, or to sync the site with an updated PDF.
---

# Notes PDF → Course Guide part

Reference part: **`parts/1.1/`** (notes.md, figures.json, fig/, viz.js) with `scripts/check/1_1.py`. Every new part
must read and behave like a sibling of it. Read `CLAUDE.md` first.

## 0. Get the new PDF
Faris replaces `reference/MLR570_Course_Notes.pdf`. Keep the file name. Then:
```
python scripts/build_course.py            # outline -> data/course.json (parts, sections, PDF pages)
```
Compare the printed outline with the previous one (`git diff data/course.json`): new parts, new sections, and
**changed page numbers** (an edit early in the PDF shifts every later page: figures.json rects of later parts must be
re-checked, see step 3).

## 1. Read before writing
Render the pages of the part to PNG and read them as images (text extraction breaks math):
```
python -I -c "import fitz,sys; d=fitz.open(sys.argv[1]); [d[i].get_pixmap(dpi=110).save(f'{sys.argv[2]}/p{i+1:03d}.png') for i in range(int(sys.argv[3])-1, int(sys.argv[4]))]" reference/MLR570_Course_Notes.pdf <scratch dir> <first page> <last page>
```
Use `page.get_text()` only to copy wording. For an *edited* part, diff the old and new PDF text per section to find
exactly what changed, and change only that.

## 2. Transcribe verbatim → `parts/<id>/notes.md`
- Start at the part's "The flow" box; the chapter banner and part title come from course.json.
- Every sentence, equation (symbol for symbol in KaTeX), list, table, box and worked example, in order. Same bold/italic.
- Box mapping (the PDF's legend): red → `:::def Title`; blue → `:::ml Title`; amber "Added note"/"Added intuition"/
  "Added clarification" → `:::add <its own title>`; grey example boxes with a header bar → `:::ex <header>`; the
  bordered "The flow" box → `:::plain`; centred "Big picture" → `:::big Big picture`. Titles verbatim.
- Boxed equations `\boxed{}`, underbraces `\underbrace{}`, matrices `bmatrix`, `\tfrac` where the PDF uses small fractions.
- Two-column layouts → `:::cols` (`figleft` when the figure is on the left, `even` for two text columns).
- Quick-reference tables: one table even when the PDF splits it over pages; first line `|{stack}`.
- Section cross-references "§1.2.7" stay as written: the renderer links them.
- Never fix the notes silently. If something is wrong (a number, a sign, a claim), keep it verbatim and tell Faris.

## 3. The PDF's own figures → `parts/<id>/figures.json`
For each figure: `{"id", "page", "rect": [x0, y0, x1, y1], "label"}` in PDF points (page 595×842; px at 110 dpi ÷ 1.528).
Make the rect generous enough to include the figure's own labels and captions but no body text. Then
`python -I scripts/extract_figures.py <id>` and place `:::fig <id>` where the figure sits in the PDF. Check each one in
the browser: nothing clipped, no stray text, readable colours. Small matrix-with-label layouts that KaTeX cannot
reproduce exactly are also extracted as figures (1.1's `data-matrix`).

## 4. Visuals → `parts/<id>/viz.js`
Start from `research/visual-plan.md` for the part; refine it if reading the part suggests better ideas, and update
the plan file. Bar: 3Blue1Brown / Seeing Theory / setosa quality, like the Survey Atlas.
- `export default { 'viz-id': { title, mount(f) } }`; `f` has `stage`, `side`, `body`, `fig`. Return an API with
  `reset()` and `state()` (used by the audit). Use `src/viz/plane.js` for 2D, `src/viz/three-lite.js` for 3D,
  `src/viz/ui.js` for controls; shared numerics in `src/viz/la.js` (add new shared numerics there, e.g. a stats module).
- Place `:::viz viz-id | one-line caption saying what to drag` right after the passage it explains.
- Real math only; defaults/presets = the notes' worked examples; labels use only the notes' terms and numbers.
- Colour language: first column/e1 green `#5fe0b0`, second coral `#ff7a59`, span/column space blue `#7cc4ff`,
  null space red `#ff5d73`, eigen directions amber `#ffc857`, the dragged vector violet `#b28cff`.
- Phones: stages stack (`.split` becomes one column under 760 px); touch drags only start on a handle.

## 5. Check script → `scripts/check/<id with _>.py`
Recompute every number in the part's worked examples and every number a visual opens on (see `1_1.py`), with
`check(name, got, want, tol)` and `done()`. It must pass: `python -I scripts/check/<id>.py`.

## 6. Audit (localhost, `python scripts/serve.py`, http://localhost:8590/part.html?p=<id>)
In the page:
```
for(let i=0;i<80&&!window.__guide?.mountAll;i++) await new Promise(r=>setTimeout(r,100));
await __guide.mountAll(); await new Promise(r=>setTimeout(r,1500));
({errors: __guide.errors, mounted: __guide.mounted.length, katex: document.querySelectorAll('.katex-error').length,
  sections: document.querySelectorAll('.nt-h').length, overflow: document.documentElement.scrollWidth - innerWidth})
```
→ `[]`, the number of `:::viz`, `0`, the part's section count, `0`. Then at 375 px (`resize_window` 375×812, reload):
`innerWidth` must be 375 and `scrollWidth - innerWidth` 0 (a grid item that sizes to content, or KaTeX's absolute
MathML layer, can silently widen a phone layout). Call each visual's `state()` and compare with the check script's
numbers. Finally compare the rendered part with the PDF pages section by section: nothing missing, nothing added.

## 7. Finish
`python scripts/build_course.py` (marks the part ready, counts visuals), commit, and report to Faris: what was added,
any errors spotted in the notes (not fixed), and anything a visual could not show.
