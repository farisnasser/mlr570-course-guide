"""Extract the PDF's own figures as exact vector SVGs, recoloured for the dark theme.

Each part lists its figures in parts/<id>/figures.json:  [{"id": "grid", "page": 5, "rect": [x0, y0, x1, y1]}, ...]
(page = 1-based PDF page, rect in PDF points, origin top-left; render the page at 72 dpi to read coordinates).
Everything outside the rect is redacted (text, and vector art that touches the outside), the page is cropped to the
rect and exported with text as paths, so labels and math look exactly like the PDF. Colours are mapped to the dark
theme: neutrals invert lightness, saturated strokes keep their hue but get lighter, light tinted fills become dark
tints, white fills become the figure background. IDs are prefixed so many SVGs can be inlined in one page.

usage: python scripts/extract_figures.py 1.1 [1.2 ...]      (or --all)
"""
import colorsys
import json
import os
import re
import sys

import fitz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF = os.path.join(ROOT, "reference", "MLR570_Course_Notes.pdf")


def dark(hexcol, attr):
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if s < 0.12 or l > 0.985 or l < 0.02:  # neutral
        if attr == "fill" and l > 0.97:
            return None  # white -> figure background
        l2 = 0.16 + 0.76 * (1 - l)
        h2, s2 = 0.63, 0.18  # cool neutral like the theme ink (#e8ecf8) / muted (#8e99bd)
    elif l > 0.82:  # light tinted fill (box backgrounds, shaded areas)
        l2, h2, s2 = 0.13 + 0.25 * (1 - l), h, min(1, s * 0.75)
    else:  # saturated ink: keep hue, lift for a dark background
        l2, h2, s2 = max(0.66, min(0.78, 1 - l * 0.55)), h, min(1, s * 1.05)
    r2, g2, b2 = colorsys.hls_to_rgb(h2, l2, s2)
    return "#%02x%02x%02x" % tuple(round(v * 255) for v in (r2, g2, b2))


def recolour(svg):
    def sub(m):
        attr, col = m.group(1), m.group(2)
        c = dark(col, attr)
        return f'{attr}="var(--fig-bg, #0b0e1a)"' if c is None else f'{attr}="{c}"'
    return re.sub(r'(fill|stroke)="(#[0-9a-fA-F]{6})"', sub, svg)


def shrink(svg):
    svg = re.sub(r"(\d+\.\d{3})\d+", r"\1", svg)  # 3 decimals is sub-pixel at any zoom we use
    svg = re.sub(r'\s+xmlns:inkscape="[^"]*"', "", svg)
    return svg


def prefix_ids(svg, pre):
    ids = set(re.findall(r'id="([^"]+)"', svg))
    for i in sorted(ids, key=len, reverse=True):
        svg = svg.replace(f'id="{i}"', f'id="{pre}{i}"').replace(f'#{i}"', f'#{pre}{i}"').replace(f'#{i})', f'#{pre}{i})')
    return svg


def extract(doc_path, page, rect, pre):
    doc = fitz.open(doc_path)
    pg = doc[page - 1]
    R, c = pg.rect, fitz.Rect(*rect)
    for r in (fitz.Rect(R.x0, R.y0, R.x1, c.y0), fitz.Rect(R.x0, c.y1, R.x1, R.y1),
              fitz.Rect(R.x0, c.y0, c.x0, c.y1), fitz.Rect(c.x1, c.y0, R.x1, c.y1)):
        if not r.is_empty:
            pg.add_redact_annot(r)
    pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE, graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED)
    pg.set_cropbox(c)
    svg = pg.get_svg_image(text_as_path=True)
    svg = svg.replace('<svg xmlns', f'<svg class="pdf-fig" role="img" xmlns', 1)
    return prefix_ids(shrink(recolour(svg)), pre)


def run(part):
    spec_path = os.path.join(ROOT, "parts", part, "figures.json")
    if not os.path.exists(spec_path):
        print(f"{part}: no figures.json")
        return
    out_dir = os.path.join(ROOT, "parts", part, "fig")
    os.makedirs(out_dir, exist_ok=True)
    for f in json.load(open(spec_path, encoding="utf-8")):
        pre = "f" + part.replace(".", "_") + "_" + re.sub(r"\W", "_", f["id"]) + "_"
        svg = extract(PDF, f["page"], f["rect"], pre)
        if f.get("label"):
            svg = svg.replace('role="img"', f'role="img" aria-label="{f["label"]}"', 1)
        with open(os.path.join(out_dir, f["id"] + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(svg)
        print(f"{part}/{f['id']}.svg  p.{f['page']}  {len(svg) // 1024} KB")


if __name__ == "__main__":
    parts = sys.argv[1:]
    if parts == ["--all"]:
        parts = sorted(d for d in os.listdir(os.path.join(ROOT, "parts")) if os.path.isdir(os.path.join(ROOT, "parts", d)))
    for p in parts:
        run(p)
