"""Build data/course.json from the notes PDF outline (chapters -> parts -> sections, with PDF page numbers).

The outline is the PDF's own bookmarks, so the site's structure always matches the PDF. Chapter intro sentences and
part titles come from the PDF too. A part is "ready" when parts/<id>/notes.md exists.

usage: python scripts/build_course.py [reference/MLR570_Course_Notes.pdf]
"""
import json
import os
import re
import sys

import fitz  # PyMuPDF

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "reference", "MLR570_Course_Notes.pdf")


def clean(t):
    # bookmarks carry LaTeX quotes and a broken en dash glyph
    return t.replace("``", "“").replace("''", "”").replace("�", "–").strip()


def chapter_intro(doc, page):
    """The sentence(s) under the 'n · Title' chapter heading, up to 'PART'."""
    text = doc[page - 1].get_text()
    m = re.search(r"\d+ · [^\n]+\n(.*?)\nPART ", text, re.S)
    return " ".join(m.group(1).split()) if m else ""


def main():
    doc = fitz.open(PDF)
    chapters = []
    for level, title, page in doc.get_toc():
        title = clean(title)
        if level == 1:
            chapters.append({"n": len(chapters) + 1, "title": title, "page": page,
                             "intro": chapter_intro(doc, page), "parts": []})
        elif level == 2:
            ch = chapters[-1]
            pid = f"{ch['n']}.{len(ch['parts']) + 1}"
            ch["parts"].append({"id": pid, "title": title, "page": page, "sections": []})
        else:
            part = chapters[-1]["parts"][-1]
            part["sections"].append({"n": f"{part['id']}.{len(part['sections']) + 1}", "title": title, "page": page})
    total = doc.page_count
    flat = [p for c in chapters for p in c["parts"]]
    for i, p in enumerate(flat):
        p["end"] = (flat[i + 1]["page"] - 1) if i + 1 < len(flat) else total
        p["ready"] = os.path.exists(os.path.join(ROOT, "parts", p["id"], "notes.md"))
        nm = os.path.join(ROOT, "parts", p["id"], "notes.md")
        p["visuals"] = len(re.findall(r"^:::viz ", open(nm, encoding="utf-8").read(), re.M)) if p["ready"] else 0
    out = {"title": "MLR 570 Course Notes", "pdfPages": total, "chapters": chapters}
    with open(os.path.join(ROOT, "data", "course.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"{len(chapters)} chapters, {len(flat)} parts ({sum(p['ready'] for p in flat)} ready), "
          f"{sum(len(p['sections']) for p in flat)} sections")


if __name__ == "__main__":
    main()
