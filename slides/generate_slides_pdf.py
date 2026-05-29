from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def wrap(text: str, max_chars: int = 72) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def parse_slides(markdown: str) -> list[tuple[str, list[str]]]:
    slides: list[tuple[str, list[str]]] = []
    title = ""
    body: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if title:
                slides.append((title, body))
            title = line[3:].strip()
            body = []
        elif line.startswith("# "):
            continue
        elif line.strip():
            body.append(line.strip())
    if title:
        slides.append((title, body))
    return slides


def main() -> None:
    source = Path("slides/viva_slides.md")
    target = Path("slides/viva_slides.pdf")
    slides = parse_slides(source.read_text(encoding="utf-8"))
    page_size = landscape(letter)
    c = canvas.Canvas(str(target), pagesize=page_size)
    width, height = page_size

    for index, (title, body) in enumerate(slides, start=1):
        c.setFillColor(colors.HexColor("#101820"))
        c.rect(0, 0, width, height, fill=True, stroke=False)
        c.setFillColor(colors.HexColor("#F2AA4C"))
        c.rect(0, height - 0.22 * inch, width, 0.22 * inch, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 26)
        c.drawString(0.65 * inch, height - 0.9 * inch, title)

        y = height - 1.55 * inch
        c.setFont("Helvetica", 18)
        for paragraph in body:
            if paragraph.startswith("- "):
                lines = wrap(paragraph[2:], 64)
                for i, line in enumerate(lines):
                    prefix = "- " if i == 0 else "  "
                    c.drawString(0.9 * inch, y, prefix + line)
                    y -= 0.34 * inch
            else:
                for line in wrap(paragraph, 70):
                    c.drawString(0.8 * inch, y, line)
                    y -= 0.34 * inch
                y -= 0.12 * inch

        c.setFillColor(colors.HexColor("#A7C7E7"))
        c.setFont("Helvetica", 10)
        c.drawRightString(width - 0.55 * inch, 0.35 * inch, f"{index}/{len(slides)}")
        c.showPage()

    c.save()
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()

