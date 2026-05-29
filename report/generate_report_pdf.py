from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


def add_markdown_line(story, line: str, styles) -> None:
    line = line.strip()
    if not line:
        story.append(Spacer(1, 0.08 * inch))
        return
    if line.startswith("# "):
        story.append(Paragraph(line[2:], styles["Title"]))
    elif line.startswith("## "):
        story.append(Paragraph(line[3:], styles["Heading1"]))
    elif line.startswith("### "):
        story.append(Paragraph(line[4:], styles["Heading2"]))
    elif line.startswith("- "):
        story.append(Paragraph("&#8226; " + line[2:], styles["ProjectBullet"]))
    elif line[0:2].isdigit() and ". " in line[:4]:
        story.append(Paragraph(line, styles["BodyText"]))
    elif line.startswith("|"):
        story.append(Paragraph(line.replace("|", " | "), styles["ProjectCode"]))
    else:
        story.append(Paragraph(line.replace("**", ""), styles["BodyText"]))


def main() -> None:
    source = Path("report/ieee_report.md")
    target = Path("report/ieee_report.pdf")
    styles = getSampleStyleSheet()
    styles["Title"].fontName = "Times-Bold"
    styles["Title"].fontSize = 18
    styles["Title"].leading = 22
    styles["Heading1"].fontName = "Times-Bold"
    styles["Heading1"].fontSize = 12
    styles["Heading1"].leading = 14
    styles["BodyText"].fontName = "Times-Roman"
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 11
    styles.add(ParagraphStyle(name="ProjectBullet", parent=styles["BodyText"], leftIndent=14, firstLineIndent=-8))
    styles.add(ParagraphStyle(name="ProjectCode", parent=styles["BodyText"], fontName="Courier", fontSize=7, textColor=colors.darkslategray))

    document = SimpleDocTemplate(
        str(target),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    story = []
    for line in source.read_text(encoding="utf-8").splitlines():
        add_markdown_line(story, line, styles)
    story.append(PageBreak())
    story.append(Paragraph("Appendix: Reproducibility Commands", styles["Heading1"]))
    story.append(Paragraph("python scripts/generate_sample_images.py", styles["ProjectCode"]))
    story.append(Paragraph("python run_enhancement.py --input data/sample_low_light --reference data/sample_reference --output results", styles["ProjectCode"]))
    story.append(Paragraph("python evaluate.py --results results/metrics.csv", styles["ProjectCode"]))
    document.build(story)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
