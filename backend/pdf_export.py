import io
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(45, 842 - 32, "MINI RESEARCHER • RESEARCH DOSSIER")
            self.drawRightString(595 - 45, 842 - 32, datetime.now().strftime("%B %d, %Y"))
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(45, 842 - 36, 595 - 45, 842 - 36)

        # Running Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(595 - 45, 28, footer_text)
        self.drawString(
            45,
            28,
            "Generated autonomously by Mini Researcher Swarm Intelligence Platform",
        )
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(45, 38, 595 - 45, 38)

        self.restoreState()


def clean_markdown_inline(text: str) -> str:
    """Converts common markdown inline formatting to ReportLab XML tags."""
    # Escape XML specials first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.*?)__", r"<b>\1</b>", text)

    # Italic: *text* or _text_
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    text = re.sub(r"(?<![a-zA-Z0-9])_(.*?)_(?![a-zA-Z0-9])", r"<i>\1</i>", text)

    # Inline code: `text`
    text = re.sub(
        r"`(.*?)`",
        r'<font face="Courier" color="#0369a1" size="9.5">\1</font>',
        text,
    )

    # Markdown links: [title](url) -> title (url)
    text = re.sub(
        r"\[(.*?)\]\((.*?)\)",
        r'<font color="#0284c7"><b>\1</b></font> (<font color="#64748b" size="8.5">\2</font>)',
        text,
    )

    return text


def build_pdf_report(
    topic: str,
    report_markdown: str,
    report_type: Optional[str] = "Standard",
    tone: Optional[str] = "Objective",
    model: Optional[str] = "Gemini 3.5 Flash Lite",
    sources: Optional[List[Dict[str, Any]]] = None,
) -> bytes:
    """
    Renders an executive publication-quality PDF from research findings and Markdown text.
    """
    buffer = io.BytesIO()

    # Document Geometry: A4 (595.28 x 841.89 pt) with 45pt (16mm) margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=48,
        bottomMargin=48,
    )

    # Styles
    base_styles = getSampleStyleSheet()

    header_super = ParagraphStyle(
        "HeaderSuper",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0284c7"),
        alignment=TA_LEFT,
        spaceAfter=6,
    )

    title_style = ParagraphStyle(
        "DocTitle",
        parent=base_styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT,
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "H1",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "H2",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "H3",
        parent=base_styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=base_styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8,
        alignment=TA_LEFT,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=16,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    quote_style = ParagraphStyle(
        "Quote",
        parent=body_style,
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#475569"),
        leftIndent=18,
        rightIndent=18,
        spaceBefore=6,
        spaceAfter=8,
    )

    story = []

    # 1. Institutional Banner
    story.append(
        Paragraph("MINI RESEARCHER • AUTONOMOUS RESEARCH DOSSIER", header_super)
    )

    # 2. Main Report Title
    clean_topic = topic.strip() if topic else "Executive Research Synthesis"
    story.append(Paragraph(clean_markdown_inline(clean_topic), title_style))

    # 3. Metadata Bar (Table)
    now_str = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    word_count = len(report_markdown.split()) if report_markdown else 0

    meta_cells = [
        [
            Paragraph(
                f"<b>Date:</b> {now_str}",
                ParagraphStyle("M1", parent=body_style, fontSize=8, leading=10),
            ),
            Paragraph(
                f"<b>Type:</b> {report_type or 'Standard'}",
                ParagraphStyle("M2", parent=body_style, fontSize=8, leading=10),
            ),
            Paragraph(
                f"<b>Tone:</b> {tone or 'Objective'}",
                ParagraphStyle("M3", parent=body_style, fontSize=8, leading=10),
            ),
            Paragraph(
                f"<b>Words:</b> {word_count}",
                ParagraphStyle("M4", parent=body_style, fontSize=8, leading=10),
            ),
        ]
    ]
    meta_table = Table(meta_cells, colWidths=[140, 130, 135, 100])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#0284c7"),
            spaceBefore=4,
            spaceAfter=14,
        )
    )

    # 4. Parse Markdown Body
    lines = report_markdown.splitlines()
    in_code_block = False
    code_lines = []

    for line in lines:
        stripped = line.strip()

        # Handle Code Blocks
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "<br/>".join(code_lines)
                code_p = Paragraph(
                    f'<font face="Courier" size="8" color="#0f172a">{code_text}</font>',
                    ParagraphStyle(
                        "CodeBlock",
                        parent=body_style,
                        backColor=colors.HexColor("#f1f5f9"),
                        borderColor=colors.HexColor("#cbd5e1"),
                        borderWidth=0.5,
                        borderPadding=6,
                        spaceBefore=6,
                        spaceAfter=8,
                    ),
                )
                story.append(code_p)
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(
                stripped.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            continue

        if not stripped:
            continue

        # Headings
        if stripped.startswith("### "):
            story.append(
                Paragraph(clean_markdown_inline(stripped[4:].strip()), h3_style)
            )
        elif stripped.startswith("## "):
            story.append(
                Paragraph(clean_markdown_inline(stripped[3:].strip()), h2_style)
            )
        elif stripped.startswith("# "):
            story.append(
                Paragraph(clean_markdown_inline(stripped[2:].strip()), h1_style)
            )
        # Blockquotes
        elif stripped.startswith("> "):
            story.append(
                Paragraph(clean_markdown_inline(stripped[2:].strip()), quote_style)
            )
        # Bullet Lists
        elif stripped.startswith("- ") or stripped.startswith("* "):
            content = clean_markdown_inline(stripped[2:].strip())
            story.append(Paragraph(f"• &nbsp; {content}", bullet_style))
        elif re.match(r"^\d+\.\s+", stripped):
            num_match = re.match(r"^(\d+\.)\s+(.*)", stripped)
            if num_match:
                prefix, content = num_match.groups()
                story.append(
                    Paragraph(
                        f"<b>{prefix}</b> {clean_markdown_inline(content)}",
                        bullet_style,
                    )
                )
        # Standard Paragraph
        else:
            story.append(Paragraph(clean_markdown_inline(stripped), body_style))

    # 5. Citations & References
    if sources and len(sources) > 0:
        story.append(Spacer(1, 14))
        story.append(
            HRFlowable(
                width="100%",
                thickness=0.5,
                color=colors.HexColor("#cbd5e1"),
                spaceBefore=10,
                spaceAfter=12,
            )
        )
        story.append(
            Paragraph(f"Curated References & Citations ({len(sources)})", h2_style)
        )

        for i, s in enumerate(sources, start=1):
            title = s.get("title") or "Source Reference"
            url = s.get("url") or ""
            ref_text = f"<b>[{i}] {clean_markdown_inline(title)}</b><br/><font color='#0284c7' size='8'>{url}</font>"
            story.append(
                Paragraph(
                    ref_text,
                    ParagraphStyle(
                        f"Ref{i}", parent=bullet_style, fontSize=8.5, spaceAfter=5
                    ),
                )
            )

    # Build Document with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
