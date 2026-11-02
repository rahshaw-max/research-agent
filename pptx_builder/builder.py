import shutil
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "templates", "UM_Semantic_Layer_for_Agents_PoV.pptx")

# Deloitte brand palette
DELOITTE_GREEN = RGBColor(0x04, 0x6A, 0x38)
BRIGHT_GREEN   = RGBColor(0x86, 0xBC, 0x25)
NEAR_BLACK     = RGBColor(0x1A, 0x1A, 0x1A)
WHITE          = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY     = RGBColor(0xF2, 0xF2, 0xF2)
MID_GRAY       = RGBColor(0x6B, 0x72, 0x7A)


def _clear_shape_text(shape):
    if shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            for run in para.runs:
                run.text = ""
        if shape.text_frame.paragraphs:
            shape.text_frame.paragraphs[0].runs[0].text = "" if shape.text_frame.paragraphs[0].runs else ""


def _set_placeholder_text(slide, placeholder_idx: int, text: str, font_size: int = None, bold: bool = False, color: RGBColor = None):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == placeholder_idx:
            tf = ph.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = text
            if font_size:
                run.font.size = Pt(font_size)
            if bold:
                run.font.bold = True
            if color:
                run.font.color.rgb = color
            return


def _add_textbox(slide, left, top, width, height, text,
                 font_size=11, bold=False, color=NEAR_BLACK,
                 fill_color=None, align=PP_ALIGN.LEFT, italic=False):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = "Calibri"
    if fill_color:
        txBox.fill.solid()
        txBox.fill.fore_color.rgb = fill_color
    return txBox


def _add_bullet_textbox(slide, left, top, width, height, items: list,
                        font_size=10, color=NEAR_BLACK, title: str = None, title_color=DELOITTE_GREEN):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True

    first = True
    if title:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        run = p.add_run()
        run.text = title
        run.font.size = Pt(font_size + 1)
        run.font.bold = True
        run.font.color.rgb = title_color
        run.font.name = "Calibri"

    for item in items:
        p = tf.paragraphs[0] if (first and not title) else tf.add_paragraph()
        first = False
        p.level = 0
        run = p.add_run()
        run.text = f"• {item}"
        run.font.size = Pt(font_size)
        run.font.color.rgb = color
        run.font.name = "Calibri"

    return txBox


def _is_placeholder(shape) -> bool:
    try:
        return shape.is_placeholder
    except Exception:
        return False


def _remove_extra_shapes(slide, keep_placeholders=True):
    to_remove = []
    for shape in slide.shapes:
        if not _is_placeholder(shape):
            to_remove.append(shape)
    for shape in to_remove:
        try:
            sp = shape._element
            sp.getparent().remove(sp)
        except Exception:
            pass


def _populate_slide_1(slide, data: dict):
    """Cover slide — title + subtitle."""
    _remove_extra_shapes(slide)
    for ph in slide.placeholders:
        if not ph.has_text_frame:
            continue
        idx = ph.placeholder_format.idx
        tf = ph.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        run = p.add_run()
        if idx == 0:  # title
            run.text = data["title"]
            run.font.size = Pt(36)
            run.font.bold = True
            run.font.color.rgb = WHITE
            run.font.name = "Calibri Light"
        elif idx == 1:  # subtitle
            run.text = data["subtitle"]
            run.font.size = Pt(16)
            run.font.color.rgb = BRIGHT_GREEN
            run.font.name = "Calibri"


def _populate_slide_2(slide, data: dict):
    """Executive Summary slide."""
    _remove_extra_shapes(slide)
    for ph in slide.placeholders:
        if not ph.has_text_frame:
            continue
        idx = ph.placeholder_format.idx
        tf = ph.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        run = p.add_run()
        if idx == 0:
            run.text = "Executive Summary"
            run.font.size = Pt(24)
            run.font.bold = True
            run.font.color.rgb = NEAR_BLACK
            run.font.name = "Calibri Light"
        elif idx == 1:
            run.text = data["executive_summary"]
            run.font.size = Pt(12)
            run.font.color.rgb = MID_GRAY
            run.font.name = "Calibri"

    _add_textbox(slide, 0.5, 2.0, 8.5, 0.35, "Key Research Highlights",
                 font_size=13, bold=True, color=DELOITTE_GREEN)
    _add_bullet_textbox(slide, 0.5, 2.4, 8.5, 3.5,
                        data.get("research_highlights", []),
                        font_size=11)


def _populate_slide_3(slide, data: dict):
    """Use Cases slide."""
    _remove_extra_shapes(slide)
    for ph in slide.placeholders:
        if not ph.has_text_frame:
            continue
        idx = ph.placeholder_format.idx
        tf = ph.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        run = p.add_run()
        if idx == 0:
            run.text = "Use Cases"
            run.font.size = Pt(24)
            run.font.bold = True
            run.font.color.rgb = NEAR_BLACK
            run.font.name = "Calibri Light"
        elif idx == 1:
            run.text = "Primary applications of the concept"
            run.font.size = Pt(12)
            run.font.color.rgb = MID_GRAY
            run.font.name = "Calibri"

    use_cases = data.get("use_cases", [])
    col_width = 4.0
    col_height = 1.5
    margin_left = 0.5
    margin_top = 2.0
    cols = 2

    for i, uc in enumerate(use_cases[:6]):
        col = i % cols
        row = i // cols
        left = margin_left + col * (col_width + 0.25)
        top = margin_top + row * (col_height + 0.2)

        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(col_width), Inches(col_height))
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT_GRAY
        tf = box.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = uc.get("title", "")
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = DELOITTE_GREEN
        run.font.name = "Calibri"

        p2 = tf.add_paragraph()
        run2 = p2.add_run()
        run2.text = uc.get("description", "")
        run2.font.size = Pt(10)
        run2.font.color.rgb = NEAR_BLACK
        run2.font.name = "Calibri"


def _populate_slide_4(slide, data: dict):
    """Architecture slide."""
    _remove_extra_shapes(slide)
    for ph in slide.placeholders:
        if not ph.has_text_frame:
            continue
        idx = ph.placeholder_format.idx
        tf = ph.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        run = p.add_run()
        if idx == 0:
            run.text = "Technical Architecture"
            run.font.size = Pt(24)
            run.font.bold = True
            run.font.color.rgb = NEAR_BLACK
            run.font.name = "Calibri Light"
        elif idx == 1:
            run.text = data.get("architecture_overview", "")
            run.font.size = Pt(11)
            run.font.color.rgb = MID_GRAY
            run.font.name = "Calibri"

    components = data.get("architecture_components", [])
    cols = 2
    comp_width = 4.0
    comp_height = 0.85
    margin_left = 0.5
    margin_top = 2.4

    for i, comp in enumerate(components[:8]):
        col = i % cols
        row = i // cols
        left = margin_left + col * (comp_width + 0.25)
        top = margin_top + row * (comp_height + 0.1)

        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(comp_width), Inches(comp_height))
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT_GRAY
        tf = box.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = comp.get("name", "")
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = DELOITTE_GREEN
        run.font.name = "Calibri"

        p2 = tf.add_paragraph()
        run2 = p2.add_run()
        run2.text = comp.get("role", "")
        run2.font.size = Pt(9)
        run2.font.color.rgb = NEAR_BLACK
        run2.font.name = "Calibri"


def _populate_slide_5(slide, data: dict):
    """Critique & Risks slide."""
    _remove_extra_shapes(slide)
    for ph in slide.placeholders:
        if not ph.has_text_frame:
            continue
        idx = ph.placeholder_format.idx
        tf = ph.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        run = p.add_run()
        if idx == 0:
            run.text = "Critique & Risk Assessment"
            run.font.size = Pt(24)
            run.font.bold = True
            run.font.color.rgb = NEAR_BLACK
            run.font.name = "Calibri Light"

    risks = data.get("key_risks", [])
    num_rows = len(risks) + 1  # +1 for header
    if num_rows < 2:
        num_rows = 2

    tbl = slide.shapes.add_table(num_rows, 2, Inches(0.5), Inches(1.6), Inches(8.5), Inches(0.4 * num_rows))
    table = tbl.table

    # Header row
    for col_idx, header in enumerate(["Risk", "Mitigation"]):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = DELOITTE_GREEN
        tf = cell.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = header
        run.font.bold = True
        run.font.color.rgb = WHITE
        run.font.size = Pt(10)
        run.font.name = "Calibri"

    # Data rows
    for row_idx, risk in enumerate(risks, start=1):
        for col_idx, key in enumerate(["risk", "mitigation"]):
            cell = table.cell(row_idx, col_idx)
            if row_idx % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GRAY
            tf = cell.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = risk.get(key, "")
            run.font.size = Pt(9)
            run.font.color.rgb = NEAR_BLACK
            run.font.name = "Calibri"

    _add_textbox(slide, 0.5, 1.6 + 0.4 * num_rows + 0.15, 8.5, 0.8,
                 data.get("critique_summary", ""), font_size=10, color=MID_GRAY, italic=True)


def _populate_slide_6(slide, data: dict):
    """Path Forward / Recommendation slide."""
    _remove_extra_shapes(slide)
    for ph in slide.placeholders:
        if not ph.has_text_frame:
            continue
        idx = ph.placeholder_format.idx
        tf = ph.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        run = p.add_run()
        if idx == 0:
            run.text = "Revised Architecture & Path Forward"
            run.font.size = Pt(24)
            run.font.bold = True
            run.font.color.rgb = NEAR_BLACK
            run.font.name = "Calibri Light"
        elif idx == 1:
            run.text = "Key revisions made based on critique findings"
            run.font.size = Pt(12)
            run.font.color.rgb = MID_GRAY
            run.font.name = "Calibri"

    _add_textbox(slide, 0.5, 1.9, 8.5, 0.35, "Architecture Revisions",
                 font_size=13, bold=True, color=DELOITTE_GREEN)
    _add_bullet_textbox(slide, 0.5, 2.3, 8.5, 2.5,
                        data.get("revisions_made", []), font_size=10)

    # Recommendation highlight box
    rec_top = 5.0
    box = slide.shapes.add_textbox(Inches(0.5), Inches(rec_top), Inches(8.5), Inches(1.0))
    box.fill.solid()
    box.fill.fore_color.rgb = DELOITTE_GREEN
    tf = box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "Recommendation: "
    run.font.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = WHITE
    run.font.name = "Calibri"

    run2 = p.add_run()
    run2.text = data.get("recommendation", "")
    run2.font.size = Pt(11)
    run2.font.color.rgb = WHITE
    run2.font.name = "Calibri"


SLIDE_POPULATORS = [
    _populate_slide_1,
    _populate_slide_2,
    _populate_slide_3,
    _populate_slide_4,
    _populate_slide_5,
    _populate_slide_6,
]


def build_presentation(session_id: str, data: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{session_id}.pptx")

    # Copy template preserving all branding
    template_path = os.path.abspath(TEMPLATE_PATH)
    shutil.copy2(template_path, output_path)

    prs = Presentation(output_path)
    slides = prs.slides

    # Ensure we have exactly 6 slides — pad or trim as needed
    while len(slides) < 6:
        # Duplicate last slide's layout
        layout = slides[-1].slide_layout if slides else prs.slide_layouts[0]
        prs.slides.add_slide(layout)

    for i, populator in enumerate(SLIDE_POPULATORS):
        if i < len(slides):
            populator(slides[i], data)

    prs.save(output_path)
    return output_path
