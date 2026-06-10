import streamlit as st  # Cubroid Lesson Generator v6
import openai
import os
import io
import re
import fitz  # PyMuPDF
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER
from PIL import Image as PILImage

# ── Brand colours ─────────────────────────────────────────────────────────────
CUBROID_BLUE   = "#1E3A8A"
CUBROID_ORANGE = "#F97316"
CUBROID_LIGHT  = "#EFF6FF"
CUBROID_GREY   = "#64748B"

SUBJECT = "Coding and Robotics"
GRADES  = ["Grade R", "Grade 1", "Grade 2", "Grade 3"]

CAPS_DOC      = "CAPS FP CODING AND ROBOTICS.pdf"
TEACHER_GUIDE = "Step 1 Teacher Guides (Book 1–12)"
MISSION_GUIDE = "Mission Coding-Phone.pdf"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cubroid Lesson Generator",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS — clean, minimal, teacher-facing ───────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
  .block-container { padding-top: 3rem; max-width: 760px; }

  /* Hero */
  .hero { text-align: center; padding: 2rem 1rem 1.5rem 1rem; }
  .hero .badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1E3A8A;
    border: 1px solid #BFDBFE;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }
  .hero h1 {
    color: #0F172A;
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 0.5rem 0;
    line-height: 1.15;
  }
  .hero h1 span { color: #1E3A8A; }
  .hero p {
    color: #64748B;
    font-size: 1.02rem;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
  }

  .step-label {
    color: #1E3A8A;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.2rem;
  }
  .form-title { color: #0F172A; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
  .form-sub   { color: #64748B; font-size: 0.92rem; margin-bottom: 1rem; }

  div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {
    border-radius: 12px;
    font-weight: 600;
  }
  /* Big home buttons */
  div.stButton > button[kind="primary"] { padding: 0.9rem 1rem; font-size: 1.05rem; }
  div.stButton > button[kind="secondary"] { padding: 0.9rem 1rem; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# ── OpenAI / LlamaIndex setup ─────────────────────────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"],
                      max_tokens=4000)
Settings.embed_model = OpenAIEmbedding(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource(show_spinner="Getting things ready for you...")
def load_index():
    parser = LlamaParse(
        api_key=st.secrets["LLAMA_CLOUD_API_KEY"],
        result_type="markdown"
    )
    docs = SimpleDirectoryReader(
        "docs",
        recursive=True,
        file_extractor={".pdf": parser}
    ).load_data()
    return VectorStoreIndex.from_documents(docs)

index = load_index()

# ── Session state defaults ────────────────────────────────────────────────────
DEFAULTS = {
    "view": "home",          # home | form | result | troubleshoot
    "grade": "Grade R",
    "term": 1,
    "week": 1,
    "robot": "Cubroid Coding Blocks",
    "duration": "30 minutes",
    "teacher_name": "",
    "school_name": "",
    "custom_notes": "",
    "result": None,
    "images": [],
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── MARKDOWN CLEANING (shared by PDF + parser) ────────────────────────────────
def md_inline(s):
    """Convert inline markdown to ReportLab markup, escaping XML first."""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", s)
    s = re.sub(r"__(.+?)__", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", s)
    s = s.replace("**", "").replace("##", "")  # stray leftovers
    return s


def clean_title(s):
    """Strip all markdown decoration from a heading."""
    return re.sub(r"[#*_`]+", "", s).strip(" :–-\t")


HEADING_RE = [
    re.compile(r"^#{1,6}\s+(.+)$"),                  # ## Heading
    re.compile(r"^\*\*([^*]+)\*\*:?\s*$"),           # **Heading**
    re.compile(r"^\d+\.\s+\*\*([^*]+)\*\*:?\s*$"),   # 3. **Heading**
]


def parse_sections(text):
    sections, title, body = [], "Overview", []
    for line in text.splitlines():
        stripped = line.strip()
        matched = None
        for rx in HEADING_RE:
            m = rx.match(stripped)
            if m:
                matched = m.group(1)
                break
        if matched is not None:
            if body and any(b.strip() for b in body):
                sections.append((title, "\n".join(body).strip()))
            title, body = clean_title(matched), []
        else:
            body.append(line)
    if body and any(b.strip() for b in body):
        sections.append((title, "\n".join(body).strip()))
    return sections


# ── SOURCE PAGE SCREENSHOTS ───────────────────────────────────────────────────
def find_doc(fp):
    """Resolve a node file path against the docs folder."""
    if os.path.isabs(fp) and os.path.exists(fp):
        return fp
    cand = os.path.join("docs", fp)
    if os.path.exists(cand):
        return cand
    name = os.path.basename(fp)
    for root, _, files in os.walk("docs"):
        if name in files:
            return os.path.join(root, name)
    return None


def render_page(fp, page_idx, zoom=2.0):
    """Render one page (0-based index) of a PDF to JPEG bytes."""
    pdf_doc = fitz.open(fp)
    page_idx = max(0, min(page_idx, len(pdf_doc) - 1))
    page = pdf_doc[page_idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    pil = PILImage.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=82)
    pdf_doc.close()
    return buf.getvalue(), page_idx


def best_page_for_text(fp, content, skip_first=True):
    """Find the page (0-based) whose text best matches the given content.
    Used when retrieval metadata has no reliable page number."""
    needles = []
    for line in content.splitlines():
        line = re.sub(r"[#*_`|]+", "", line).strip().lower()
        if len(line) >= 20:
            needles.append(line[:50])
        if len(needles) >= 10:
            break
    if not needles:
        return None
    try:
        pdf_doc = fitz.open(fp)
    except Exception:
        return None
    best, best_score = None, 0
    start = 1 if (skip_first and len(pdf_doc) > 2) else 0
    for pno in range(start, len(pdf_doc)):
        ptext = pdf_doc[pno].get_text().lower()
        ptext = re.sub(r"\s+", " ", ptext)
        score = sum(1 for n in needles if re.sub(r"\s+", " ", n) in ptext)
        if score > best_score:
            best, best_score = pno, score
    pdf_doc.close()
    return best


# Patterns that find page citations in the generated lesson text,
# e.g. "(Step1 Book 3, p.12)", "(Mission Guide, p.7)", "(CAPS, p.14)"
PAGE_REF = r"(?:pages?|pg\.?|pp\.?|p\.)\s*(\d+)"
CITATION_PATTERNS = [
    (re.compile(r"step\s*1?\s*book\s*(\d+)[^\n.;)]*?" + PAGE_REF, re.I),
     lambda m: (f"Step1 Book {m.group(1)}.pdf", int(m.group(2)))),
    (re.compile(r"mission[^\n.;)]*?" + PAGE_REF, re.I),
     lambda m: (MISSION_GUIDE, int(m.group(1)))),
    (re.compile(r"caps[^\n.;)]*?" + PAGE_REF, re.I),
     lambda m: (CAPS_DOC, int(m.group(1)))),
]


def extract_page_screenshots(source_nodes, result_text="", max_shots=6, zoom=2.0):
    """Screenshot the exact pages the lesson plan cites, plus the pages the
    retrieved content actually lives on. Never cover pages."""
    shots, seen = [], set()

    def add(fname, page_1based):
        if len(shots) >= max_shots:
            return
        fp = find_doc(fname)
        if not fp:
            return
        name = os.path.basename(fp)
        key = (name.lower(), page_1based)
        if key in seen:
            return
        try:
            data, used_idx = render_page(fp, page_1based - 1, zoom)
        except Exception:
            return
        seen.add(key)
        seen.add((name.lower(), used_idx + 1))
        shots.append({"bytes": data, "source": name, "page": used_idx + 1})

    # 1) Pages explicitly cited in the lesson text (most relevant)
    for rx, getter in CITATION_PATTERNS:
        for m in rx.finditer(result_text):
            fname, pg = getter(m)
            if pg > 1:
                add(fname, pg)

    # 2) Pages where the retrieved content actually lives
    for node in source_nodes:
        if len(shots) >= max_shots:
            break
        n = getattr(node, "node", node)
        meta = getattr(n, "metadata", None) or {}
        fp = find_doc(meta.get("file_path") or meta.get("file_name", ""))
        if not fp:
            continue
        page = None
        try:
            page = int(str(meta.get("page_label") or meta.get("page")))
        except (TypeError, ValueError):
            page = None
        if page is None or page <= 1:
            try:
                content = n.get_content()
            except Exception:
                content = ""
            pno = best_page_for_text(fp, content)
            page = (pno + 1) if pno is not None else None
        if page and page > 1:
            add(os.path.basename(fp), page)
    return shots


# Map each screenshot to the lesson section it belongs with, based on which
# source document it came from.
SECTION_HINTS = [
    ("caps",    ["caps alignment"]),
    ("mission", ["further work", "mission"]),
    ("step",    ["guided practice", "direct instruction",
                 "independent", "teacher guide reference"]),
]


def assign_images(sections, images):
    """Return {section_index: [images]} placing each screenshot under the
    section that matches its source document type."""
    titles = [t.lower() for t, _ in sections]
    img_map, used = {}, set()

    def find_section(hints, allow_reuse):
        for h in hints:
            for idx, t in enumerate(titles):
                if h in t and (allow_reuse or idx not in used):
                    return idx
        return None

    for img in images:
        src = img["source"].lower()
        target = None
        for key, hints in SECTION_HINTS:
            if key in src:
                target = find_section(hints, allow_reuse=False)
                if target is None:
                    target = find_section(hints, allow_reuse=True)
                break
        if target is None:
            target = len(sections) - 1
        used.add(target)
        img_map.setdefault(target, []).append(img)
    return img_map


def format_sources(source_nodes):
    """Build a deduplicated reference list from the retrieved nodes."""
    seen, lines = set(), []
    for node in source_nodes:
        meta = node.metadata or {}
        name = os.path.basename(meta.get("file_name") or meta.get("file_path", ""))
        if not name:
            continue
        page = meta.get("page_label") or meta.get("page")
        key = (name, str(page))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {name}" + (f", page {page}" if page else ""))
    return "\n".join(lines)


# ── PDF GENERATION ────────────────────────────────────────────────────────────
def build_pdf(text, images, grade, subject, term, week, duration, teacher, school):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2.5*cm, bottomMargin=2*cm)
    rl_blue   = colors.HexColor(CUBROID_BLUE)
    rl_orange = colors.HexColor(CUBROID_ORANGE)
    rl_light  = colors.HexColor(CUBROID_LIGHT)
    rl_grey   = colors.HexColor(CUBROID_GREY)

    body_style = ParagraphStyle("body", fontName="Helvetica", fontSize=9,
                                textColor=colors.HexColor("#1E293B"), leading=14, spaceAfter=4)
    bullet_style = ParagraphStyle("bullet", fontName="Helvetica", fontSize=9,
                                  textColor=colors.HexColor("#1E293B"), leading=14,
                                  leftIndent=12, spaceAfter=2)
    heading_style = ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=11,
                                   textColor=rl_blue, spaceBefore=10, spaceAfter=4)
    meta_label = ParagraphStyle("ml", fontName="Helvetica-Bold", fontSize=7, textColor=rl_grey)
    meta_val   = ParagraphStyle("mv", fontName="Helvetica", fontSize=9, textColor=rl_blue)
    caption    = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=7,
                                textColor=rl_grey, alignment=TA_CENTER)
    footer_s   = ParagraphStyle("foot", fontName="Helvetica", fontSize=7,
                                textColor=rl_grey, alignment=TA_CENTER)
    title_s    = ParagraphStyle("ttl", fontName="Helvetica-Bold", fontSize=20,
                                textColor=colors.white, alignment=TA_CENTER)

    W, _ = A4
    cw = W - 4*cm
    story = []

    # Header banner
    hdr = Table([[Paragraph("CUBROID", title_s), Paragraph("LESSON PLAN", title_s)]],
                colWidths=[cw/2, cw/2])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), rl_blue),
        ("TOPPADDING",    (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 0.3*cm))

    # Meta bar
    meta_items = [("Grade", grade), ("Subject", subject), ("Term", str(term)),
                  ("Week", str(week)), ("Duration", duration)]
    if teacher:
        meta_items.append(("Teacher", teacher))
    if school:
        meta_items.append(("School", school))
    n = len(meta_items)
    meta_cells = []
    for lbl, val in meta_items:
        meta_cells.append(Table([[Paragraph(lbl, meta_label)], [Paragraph(val, meta_val)]],
                                colWidths=[cw/n]))
    mt = Table([meta_cells], colWidths=[cw/n]*n)
    mt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), rl_light),
        ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID",     (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.5*cm))

    # Place screenshots under their matching sections
    sections = parse_sections(text)
    img_map = assign_images(sections, images) if images else {}

    for i, (sec_title, sec_body) in enumerate(sections):
        if not sec_title and not sec_body:
            continue
        # Heading with orange left bar
        ht = Table([[Paragraph(clean_title(sec_title).upper(), heading_style)]], colWidths=[cw])
        ht.setStyle(TableStyle([
            ("LINEBEFORE",    (0,0), (0,0), 3, rl_orange),
            ("LEFTPADDING",   (0,0), (0,0), 10),
            ("TOPPADDING",    (0,0), (0,0), 6),
            ("BOTTOMPADDING", (0,0), (0,0), 6),
            ("BACKGROUND",    (0,0), (0,0), colors.HexColor("#F8FAFC")),
        ]))
        story.append(ht)

        for line in sec_body.splitlines():
            line = line.strip()
            if not line:
                story.append(Spacer(1, 3))
            elif line.startswith(("- ", "* ", "• ")):
                story.append(Paragraph("• " + md_inline(line[2:]), bullet_style))
            elif re.match(r"^\d+\.\s", line):
                story.append(Paragraph(md_inline(line), bullet_style))
            else:
                story.append(Paragraph(md_inline(line), body_style))

        # Embed screenshots assigned to this section — compact, 2 per row
        sec_imgs = img_map.get(i, [])
        if sec_imgs:
            cells = []
            cell_w = (cw / 2) - 0.4*cm
            for img_data in sec_imgs:
                try:
                    img_buf = io.BytesIO(img_data["bytes"])
                    pil_img = PILImage.open(img_buf)
                    ow, oh = pil_img.size
                    scale = min((cell_w - 0.3*cm) / ow, (5.5*cm) / oh)
                    dw, dh = ow * scale, oh * scale
                    img_buf.seek(0)
                    rl_img = RLImage(img_buf, width=dw, height=dh)
                    cap_p = Paragraph(
                        f"Source: {img_data['source']}, p.{img_data['page']}", caption)
                    cell = Table([[rl_img], [cap_p]], colWidths=[dw + 0.2*cm])
                    cell.setStyle(TableStyle([
                        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                        ("BOX",           (0,0), (0,0), 0.5, colors.HexColor("#CBD5E1")),
                        ("TOPPADDING",    (0,0), (-1,-1), 2),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
                    ]))
                    cells.append(cell)
                except Exception:
                    continue
            for r in range(0, len(cells), 2):
                row = cells[r:r + 2]
                if len(row) == 1:
                    rt = Table([row], colWidths=[cw])
                else:
                    rt = Table([row], colWidths=[cw/2, cw/2])
                rt.setStyle(TableStyle([
                    ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                    ("VALIGN",        (0,0), (-1,-1), "TOP"),
                    ("TOPPADDING",    (0,0), (-1,-1), 3),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ]))
                story.append(rt)

        story.append(Spacer(1, 0.2*cm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=rl_grey))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Generated by Cubroid LMS  •  {grade} | Term {term} | Week {week} | {subject}",
        footer_s))
    doc.build(story)
    buf.seek(0)
    return buf


# ── GENERATION ────────────────────────────────────────────────────────────────
def generate_lesson_plan():
    notes_line = (f"Additional teacher notes: {st.session_state.custom_notes}"
                  if st.session_state.custom_notes else "")
    prompt = f"""
    Create a fully structured {st.session_state.duration} CAPS-aligned
    {SUBJECT} lesson plan for {st.session_state.grade} (South African
    Foundation Phase), Term {st.session_state.term},
    Week {st.session_state.week}. Robot used: {st.session_state.robot}.
    {notes_line}

    You have three kinds of source documents:
    1. The CAPS curriculum document ("{CAPS_DOC}") — use it for curriculum
       alignment.
    2. The Cubroid {TEACHER_GUIDE} — use these for the actual lesson
       activities.
    3. The Mission Guide ("{MISSION_GUIDE}") — use it ONLY for the
       "Further Work / Extension" section.

    LEVEL OF DETAIL — very important:
    - Write for a Foundation Phase teacher who may never have used the robot
      before. Nothing may be vague.
    - Every lesson phase (Hook, Direct Instruction, Guided Practice,
      Independent/Group Task, Wrap-up) must be a numbered sequence of
      teacher steps. Each step must say: what the teacher does, what the
      teacher says (give suggested wording in quotation marks), what the
      learners do, and roughly how many minutes it takes.
    - Wherever an activity comes from a teacher guide, cite it in-line in
      brackets, e.g. (Step1 Book 3, p.12).
    - EVERY reference to a source document must include a page number:
      (Step1 Book 3, p.12), (Mission Guide, p.7), (CAPS, p.14). Use the page
      numbers that appear in the retrieved content. Never cite a document
      without a page number.
    - Describe what the blocks/components in use look like (colour, shape,
      symbol) so the teacher can identify them, as described in the guide.
    - Include practical management detail: how to hand out the blocks, group
      sizes, what a finished example looks like, common mistakes to watch for.

    CAPS ALIGNMENT — very important:
    - Quote the specific study area, topic and skill CODES exactly as they
      are numbered in the CAPS document (e.g. "C.1", "C.2.1", "R.1") for
      {st.session_state.grade}, Term {st.session_state.term}.
    - List each code on its own bullet: the code, its official title, then
      one line on how this lesson addresses it.
    - Only use codes that actually appear in the retrieved CAPS content. If
      you cannot see the exact codes, name the topics instead and clearly
      state that the code reference should be verified — never invent codes.

    Format the output in markdown. Use "## " for every section heading
    (no bold-only headings, no numbered headings). Use "- " for bullets and
    "1." numbering for teacher steps. Produce exactly these sections:

    ## CAPS Alignment
    ## Learning Objectives
    ## Introduction / Hook (5 min)
    ## Direct Instruction (10 min)
    ## Guided Practice (15 min)
    ## Independent / Group Task (10 min)
    ## Wrap-up & Assessment (5 min)
    ## Resources Required
    ## Differentiation
    ## Teacher Guide Reference
    ## Further Work (Mission Guide)

    In "Teacher Guide Reference", list the specific Step 1 Book(s) and
    pages the lesson draws on, so the teacher can refer back to the book.
    In "Further Work (Mission Guide)", cite the exact Mission Guide pages,
    e.g. (Mission Guide, p.7).
    Keep activities age-appropriate for {st.session_state.grade}.
    Use only information from the provided documents.
    If something is not covered in the documents, say so.
    """
    engine = index.as_query_engine(similarity_top_k=10)
    response = engine.query(prompt)
    result_text = str(response)
    sources_md = format_sources(response.source_nodes)
    if sources_md:
        result_text += f"\n\n## Source Documents\n{sources_md}"
    st.session_state.result = result_text
    st.session_state.images = extract_page_screenshots(
        response.source_nodes, result_text)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.view == "home":
    st.markdown(f"""
    <div class="hero">
      <div class="badge">Foundation Phase &nbsp;•&nbsp; Grade R–3</div>
      <h1>Welcome, Teacher 👋</h1>
      <p>Create a ready-to-print {SUBJECT} lesson plan for your class,
      or get help with your Cubroid robot.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ Generate Lesson Plan", type="primary", use_container_width=True):
            st.session_state.view = "form"
            st.rerun()
    with col2:
        if st.button("🔧 Troubleshoot My Robot", use_container_width=True):
            st.session_state.view = "troubleshoot"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: FORM
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.view == "form":
    if st.button("← Back"):
        st.session_state.view = "home"
        st.rerun()

    st.markdown("""
    <div class="step-label">Lesson Plan</div>
    <div class="form-title">Tell us about your class</div>
    <div class="form-sub">We'll build a CAPS-aligned lesson plan around these details.</div>
    """, unsafe_allow_html=True)

    with st.form("lesson_form"):
        c1, c2 = st.columns(2)
        with c1:
            grade = st.selectbox("Grade", GRADES,
                                 index=GRADES.index(st.session_state.grade))
            term = st.selectbox("Term", [1, 2, 3, 4],
                                index=st.session_state.term - 1)
            week = st.number_input("Week", min_value=1, max_value=10,
                                   value=st.session_state.week)
        with c2:
            duration = st.selectbox("Lesson duration",
                                    ["30 minutes", "45 minutes", "60 minutes"])
            robot = st.selectbox("Robot kit",
                                 ["Cubroid Coding Blocks", "Cubroid Artibo"])

        c3, c4 = st.columns(2)
        with c3:
            teacher_name = st.text_input("Your name (optional)",
                                         value=st.session_state.teacher_name)
        with c4:
            school_name = st.text_input("School name (optional)",
                                        value=st.session_state.school_name)

        custom_notes = st.text_area(
            "Anything we should know about your class? (optional)",
            value=st.session_state.custom_notes,
            placeholder="e.g. 25 learners, first time using the robots, focus on teamwork...",
            height=90,
        )

        submitted = st.form_submit_button("✨ Generate Lesson Plan",
                                          type="primary", use_container_width=True)

    if submitted:
        st.session_state.update(
            grade=grade, term=term, week=week, duration=duration, robot=robot,
            teacher_name=teacher_name, school_name=school_name,
            custom_notes=custom_notes,
        )
        with st.spinner("Building your lesson plan — this takes about a minute..."):
            generate_lesson_plan()
        st.session_state.view = "result"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: RESULT
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.view == "result":
    top1, top2 = st.columns([3, 1])
    with top1:
        st.markdown(f"""
        <div class="step-label">Lesson Plan Ready</div>
        <div class="form-title">{st.session_state.grade} · Term {st.session_state.term} · Week {st.session_state.week}</div>
        <div class="form-sub">{SUBJECT} · {st.session_state.duration} · {st.session_state.robot}</div>
        """, unsafe_allow_html=True)
    with top2:
        if st.button("← Start Over", use_container_width=True):
            st.session_state.view = "home"
            st.session_state.result = None
            st.session_state.images = []
            st.rerun()

    pdf_buf = build_pdf(
        st.session_state.result,
        st.session_state.images,
        st.session_state.grade,
        SUBJECT,
        st.session_state.term,
        st.session_state.week,
        st.session_state.duration,
        st.session_state.teacher_name,
        st.session_state.school_name,
    )
    fname = (f"Cubroid_LP_{st.session_state.grade.replace(' ','')}"
             f"_T{st.session_state.term}_W{st.session_state.week}.pdf")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇️ Download PDF", data=pdf_buf, file_name=fname,
                           mime="application/pdf", type="primary",
                           use_container_width=True)
    with d2:
        if st.button("✏️ Change Details & Regenerate", use_container_width=True):
            st.session_state.view = "form"
            st.session_state.result = None
            st.session_state.images = []
            st.rerun()

    st.markdown("---")
    st.markdown(st.session_state.result)

    if st.session_state.images:
        st.markdown("---")
        st.markdown("##### 📎 Pages from your source documents (included in the PDF)")
        img_cols = st.columns(min(len(st.session_state.images), 3))
        for i, img_data in enumerate(st.session_state.images):
            with img_cols[i % len(img_cols)]:
                st.image(img_data["bytes"],
                         caption=f"{img_data['source']} p.{img_data['page']}",
                         use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: TROUBLESHOOT
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.view == "troubleshoot":
    if st.button("← Back"):
        st.session_state.view = "home"
        st.rerun()

    st.markdown("""
    <div class="step-label">Troubleshooting</div>
    <div class="form-title">What's going wrong?</div>
    <div class="form-sub">Describe the problem and we'll find the fix in the official Cubroid guides.</div>
    """, unsafe_allow_html=True)

    question = st.text_area(
        "Describe the problem",
        placeholder="e.g. The Cubroid blocks won't connect to the tablet via Bluetooth",
        height=120,
        label_visibility="collapsed",
    )
    if st.button("🔍 Find Solution", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please describe the problem first.")
        else:
            with st.spinner("Searching the guides..."):
                engine = index.as_query_engine(similarity_top_k=4)
                response = engine.query(f"""
                    A teacher has this problem with a Cubroid robot: {question}
                    Using only the troubleshooting guides provided, give a clear
                    numbered step-by-step solution.
                    If the answer is not in the documents, say so clearly.
                """)
            st.success("Here's what the guides say:")
            st.markdown(str(response))
