import streamlit as st  # Cubroid Lesson Generator v2
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
    layout="wide",
)

# ── Custom CSS — clean, minimal, polished ─────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
  .block-container { padding-top: 2.5rem; max-width: 1100px; }

  /* Sidebar — light, quiet */
  section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
  }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
  section[data-testid="stSidebar"] label { font-size: 0.8rem !important; color: #475569 !important; }

  /* Hero */
  .hero {
    text-align: center;
    padding: 2.5rem 1rem 2rem 1rem;
  }
  .hero .badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1E3A8A;
    border: 1px solid #BFDBFE;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 1rem;
  }
  .hero h1 {
    color: #0F172A;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
  }
  .hero h1 span { color: #1E3A8A; }
  .hero p {
    color: #64748B;
    font-size: 1.05rem;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.6;
  }

  /* Config summary pills */
  .pill-row { text-align: center; margin: 1.2rem 0 2rem 0; }
  .pill {
    display: inline-block;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 999px;
    padding: 6px 16px;
    margin: 4px;
    font-size: 0.85rem;
    color: #334155;
    font-weight: 500;
  }
  .pill b { color: #1E3A8A; font-weight: 700; }

  /* Section cards */
  .card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: box-shadow .15s ease, border-color .15s ease;
  }
  .card:hover { box-shadow: 0 4px 16px rgba(15,23,42,0.06); border-color: #CBD5E1; }
  .card h4 {
    margin: 0 0 6px 0;
    color: #0F172A;
    font-size: 0.92rem;
    font-weight: 700;
  }
  .card .num {
    display: inline-block;
    width: 22px; height: 22px;
    line-height: 22px;
    text-align: center;
    background: #1E3A8A;
    color: white;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-right: 8px;
  }
  .card p { margin: 0; color: #64748B; font-size: 0.84rem; line-height: 1.5; }

  /* Sources strip */
  .sources {
    background: #F8FAFC;
    border: 1px dashed #CBD5E1;
    border-radius: 14px;
    padding: 16px 20px;
    margin-top: 0.5rem;
    font-size: 0.84rem;
    color: #475569;
    line-height: 1.7;
  }
  .sources b { color: #1E3A8A; }

  div.stButton > button, div.stDownloadButton > button { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── OpenAI / LlamaIndex setup ─────────────────────────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
Settings.embed_model = OpenAIEmbedding(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource(show_spinner="Loading knowledge base...")
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
    "step": 1,
    "grade": "Grade R",
    "term": 1,
    "week": 1,
    "robot": "Cubroid Coding Blocks",
    "duration": "30 minutes",
    "teacher_name": "",
    "school_name": "",
    "custom_notes": "",
    "mode": "lesson",
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
    re.compile(r"^\d+\.\s+([A-Z][A-Za-z &/()'\-]{2,50}):?\s*$"),  # 3. Heading
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


def extract_page_screenshots(source_nodes, max_shots=4, zoom=2.0):
    """Render full-page screenshots of the source pages the answer drew on."""
    shots, seen = [], set()
    for node in source_nodes:
        meta = node.metadata or {}
        fp = find_doc(meta.get("file_path") or meta.get("file_name", ""))
        if not fp:
            continue
        try:
            page_num = int(str(meta.get("page_label", meta.get("page", 1)))) - 1
        except (TypeError, ValueError):
            page_num = 0
        name = os.path.basename(fp)
        key = (name, page_num)
        if key in seen:
            continue
        seen.add(key)
        try:
            pdf_doc = fitz.open(fp)
            page = pdf_doc[max(0, min(page_num, len(pdf_doc) - 1))]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            pil = PILImage.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=82)
            shots.append({
                "bytes": buf.getvalue(),
                "source": name,
                "page": page_num + 1,
            })
            pdf_doc.close()
        except Exception:
            continue
        if len(shots) >= max_shots:
            break
    return shots


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

    # Distribute screenshots across body sections
    sections = parse_sections(text)
    img_map = {}
    if images and len(sections) > 2:
        body_idxs = list(range(1, len(sections)))
        step = max(1, len(body_idxs) // len(images))
        for k, img_data in enumerate(images):
            idx = body_idxs[min(k * step, len(body_idxs) - 1)]
            img_map.setdefault(idx, img_data)

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

        # Embed screenshot if assigned to this section
        if i in img_map:
            try:
                img_data = img_map[i]
                img_buf = io.BytesIO(img_data["bytes"])
                pil_img = PILImage.open(img_buf)
                ow, oh = pil_img.size
                max_w = cw * 0.55
                scale = min(max_w / ow, (7*cm) / oh)
                dw, dh = ow * scale, oh * scale
                img_buf.seek(0)
                rl_img = RLImage(img_buf, width=dw, height=dh)
                cap_p = Paragraph(
                    f"Source: {img_data['source']}, p.{img_data['page']}", caption)
                img_tbl = Table([[rl_img], [cap_p]], colWidths=[dw])
                img_tbl.setStyle(TableStyle([
                    ("ALIGN", (0,0), (-1,-1), "CENTER"),
                    ("BOX", (0,0), (0,0), 0.5, colors.HexColor("#CBD5E1")),
                    ("TOPPADDING", (0,0), (-1,-1), 3),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ]))
                wrap = Table([[img_tbl]], colWidths=[cw])
                wrap.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER")]))
                story.append(Spacer(1, 0.2*cm))
                story.append(wrap)
            except Exception:
                pass

        story.append(Spacer(1, 0.2*cm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=rl_grey))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"Generated by Cubroid LMS  •  {grade} | Term {term} | Week {week} | {subject}",
        footer_s))
    doc.build(story)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"## 🤖 Cubroid LMS")
    st.caption(f"Foundation Phase • {SUBJECT}")
    st.markdown("---")
    mode_choice = st.radio("Mode", ["📋 Lesson Planner", "🔧 Troubleshooting"],
                           label_visibility="collapsed")
    st.session_state.mode = "lesson" if "Lesson" in mode_choice else "troubleshoot"

    if st.session_state.mode == "lesson":
        st.markdown("##### Class details")
        st.session_state.grade = st.selectbox(
            "Grade", GRADES, index=GRADES.index(st.session_state.grade))
        robots = ["Cubroid Coding Blocks", "Cubroid Artibo"]
        st.session_state.robot = st.selectbox("Robot kit", robots)
        st.session_state.term = st.selectbox("Term", [1, 2, 3, 4],
                                             index=st.session_state.term - 1)
        st.session_state.week = st.number_input("Week", min_value=1, max_value=10,
                                                value=st.session_state.week)
        st.session_state.duration = st.selectbox(
            "Duration", ["30 minutes", "45 minutes", "60 minutes"])

        st.markdown("##### Teacher info")
        st.session_state.teacher_name = st.text_input(
            "Teacher name", value=st.session_state.teacher_name)
        st.session_state.school_name = st.text_input(
            "School name", value=st.session_state.school_name)

        st.markdown("##### Notes")
        st.session_state.custom_notes = st.text_area(
            "Extra context for the AI",
            value=st.session_state.custom_notes,
            placeholder="e.g. Class has 25 learners, focus on teamwork...",
            height=90,
        )
        st.markdown("---")
        if st.button("✨ Generate Lesson Plan", type="primary", use_container_width=True):
            st.session_state.step = 2
            st.session_state.result = None
            st.session_state.images = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# ── LESSON PLANNER ────────────────────────────────────────────────────────────
if st.session_state.mode == "lesson":

    # Trigger generation
    if st.session_state.step == 2 and st.session_state.result is None:
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
           alignment: name the specific content area, topic and skills for this
           grade and term.
        2. The Cubroid {TEACHER_GUIDE} — use these for the actual lesson
           activities. You MUST state exactly which Step 1 Book (and pages,
           if available) the activities come from.
        3. The Mission Guide ("{MISSION_GUIDE}") — use it ONLY for the
           "Further Work / Extension" section.

        Format the output in markdown. Use "## " for every section heading
        (no bold-only headings, no numbered headings). Use "- " for bullets.
        Produce exactly these sections:

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
        Keep activities age-appropriate for {st.session_state.grade}.
        Use only information from the provided documents.
        If something is not covered in the documents, say so.
        """
        with st.spinner("Generating lesson plan and capturing source pages..."):
            engine = index.as_query_engine(similarity_top_k=8)
            response = engine.query(prompt)
            result_text = str(response)
            sources_md = format_sources(response.source_nodes)
            if sources_md:
                result_text += f"\n\n## Source Documents\n{sources_md}"
            st.session_state.result = result_text
            st.session_state.images = extract_page_screenshots(response.source_nodes)
        st.session_state.step = 1
        st.rerun()

    # Idle state — landing page
    if st.session_state.result is None:
        st.markdown(f"""
        <div class="hero">
          <div class="badge">Foundation Phase &nbsp;•&nbsp; CAPS Aligned</div>
          <h1>Cubroid <span>Lesson Generator</span></h1>
          <p>Polished, print-ready {SUBJECT} lesson plans for Grade R–3 —
          grounded in the CAPS curriculum, Step 1 Teacher Guides and
          Mission Guides, with source-page screenshots included.</p>
        </div>
        <div class="pill-row">
          <span class="pill"><b>{st.session_state.grade}</b></span>
          <span class="pill">{SUBJECT}</span>
          <span class="pill">Term <b>{st.session_state.term}</b> · Week <b>{st.session_state.week}</b></span>
          <span class="pill">{st.session_state.duration}</span>
          <span class="pill">{st.session_state.robot}</span>
        </div>
        """, unsafe_allow_html=True)

        preview_sections = [
            ("CAPS Alignment",          f"Content area, topic & skills from the official CAPS {SUBJECT} document"),
            ("Learning Objectives",     "Three curriculum-aligned objectives for the lesson"),
            ("Lesson Flow",             "Hook, direct instruction, guided practice, group task and wrap-up — fully timed"),
            ("Teacher Guide Reference", "The exact Step 1 Book and pages the activities come from"),
            ("Further Work",            "Extension activities drawn from the Cubroid Mission Guide"),
            ("Source Screenshots",      "Page images captured from the actual curriculum and guide documents"),
        ]
        cols = st.columns(2)
        for idx, (title, desc) in enumerate(preview_sections):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="card">
                  <h4><span class="num">{idx + 1}</span>{title}</h4>
                  <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sources">
          <b>Knowledge base</b> &nbsp;—&nbsp; {CAPS_DOC} (Curriculum)
          &nbsp;·&nbsp; {TEACHER_GUIDE} &nbsp;·&nbsp; {MISSION_GUIDE}
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.custom_notes:
            st.markdown(f"""
            <div class="card" style="margin-top:14px">
              <h4>Your notes</h4>
              <p>{st.session_state.custom_notes}</p>
            </div>
            """, unsafe_allow_html=True)

        st.info("👈  Set the class details in the sidebar, then click **Generate Lesson Plan**")

    # Result state
    else:
        st.markdown("### ✅ Lesson Plan Ready")

        if st.session_state.images:
            with st.expander(
                f"🖼️ {len(st.session_state.images)} source-page screenshots "
                "(included in the PDF)",
                expanded=False
            ):
                img_cols = st.columns(min(len(st.session_state.images), 4))
                for i, img_data in enumerate(st.session_state.images):
                    with img_cols[i % len(img_cols)]:
                        st.image(img_data["bytes"],
                                 caption=f"{img_data['source']} p.{img_data['page']}",
                                 use_container_width=True)

        col_text, col_dl = st.columns([3, 1])
        with col_text:
            st.markdown(st.session_state.result)

        with col_dl:
            st.markdown("#### 📥 Export")
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
            st.download_button(
                "⬇️ Download PDF",
                data=pdf_buf,
                file_name=fname,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
            st.download_button(
                "📄 Download Text",
                data=st.session_state.result,
                file_name=fname.replace(".pdf", ".txt"),
                mime="text/plain",
                use_container_width=True,
            )
            if st.button("🔄 Generate New", use_container_width=True):
                st.session_state.result = None
                st.session_state.images = []
                st.rerun()


# ── TROUBLESHOOTING ───────────────────────────────────────────────────────────
else:
    st.markdown("### 🔧 Troubleshooting Assistant")
    st.markdown("Describe a problem and get step-by-step guidance from the official Cubroid guides.")
    question = st.text_area(
        "What is the problem?",
        placeholder="e.g. The Cubroid blocks won't connect to the tablet via Bluetooth",
        height=120,
    )
    if st.button("🔍 Find Solution", type="primary"):
        if not question.strip():
            st.warning("Please describe the problem first.")
        else:
            with st.spinner("Searching guides..."):
                engine = index.as_query_engine(similarity_top_k=4)
                response = engine.query(f"""
                    A teacher has this problem with a Cubroid robot: {question}
                    Using only the troubleshooting guides provided, give a clear
                    numbered step-by-step solution.
                    If the answer is not in the documents, say so clearly.
                """)
            st.success("Solution found!")
            st.markdown(str(response))
