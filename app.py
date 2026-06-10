import streamlit as st
import openai
import os
import io
import fitz  # PyMuPDF
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from PIL import Image as PILImage

# ── Brand colours ─────────────────────────────────────────────────────────────
CUBROID_BLUE   = "#1E3A8A"
CUBROID_ORANGE = "#F97316"
CUBROID_LIGHT  = "#EFF6FF"
CUBROID_GREY   = "#64748B"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cubroid Lesson Generator",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  section[data-testid="stSidebar"] { background: #1E3A8A; }
  section[data-testid="stSidebar"] * { color: white !important; }
  .section-card {
    background: #EFF6FF;
    border-left: 4px solid #F97316;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
  }
  .section-card h4 {
    margin: 0 0 8px 0;
    color: #1E3A8A;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .app-header h1 { color: #1E3A8A; font-size: 1.6rem; font-weight: 700; margin: 0; }
  .app-header p  { color: #64748B; margin: 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── OpenAI / LlamaIndex setup ─────────────────────────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
Settings.embed_model = OpenAIEmbedding(api_key=st.secrets["OPENAI_API_KEY"])

from llama_parse import LlamaParse

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_index():
    parser = LlamaParse(
        api_key=st.secrets["llx-640ulN0pgNuumU8IqrW68PT9mzcZbqX9KhknqMCiNKOD4kKZ"],
        result_type="markdown"
    )
    docs = SimpleDirectoryReader("docs", file_extractor={".pdf": parser}).load_data()
    return VectorStoreIndex.from_documents(docs)

index = load_index()

# ── Session state defaults ────────────────────────────────────────────────────
DEFAULTS = {
    "step": 1,
    "grade": "Grade 4",
    "term": 1,
    "week": 1,
    "subject": "Technology",
    "robot": "Cuboid Mini",
    "duration": "45 minutes",
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


# ── IMAGE EXTRACTION ──────────────────────────────────────────────────────────
def extract_images_from_nodes(source_nodes, max_images=4, min_bytes=8000):
    images = []
    seen = set()
    docs_dir = "docs"
    for node in source_nodes:
        meta = node.metadata or {}
        fp = meta.get("file_path") or meta.get("file_name", "")
        page_num = int(meta.get("page_label", meta.get("page", 1))) - 1
        if not os.path.isabs(fp):
            fp = os.path.join(docs_dir, fp)
        if not os.path.exists(fp):
            continue
        try:
            pdf_doc = fitz.open(fp)
            for p in [max(0, page_num), min(page_num + 1, len(pdf_doc) - 1)]:
                for img in pdf_doc[p].get_images(full=True):
                    xref = img[0]
                    if xref in seen:
                        continue
                    base = pdf_doc.extract_image(xref)
                    raw = base["image"]
                    if len(raw) < min_bytes:
                        continue
                    try:
                        pil = PILImage.open(io.BytesIO(raw))
                        w, h = pil.size
                        if w < 80 or h < 80 or max(w, h) / max(min(w, h), 1) > 6:
                            continue
                        buf = io.BytesIO()
                        pil.convert("RGB").save(buf, format="JPEG", quality=85)
                        images.append({
                            "bytes": buf.getvalue(),
                            "source": os.path.basename(fp),
                            "page": p + 1,
                        })
                        seen.add(xref)
                    except Exception:
                        continue
                    if len(images) >= max_images:
                        return images
        except Exception:
            continue
    return images


# ── PDF GENERATION ────────────────────────────────────────────────────────────
def parse_sections(text):
    import re
    sections = []
    title = "Overview"
    body = []
    for line in text.splitlines():
        m = re.match(r'^(?:#+\s*|(\d+)\.\s+)(.+)$', line)
        if m:
            if body:
                sections.append((title, "\n".join(body).strip()))
            title = (m.group(2) or m.group(0)).strip()
            body = []
        else:
            body.append(line)
    if body:
        sections.append((title, "\n".join(body).strip()))
    return sections


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
    sub_s      = ParagraphStyle("sub", fontName="Helvetica", fontSize=10,
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

    # Distribute images across body sections
    sections = parse_sections(text)
    img_map = {}
    if images:
        body_idxs = [i for i in range(2, len(sections) - 1)]
        step = max(1, len(body_idxs) // len(images))
        for k, img_data in enumerate(images):
            idx = body_idxs[min(k * step, len(body_idxs) - 1)]
            img_map[idx] = img_data

    for i, (sec_title, sec_body) in enumerate(sections):
        if not sec_title and not sec_body:
            continue
        # Heading with orange left bar
        ht = Table([[Paragraph(sec_title.upper(), heading_style)]], colWidths=[cw])
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
                story.append(Paragraph("• " + line[2:], bullet_style))
            elif len(line) > 2 and line[0].isdigit() and line[1] == ".":
                story.append(Paragraph(line, bullet_style))
            else:
                story.append(Paragraph(line, body_style))

        # Embed image if assigned to this section
        if i in img_map:
            try:
                img_data = img_map[i]
                img_buf = io.BytesIO(img_data["bytes"])
                pil_img = PILImage.open(img_buf)
                ow, oh = pil_img.size
                max_w = cw * 0.42
                scale = min(max_w / ow, (4.5*cm) / oh)
                dw, dh = ow * scale, oh * scale
                img_buf.seek(0)
                rl_img = RLImage(img_buf, width=dw, height=dh)
                cap_p = Paragraph(f"Source: {img_data['source']} p.{img_data['page']}", caption)
                img_tbl = Table([[rl_img], [cap_p]], colWidths=[dw])
                img_tbl.setStyle(TableStyle([
                    ("ALIGN", (0,0), (-1,-1), "CENTER"),
                    ("TOPPADDING", (0,0), (-1,-1), 3),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ]))
                layout = Table([["", img_tbl]], colWidths=[cw - dw - 0.5*cm, dw + 0.5*cm])
                layout.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
                story.append(layout)
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
    st.markdown("## 🤖 Cubroid LMS")
    st.markdown("---")
    mode_choice = st.radio("Mode", ["📋 Lesson Planner", "🔧 Troubleshooting"],
                           label_visibility="collapsed")
    st.session_state.mode = "lesson" if "Lesson" in mode_choice else "troubleshoot"

    if st.session_state.mode == "lesson":
        st.markdown("### 📐 Class Details")
        grades = ["Grade 4","Grade 5","Grade 6","Grade 7","Grade 8","Grade 9"]
        st.session_state.grade = st.selectbox(
            "Grade", grades, index=grades.index(st.session_state.grade))
        subjects = ["Technology","Natural Sciences","Mathematics","Life Skills"]
        st.session_state.subject = st.selectbox("Subject", subjects)
        robots = ["Cuboid Mini","Cuboid Pro","Cuboid Starter"]
        st.session_state.robot = st.selectbox("Robot", robots)
        st.session_state.term = st.selectbox("Term", [1,2,3,4],
                                             index=st.session_state.term - 1)
        st.session_state.week = st.number_input("Week", min_value=1, max_value=10,
                                                value=st.session_state.week)
        st.session_state.duration = st.selectbox("Duration", ["45 minutes","60 minutes"])

        st.markdown("### 👤 Teacher Info")
        st.session_state.teacher_name = st.text_input(
            "Teacher Name", value=st.session_state.teacher_name)
        st.session_state.school_name = st.text_input(
            "School Name", value=st.session_state.school_name)

        st.markdown("### 📝 Custom Notes")
        st.session_state.custom_notes = st.text_area(
            "Extra context for the AI",
            value=st.session_state.custom_notes,
            placeholder="e.g. Class has 25 learners, focus on teamwork...",
            height=100,
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
st.markdown("""
<div class="app-header">
  <h1>Cubroid Lesson Generator</h1>
  <p>AI-powered lesson plans grounded in Cubroid curriculum documents</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── LESSON PLANNER ────────────────────────────────────────────────────────────
if st.session_state.mode == "lesson":

    # Trigger generation
    if st.session_state.step == 2 and st.session_state.result is None:
        notes_line = (f"Additional teacher notes: {st.session_state.custom_notes}"
                      if st.session_state.custom_notes else "")
        prompt = f"""
        Create a fully structured {st.session_state.duration} lesson plan for
        {st.session_state.grade}, Term {st.session_state.term},
        Week {st.session_state.week}. Subject: {st.session_state.subject}.
        Robot used: {st.session_state.robot}. {notes_line}

        Structure it with these exact sections:
        1. Header (grade, subject, term, week, duration, CAPS alignment)
        2. Learning Objectives (3 bullet points)
        3. Introduction / Hook (5 min)
        4. Direct Instruction (10 min)
        5. Guided Practice (15 min)
        6. Independent / Group Task (10 min)
        7. Wrap-up & Assessment (5 min)
        8. Resources Required
        9. Differentiation (support strategies + extension activities)

        Use only information from the provided documents.
        If something is not covered in the documents, say so.
        """
        with st.spinner("✨ Generating lesson plan and pulling images from source docs..."):
            engine = index.as_query_engine(similarity_top_k=6)
            response = engine.query(prompt)
            st.session_state.result = str(response)
            st.session_state.images = extract_images_from_nodes(response.source_nodes)
        st.session_state.step = 1
        st.rerun()

    # Idle state — preview
    if st.session_state.result is None:
        st.markdown("### 📋 What will be generated")
        st.markdown(f"""
        <div class="section-card">
          <h4>📐 Class Configuration</h4>
          <b>{st.session_state.grade}</b> &nbsp;|&nbsp;
          <b>{st.session_state.subject}</b> &nbsp;|&nbsp;
          Term {st.session_state.term}, Week {st.session_state.week} &nbsp;|&nbsp;
          {st.session_state.duration} &nbsp;|&nbsp;
          {st.session_state.robot}
        </div>
        """, unsafe_allow_html=True)

        preview_sections = [
            ("🎯", "Learning Objectives",     "3 curriculum-aligned objectives"),
            ("🔥", "Introduction / Hook",     "~5 min engaging opener"),
            ("📖", "Direct Instruction",      "~10 min concept delivery"),
            ("🤝", "Guided Practice",         "~15 min teacher-led activity"),
            ("🧪", "Independent / Group Task", f"~10 min learner task with {st.session_state.robot}"),
            ("✅", "Wrap-up & Assessment",    "~5 min reflection + check"),
            ("📦", "Resources Required",      "Full materials list"),
            ("♿", "Differentiation",         "Support + extension strategies"),
        ]
        cols = st.columns(2)
        for idx, (icon, title, desc) in enumerate(preview_sections):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="section-card">
                  <h4>{icon} {title}</h4>
                  <span style="color:{CUBROID_GREY};font-size:0.85rem">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        if st.session_state.custom_notes:
            st.markdown(f"""
            <div class="section-card" style="border-left-color:{CUBROID_BLUE}">
              <h4>📝 Your Custom Notes</h4>
              <span style="font-size:0.9rem">{st.session_state.custom_notes}</span>
            </div>
            """, unsafe_allow_html=True)

        st.info("👈  Adjust settings in the sidebar, then click **✨ Generate Lesson Plan**")

    # Result state
    else:
        st.markdown("### ✅ Lesson Plan Ready")

        if st.session_state.images:
            with st.expander(
                f"🖼️ {len(st.session_state.images)} images pulled from source documents",
                expanded=False
            ):
                img_cols = st.columns(min(len(st.session_state.images), 4))
                for i, img_data in enumerate(st.session_state.images):
                    with img_cols[i]:
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
                st.session_state.subject,
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
        placeholder="e.g. The Cuboid Mini won't connect to the tablet via Bluetooth",
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
