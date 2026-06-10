import streamlit as st
import openai
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cubroid Lesson Plan Generator", layout="centered")

openai.api_key = st.secrets["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
Settings.embed_model = OpenAIEmbedding(api_key=st.secrets["OPENAI_API_KEY"])

# ── Load & index documents ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_index():
    docs = SimpleDirectoryReader("docs").load_data()
    return VectorStoreIndex.from_documents(docs)

index = load_index()

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🤖 Cubroid Lesson Plan Generator")
st.markdown("Fill in the details below to generate a complete lesson plan.")

mode = st.radio("Mode", ["📋 Lesson Plan Generator", "🔧 Troubleshooting Assistant"], horizontal=True)
st.divider()

if mode == "📋 Lesson Plan Generator":
    col1, col2 = st.columns(2)
    with col1:
        grade    = st.selectbox("Grade", ["Grade 4","Grade 5","Grade 6","Grade 7","Grade 8","Grade 9"])
        term     = st.selectbox("Term", [1, 2, 3, 4])
        week     = st.number_input("Week", min_value=1, max_value=10, value=1)
    with col2:
        subject  = st.selectbox("Subject", ["Technology","Natural Sciences","Mathematics","Life Skills"])
        robot    = st.selectbox("Robot Type", ["Cuboid Mini","Cuboid Pro","Cuboid Starter"])
        duration = st.selectbox("Lesson Duration", ["45 minutes","60 minutes"])

    if st.button("Generate Lesson Plan", type="primary", use_container_width=True):
        prompt = f"""
        Create a fully structured {duration} lesson plan for {grade}, Term {term}, Week {week}.
        Subject: {subject}. Robot used: {robot}.

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
        with st.spinner("Generating lesson plan..."):
            engine   = index.as_query_engine(similarity_top_k=5)
            response = engine.query(prompt)
            result   = str(response)

        st.success("Lesson plan ready!")
        st.markdown(result)
        st.download_button(
            "⬇️ Download Lesson Plan",
            data=result,
            file_name=f"lesson_plan_{grade.replace(' ','')}_T{term}_W{week}.txt",
            mime="text/plain",
            use_container_width=True
        )

else:  # Troubleshooting mode
    question = st.text_area(
        "Describe the problem",
        placeholder="e.g. The Cuboid Mini won't connect to the tablet",
        height=120
    )
    if st.button("Find Solution", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please describe the problem first.")
        else:
            prompt = f"""
            A teacher has the following problem with a Cubroid robot: {question}

            Using only the troubleshooting guides and software specs provided,
            give a clear numbered step-by-step solution.
            If the answer is not in the documents, say so clearly.
            """
            with st.spinner("Searching guides..."):
                engine   = index.as_query_engine(similarity_top_k=4)
                response = engine.query(prompt)

            st.success("Solution found!")
            st.markdown(str(response))
