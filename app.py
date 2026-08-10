import streamlit as st
import ollama
import numpy as np
import time
from pypdf import PdfReader

# Load benchmark data, provide mock backup if missing
try:
    from test_cases import EVAL_DATASET
except ImportError:
    EVAL_DATASET = [
        {
            "id": "TC-001",
            "category": "Customer Support",
            "context": "Our refund policy allows returns within 30 days of purchase with a receipt.",
            "question": "What is the return window?",
            "expected_keywords": ["30 days", "receipt"]
        }
    ]

# 1. Page Configuration
st.set_page_config(page_title="Aegis Eval Harness", page_icon="🛡️", layout="wide")

# Initialize Session State for tracking interactions
if "results_summary" not in st.session_state:
    st.session_state.results_summary = []
if "full_transcript" not in st.session_state:
    st.session_state.full_transcript = "Aegis Eval Harness - Run Transcript\n===================================\n"
if "manual_history" not in st.session_state:
    st.session_state.manual_history = []
if "uploaded_pdf_text" not in st.session_state:
    st.session_state.uploaded_pdf_text = ""
if "pdf_metadata" not in st.session_state:
    st.session_state.pdf_metadata = {"chars": 0, "pages": 0}

# 2. Styling
st.markdown("""
    <style>
    .glow-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.05em;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="glow-title">🛡️ Aegis Automated Eval Harness</h1>', unsafe_allow_html=True)
st.write("Programmatically bench-test applications or ask direct custom questions against your documents.")

# Calculate stats
base_characters = sum(len(str(case.values())) for case in EVAL_DATASET)
base_pages = max(1, base_characters // 1500)
active_total_characters = base_characters + st.session_state.pdf_metadata["chars"]
active_total_pages = base_pages + st.session_state.pdf_metadata["pages"]
active_vector_blocks = len(EVAL_DATASET) + (1 if st.session_state.uploaded_pdf_text else 0)

# 3. Sidebar Controls
st.sidebar.markdown("### ⚙️ Harness Controls")
st.sidebar.write("Total Benchmark Test Cases Loaded:", len(EVAL_DATASET))

# PDF Uploader
st.sidebar.markdown("### 📂 Upload Knowledge Documents")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document to query:", type=["pdf"])

if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)
        raw_text_list = []
        for page in reader.pages:
            text_content = page.extract_text()
            if text_content:
                raw_text_list.append(text_content)
        full_pdf_text = "\n".join(raw_text_list)
        st.session_state.uploaded_pdf_text = full_pdf_text
        st.session_state.pdf_metadata["chars"] = len(full_pdf_text)
        st.session_state.pdf_metadata["pages"] = len(reader.pages)
        st.sidebar.success(f"Loaded: {len(reader.pages)} PDF Pages!")
    except Exception as e:
        st.sidebar.error(f"Error parsing PDF: {e}")

# Session Purge Button
if st.sidebar.button("🧼 Session Reset & Data Purge", use_container_width=True):
    st.session_state.results_summary = []
    st.session_state.full_transcript = "Aegis Eval Harness - Run Transcript\n===================================\n"
    st.session_state.manual_history = []
    st.session_state.uploaded_pdf_text = ""
    st.session_state.pdf_metadata = {"chars": 0, "pages": 0}
    st.rerun()

# --- MAIN LAYOUT: SPLIT INTO TWO VISIBLE COLUMNS ---
col_left, col_right = st.columns([1, 1], gap="large")

# LEFT COLUMN: INTERACTIVE QUESTION MODE (Type anything here)
with col_left:
    st.markdown("## 💬 Interactive Question Mode")
    st.write("Type a custom question below. You can test your uploaded PDF or generic text here.")
    
    # Choose what the AI should read to answer your typed question
    context_options = ["Use a Blank Custom Context Box", "Inherit from a Dataset Reference Case"]
    if st.session_state.uploaded_pdf_text:
        context_options.append("🔥 Use Content from Uploaded PDF Document")
        
    context_source = st.radio("What text should the AI read to answer your question?", context_options, key="ctx_src")
    
    selected_context = ""
    if context_source == "Inherit from a Dataset Reference Case":
        case_options = {f"{case['id']} - {case['category']}": case for case in EVAL_DATASET}
        chosen_option = st.selectbox("Select reference case:", list(case_options.keys()))
        selected_context = case_options[chosen_option]["context"]
    elif context_source == "🔥 Use Content from Uploaded PDF Document":
        selected_context = st.session_state.uploaded_pdf_text
        st.info(f"✅ AI will look inside your uploaded PDF ({st.session_state.pdf_metadata['pages']} pages loaded).")
    else:
        selected_context = st.text_area("Type or paste temporary context text here:", value="The secret password is apple123.")

    # THE TYPING INPUT BOX
    user_custom_question = st.text_input("✍️ Type your custom question here and press Enter:")
    
    if user_custom_question:
        start_manual_time = time.time()
        
        with st.spinner("AI is thinking..."):
            try:
                manual_response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {"role": "system", "content": "Answer the question briefly using ONLY the provided context."},
                        {"role": "user", "content": f"Context: {selected_context}\nQuestion: {user_custom_question}"}
                    ],
                    options={"temperature": 0.0}
                )
                manual_output = manual_response['message']['content']
            except Exception:
                manual_output = "Error: Could not connect to your local Ollama server. Make sure Ollama is running Llama3.2."
        
        end_manual_time = time.time()
        
        # Save to chat history
        st.session_state.manual_history.insert(0, {
            "question": user_custom_question,
            "answer": manual_output,
            "context": selected_context[:300] + "...",
            "latency": end_manual_time - start_manual_time
        })
        st.session_state.full_transcript += f"\n[User Question]: {user_custom_question}\n[AI Answer]: {manual_output}\n"

    # Display your typed question answers
    if st.session_state.manual_history:
        st.markdown("### 🕒 Response History")
        for idx, item in enumerate(st.session_state.manual_history):
            st.info(f"**Question:** {item['question']}\n\n🤖 **AI Output:** {item['answer']}")
            with st.expander("🔍 View Raw Text Source Used"):
                st.write(item['context'])
            st.caption(f"⏱️ Speed: {item['latency']:.2f} seconds")

# RIGHT COLUMN: AUTOMATED SUITE RUNNER
with col_right:
    st.markdown("## 📊 Automated Benchmark Suite")
    st.write("Click below to test the entire pre-built evaluation dataset instantly.")
    
    run_button = st.button("⚡ Run Core Benchmark Validation Suite", use_container_width=True)
    
    if run_button:
        st.subheader("🏃‍♂️ Live Execution Stream")
        progress_bar = st.progress(0)
        st.session_state.results_summary = []
        
        for idx, case in enumerate(EVAL_DATASET):
            start_case_time = time.time()
            
            try:
                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {"role": "system", "content": "Answer using context only."},
                        {"role": "user", "content": f"Context: {case['context']}\nQuestion: {case['question']}"}
                    ],
                    options={"temperature": 0.0}
                )
                ai_output = response['message']['content'].lower()
            except Exception:
                ai_output = "ollama engine connection offline"
            
            matched_keywords = [word for word in case['expected_keywords'] if word.lower() in ai_output]
            accuracy_score = (len(matched_keywords) / len(case['expected_keywords'])) * 100 if case['expected_keywords'] else 0
            
            st.session_state.results_summary.append({
                "id": case["id"], "score": accuracy_score, "category": case["category"], "latency": time.time() - start_case_time
            })
            
            st.write(f"🔹 **{case['id']}**: Match Score = {accuracy_score:.1f}%")
            progress_bar.progress((idx + 1) / len(EVAL_DATASET))
            time.sleep(0.05)

    # Show Telemetry and Metrics if a benchmark run happened
    if st.session_state.results_summary:
        st.markdown("### ⏱️ System Telemetry & Run Artifacts")
        t1, t2 = st.columns(2)
        t1.metric("Total Loaded Characters", f"{active_total_characters} chars")
        t2.metric("Data Vector Blocks", f"{active_vector_blocks} nodes")
        
        st.download_button(
            label="📥 Download Summary Transcript File",
            data=st.session_state.full_transcript,
            file_name="eval_transcript.txt",
            mime="text/plain",
            use_container_width=True
        )