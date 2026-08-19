import os
import time
import streamlit as st
import pypdf

st.set_page_config(page_title="Aegis Evaluation Harness", page_icon="🛡️", layout="wide")

# ==========================================
# GROQ INFERENCE & MODEL DISCOVERY CORE
# ==========================================

def get_groq_client():
    """Extracts GROQ_API_KEY from Streamlit secrets or OS environment."""
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = str(st.secrets["GROQ_API_KEY"]).strip()
        elif hasattr(st.secrets, "get") and st.secrets.get("GROQ_API_KEY"):
            api_key = str(st.secrets.get("GROQ_API_KEY")).strip()
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if not api_key:
        return None, "GROQ_API_KEY secret not found in Streamlit Secrets or Environment."

    try:
        from groq import Groq
        return Groq(api_key=api_key), None
    except Exception as e:
        return None, f"Groq initialization failed: {e}"


def run_llm_inference(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """Executes inference via local Ollama first, falling back to dynamic Groq Cloud models."""
    # 1. Local Ollama Attempt
    try:
        import ollama
        res = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": temperature}
        )
        if res and "message" in res and "content" in res["message"]:
            return res["message"]["content"].strip()
    except Exception:
        pass

    # 2. Groq Cloud Fallback
    client, err = get_groq_client()
    if err:
        return f"Configuration Error: {err}"

    candidate_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "llama3-70b-8192",
        "llama3-8b-8192"
    ]

    try:
        active_models = [
            m.id for m in client.models.list().data 
            if not any(x in m.id for x in ["whisper", "guard", "vision", "embed"])
        ]
        if active_models:
            candidate_models = active_models + candidate_models
    except Exception:
        pass

    last_error = ""
    for model_id in candidate_models:
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            last_error = str(e)
            continue

    return f"Cloud Inference Error: {last_error}"


# ==========================================
# BENCHMARK EVALUATION ENGINE
# ==========================================

BENCHMARK_SUITE = [
    {
        "id": "TC-01",
        "category": "Deterministic Arithmetic",
        "context": "Annual system operations report indicates 24 server nodes running at $150 per month each.",
        "question": "What is the total annual cost to run all 24 server nodes?",
        "ground_truth": "$43,200",
        "eval_keywords": ["43200", "43,200"]
    },
    {
        "id": "TC-02",
        "category": "Domain Terminology",
        "context": "The Aegis security architecture enforces zero-trust token isolation using asymmetric ECDSA verification.",
        "question": "Which cryptographic algorithm does Aegis utilize for zero-trust isolation?",
        "ground_truth": "Asymmetric ECDSA verification",
        "eval_keywords": ["ecdsa", "asymmetric"]
    },
    {
        "id": "TC-03",
        "category": "Negative Constraint (No Hallucination)",
        "context": "The company was founded in 2021 by Sarah Connor and operates offices in Seattle and Austin.",
        "question": "What is the company's annual revenue in Tokyo?",
        "ground_truth": "The provided text contains no information regarding Tokyo or annual revenue.",
        "eval_keywords": ["not mentioned", "no information", "not provided", "does not contain", "unspecified"]
    }
]


def score_response(output: str, keywords: list) -> float:
    """Calculates semantic keyword hit percentage against ground-truth criteria."""
    out_lower = output.lower()
    hits = sum(1 for kw in keywords if kw.lower() in out_lower)
    return round((hits / len(keywords)) * 100, 1)


# ==========================================
# STREAMLIT UI INTERFACE
# ==========================================

st.title("🛡️ Aegis Automated Evaluation Harness")
st.markdown("Automated regression testing, ground-truth evaluation, and factual adherence verification suite.")

# Sidebar - Document Ingestion & Controls
with st.sidebar:
    st.header("⚙️ Harness Controls")
    st.info(f"Loaded Benchmark Test Cases: **{len(BENCHMARK_SUITE)}**")
    
    st.subheader("📁 Upload Knowledge Documents")
    uploaded_file = st.file_uploader("Upload a PDF document to evaluate:", type=["pdf"])
    
    pdf_text = ""
    if uploaded_file:
        try:
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pdf_text += extracted + "\n"
            st.success(f"Loaded: {len(reader.pages)} PDF Pages ({len(pdf_text)} characters)")
        except Exception as e:
            st.error(f"PDF Parse Error: {e}")

    if st.button("🧹 Session Reset & Data Purge"):
        st.session_state.clear()
        st.rerun()

# Main Interface Tabs
tab1, tab2 = st.tabs(["🧪 Custom QA & Document Evaluation", "📊 Run Core Benchmark Suite"])

with tab1:
    st.subheader("Custom Evaluation Prompting")
    
    context_mode = st.radio(
        "Select context source for model evaluation:",
        ["Inherit from Uploaded PDF Document", "Use Blank Custom Context", "Inherit from Benchmark Reference Case"]
    )

    eval_context = ""
    if context_mode == "Inherit from Uploaded PDF Document":
        if pdf_text:
            eval_context = pdf_text
            st.success(f"Context locked to uploaded PDF ({len(pdf_text)} chars).")
        else:
            st.warning("Upload a PDF in the sidebar first to use this mode.")
    elif context_mode == "Inherit from Benchmark Reference Case":
        eval_context = BENCHMARK_SUITE[0]["context"]
        st.text_area("Reference Context:", eval_context, height=100, disabled=True)
    else:
        eval_context = st.text_area("Enter Custom Reference Context:", height=100)

    user_query = st.text_input("Enter question to evaluate model response:")

    if user_query:
        with st.spinner("Executing inference and adherence evaluation..."):
            t0 = time.time()
            system_prompt = (
                "You are an accurate, deterministic knowledge evaluation assistant. "
                "Answer the user's question strictly and factually using ONLY the provided context. "
                "If the information is not present in the context, clearly state that it is not provided."
            )
            full_prompt = f"Context:\n{eval_context}\n\nQuestion: {user_query}" if eval_context else user_query
            
            output = run_llm_inference(system_prompt, full_prompt)
            latency = round(time.time() - t0, 3)

        st.markdown("### ⏱️ Response History")
        st.markdown(f"**Question:** {user_query}")
        st.info(f"**AI Output:** {output}")
        st.caption(f"⚡ Latency: {latency}s")

        with st.expander("🔍 View Raw Context Ingested"):
            st.text(eval_context if eval_context else "No external context provided.")

with tab2:
    st.subheader("Automated Benchmark Suite Execution")
    st.markdown("Run the deterministic test suite to calculate semantic keyword adherence and latency.")

    if st.button("🚀 Run Core Benchmark Validation Suite"):
        results = []
        progress_bar = st.progress(0)

        for i, test in enumerate(BENCHMARK_SUITE):
            t0 = time.time()
            system_prompt = (
                "You are a deterministic QA testing engine. Answer the question factually "
                "based exclusively on the given context. If unmentioned, state no information is available."
            )
            full_prompt = f"Context:\n{test['context']}\n\nQuestion: {test['question']}"
            
            model_res = run_llm_inference(system_prompt, full_prompt)
            latency = round(time.time() - t0, 3)
            score = score_response(model_res, test["eval_keywords"])

            results.append({
                "Test ID": test["id"],
                "Category": test["category"],
                "Score": f"{score}%",
                "Latency": f"{latency}s",
                "Question": test["question"],
                "Model Output": model_res,
                "Ground Truth Target": test["ground_truth"]
            })
            progress_bar.progress((i + 1) / len(BENCHMARK_SUITE))

        st.success("Benchmark Run Completed!")
        st.table([{
            "Test ID": r["Test ID"],
            "Category": r["Category"],
            "Score": r["Score"],
            "Latency": r["Latency"]
        } for r in results])

        for r in results:
            with st.expander(f"Details: {r['Test ID']} - {r['Category']} (Score: {r['Score']})"):
                st.markdown(f"**Question:** {r['Question']}")
                st.markdown(f"**Target Ground Truth:** {r['Ground Truth Target']}")
                st.markdown(f"**Actual Model Response:** {r['Model Output']}")