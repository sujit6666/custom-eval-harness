import streamlit as st
import ollama
import numpy as np
import time
from test_cases import EVAL_DATASET

# 1. Page Configuration and Layout
st.set_page_config(page_title="Aegis Eval Harness", page_icon="🛡️", layout="wide")

# 2. Inject Fully Adaptive Theme Styles for Text & Form Elements
st.markdown("""
    <style>
    /* Elegant Title Styling */
    .glow-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.05em;
        margin-bottom: 5px;
    }
    
    /* Support text adjustments matching layout themes */
    [data-theme="light"] .stMarkdown p { color: #0f172a !important; }
    [data-theme="dark"] .stMarkdown p { color: #f1f5f9 !important; }
    
    /* Crisp container box styles for individual case tracking cards */
    .eval-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="glow-title">🛡️ Aegis Automated Eval Harness</h1>', unsafe_allow_html=True)
st.write("Programmatically bench-test, validate, and stress-test LLM applications using exact-match key precision metrics.")

# 3. Sidebar Statistics Setup
st.sidebar.markdown("### ⚙️ Harness Controls")
st.sidebar.write("Total Benchmark Test Cases Loaded:", len(EVAL_DATASET))

# 4. Handle Execution State Click Triggers
if st.sidebar.button("⚡ Run Core Benchmark Validation Suite", use_container_width=True):
    st.subheader("🏃‍♂️ Live Benchmark Execution Stream")
    
    progress_bar = st.progress(0)
    results_summary = []
    
    # Process through our test cases sequentially
    for idx, case in enumerate(EVAL_DATASET):
        with st.status(f"Processing: `{case['id']}` - {case['category']}...", expanded=True) as status:
            
            system_prompt = (
                "You are an automated support bot. Answer the user's question using ONLY the provided Context text. "
                "Be extremely brief, factual, and direct. Do not write an explanation or conversational filler."
            )
            user_prompt = f"Context: {case['context']}\nQuestion: {case['question']}"

            # Query our local offline machine model engine
            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {"role": "system", "system_prompt": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.0}
            )
            
            ai_output = response['message']['content'].lower()
            
            # Extract keywords matched
            matched_keywords = [word for word in case['expected_keywords'] if word.lower() in ai_output]
            
            total_words = len(case['expected_keywords'])
            matched_count = len(matched_keywords)
            accuracy_score = (matched_count / total_words) * 100
            
            results_summary.append({"id": case["id"], "score": accuracy_score, "category": case["category"]})
            
            # Print out live visual cards onto our display surface
            st.markdown(f"**❓ Question:** {case['question']}")
            st.markdown(f"**🤖 AI Model Output:** *\"{ai_output.strip()}\"*")
            st.markdown(f"🎯 **Score:** `{matched_count}/{total_words}` Target Chunks Match ➔ **{accuracy_score:.1f}%**")
            st.caption(f"📍 Tracked Matches: {matched_keywords}")
            
            status.update(label=f"Completed: {case['id']} | Score: {accuracy_score:.1f}%", state="complete")
            
        # Update our top progress visual bar fluidly
        progress_bar.progress((idx + 1) / len(EVAL_DATASET))
        time.sleep(0.5)

    # 5. Core Performance Report Calculations Dashboard Render blocks
    st.markdown("---")
    st.markdown("## 📊 Performance Statistics Metrics Dashboard")
    
    scores_list = [res["score"] for res in results_summary]
    np_scores = np.array(scores_list)
    global_system_accuracy = np.mean(np_scores)
    
    # Beautiful large layout presentation metrics widgets
    col1, col2, col3 = st.columns(3)
    col1.metric("Global Core System Performance Accuracy", f"{global_system_accuracy:.2f}%")
    col2.metric("Highest Individual Match Score", f"{np.max(np_scores):.1f}%")
    col3.metric("Lowest Individual Match Score", f"{np.min(np_scores):.1f}%")
    
    # Display historical verification grid rows
    st.markdown("### 🗂️ Test Case Breakdown Report Grid")
    for res in results_summary:
        if res["score"] >= 70:
            st.success(f"✔️ **{res['id']}** ({res['category']}) ➔ **Score: {res['score']:.1f}%**")
        elif res["score"] >= 40:
            st.warning(f"⚠️ **{res['id']}** ({res['category']}) ➔ **Score: {res['score']:.1f}%**")
        else:
            st.error(f"❌ **{res['id']}** ({res['category']}) ➔ **Score: {res['score']:.1f}%**")
else:
    st.info("💡 Click the button inside the sidebar management menu control dashboard panel on the left to fire up your benchmark testing sequence.")
