# 🛡️ Aegis Automated Evaluation Harness

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aegis-eval-harness.streamlit.app/)

An enterprise-grade LLM engineering validation suite built to programmatically bench-test, validate, and track regressions in localized language models.

🔗 **Live App:** https://aegis-eval-harness.streamlit.app/

## 🚀 Key Architectural Features
- **Deterministic Evaluation Pipelines:** Evaluates model outputs against strict data assertions using zero-temperature parameters.
- **Exact-Match Keyword Precision Metrics:** Uses custom indexing algorithms to compute percentage accuracy targets programmatically.
- **Interactive Metrics Dashboard UI:** Built as a theme-adaptive Streamlit web environment with integrated telemetry stat counters.

## 🛠️ Project File Components
- `app.py`: High-contrast, theme-adaptive interface dashboard setup file.
- `eval_engine.py`: Core automated validation testing loop execution file.
- `test_cases.py`: The master dataset file holding our verification targets.
- `requirements.txt`: Project dependencies for cloud deployment.
- `.gitignore`: Filtering system file blocking internal file configurations.

## ⚙️ Quick Installation Guide
To fire up this benchmarking system locally, activate your command terminal and run:

```cmd
# Activate your sandbox environment
venv\Scripts\activate

# Install the necessary library files
pip install -r requirements.txt

# Relaunch the execution server
streamlit run app.py