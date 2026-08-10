# 🛡️ Aegis Automated Evaluation Harness

An enterprise-grade LLM engineering validation suite built to programmatically bench-test, validate, and track regressions in localized language models completely offline.

## 🚀 Key Architectural Features
- **Deterministic Evaluation Pipelines:** Evaluates model outputs against strict data assertions using zero-temperature parameters.
- **Exact-Match Keyword Precision Metrics:** Uses custom indexing algorithms to compute percentage accuracy targets programmatically.
- **Interactive Metrics Dashboard UI:** Built as a theme-adaptive Streamlit web environment with integrated telemetry stat counters.

## 🛠️ Project File Components
- `app.py`: High-contrast, theme-adaptive interface dashboard setup file.
- `eval_engine.py`: Core automated validation testing loop execution file.
- `test_cases.py`: The master dataset file holding our verification targets.
- `.gitignore`: Filtering system file blocking internal file configurations.

## ⚙️ Quick Installation Guide
To fire up this benchmarking system locally, activate your command terminal and run:

```cmd
# Activate your sandbox environment
venv\Scripts\activate

# Install the necessary library files
pip install numpy ollama streamlit

# Relaunch the execution server
streamlit run app.py
```
