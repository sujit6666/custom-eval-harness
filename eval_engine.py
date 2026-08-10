import ollama
import numpy as np
from test_cases import EVAL_DATASET

def run_automated_evaluation():
    print("🚀 Initializing Automated LLM Evaluation Run...")
    print("--------------------------------------------------")
    
    total_score = 0
    results_summary = []

    # 1. Loop through each test case in our dataset dictionary array
    for case in EVAL_DATASET:
        print(f"\nEvaluating: [{case['id']}] - Category: {case['category']}")
        print(f"❓ Question: {case['question']}")
        
        # 2. Build a strict system instruction prompt to minimize text variations
        system_prompt = (
            "You are an automated support bot. Answer the user's question using ONLY the provided Context text. "
            "Be extremely brief, factual, and direct. Do not write a greeting or an explanation."
        )
        user_prompt = f"Context: {case['context']}\nQuestion: {case['question']}"

        # 3. Call your free offline local AI model brain execution block
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.0} # Lower temperature enforces strict deterministic answers
        )
        
        ai_output = response['message']['content'].lower()
        print(f"🤖 AI Response: \"{ai_output.strip()}\"")

        # 4. Keyword Exact Match Precision Grade Verification Step
        # Check how many golden target keywords exist inside the AI output string text block
        matched_keywords = []
        for word in case['expected_keywords']:
            if word.lower() in ai_output:
                matched_keywords.append(word)

        # Calculate accurate matching percentage accuracy metrics for this run loop
        total_keywords_count = len(case['expected_keywords'])
        matched_count = len(matched_keywords)
        case_accuracy_score = (matched_count / total_keywords_count) * 100
        total_score += case_accuracy_score

        print(f"🎯 Grade: {matched_count}/{total_keywords_count} Keywords Found | Match Score: {case_accuracy_score:.1f}%")
        print(f"📍 Matched Words: {matched_keywords}")
        
        results_summary.append({
            "id": case["id"],
            "score": case_accuracy_score
        })

    # 5. Output Summary Metrics Dashboard Data Compilation Blocks
    print("\n==================================================")
    print("📊 FINAL HARNESS EVALUATION METRICS REPORT")
    print("==================================================")
    
    # Calculate global system accuracy performance averages using numpy arrays
    scores_array = np.array([res["score"] for res in results_summary])
    mean_system_accuracy = np.mean(scores_array)
    
    for summary in results_summary:
        print(f"📌 Case ID {summary['id']}: Accuracy Score -> {summary['score']:.1f}%")
        
    print("--------------------------------------------------")
    print(f"📈 GLOBAL LLM SYSTEM PERFORMANCE ACCURACY: {mean_system_accuracy:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    run_automated_evaluation()
