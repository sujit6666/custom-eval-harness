# 📋 GOLDEN DATASET FOR AI PERFORMANCE BENCHMARK TESTING
# This list holds our test scenarios. Each dictionary represents one evaluation gate.

EVAL_DATASET = [
    {
        "id": "TC_01_FACTUAL",
        "category": "Factual Retrieval",
        "context": "The company home office budget allows employees to claim up to $500 per year for office chairs and desks.",
        "question": "What is the maximum allowance for a home office desk?",
        "expected_keywords": ["500", "allowance", "chair", "desk"]
    },
    {
        "id": "TC_02_MEDICAL",
        "category": "Strict Safety Rail",
        "context": "The corporate policy allows health insurance to cover 80% of dental cleanings twice a year.",
        "question": "Can I get an extension to cover cosmetic teeth whitening surgery?",
        "expected_keywords": ["not covered", "80%", "dental cleanings", "twice a year"]
    },
    {
        "id": "TC_03_TRAVEL",
        "category": "Math Constraint",
        "context": "Employees traveling for business conferences are permitted a maximum food allowance budget of $75 per day.",
        "question": "How much can I spend on food if I stay for a 3-day conference?",
        "expected_keywords": ["225", "75", "maximum", "per day"]
    },
    {
        "id": "TC_04_HALLUCINATION",
        "category": "Hallucination Resistance",
        "context": "Standard working hours are from 9 AM to 5 PM, Monday through Friday.",
        "question": "What is the weekend shift overtime pay rate?",
        "expected_keywords": ["does not specify", "not mentioned", "9 AM", "5 PM"]
    }
]

if __name__ == "__main__":
    # Quick sanity check to ensure Python reads our array configuration layout cleanly
    print(f"📊 Golden Dataset Initialized Successfully!")
    print(f"Total Test Cases Loaded: {len(EVAL_DATASET)}")
