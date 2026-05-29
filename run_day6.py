import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from evaluation.guardrails import ClinicalRagasEvaluator

def execute_day_6_pipeline():
    print("===================================================")
    print("  STARTING DAY 6: RAGAS TELEMETRY & HALLUCINATION CHECK ")
    print("===================================================")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[CRITICAL ERROR] Please export GROQ_API_KEY='your_key' in your terminal.")
        return

    # Initialize the strict medical evaluator
    evaluator = ClinicalRagasEvaluator(api_key=api_key)
    
    print("\n[SCENARIO 1: PERFECT CLINICAL RAG]")
    query_1 = "What are the primary symptoms of acute appendicitis?"
    context_1 = ["Acute appendicitis typically presents with periumbilical pain migrating to the right lower quadrant, fever, and nausea."]
    answer_1 = "The primary symptoms of acute appendicitis include pain that starts near the belly button and moves to the lower right abdomen, accompanied by fever and nausea."
    
    print("Grading Scenario 1 (Expected: High Scores)...")
    scores_1 = evaluator.grade_transaction(query_1, answer_1, context_1)
    
    print(f"  -> Faithfulness:      {scores_1.get('faithfulness', 0):.2f}/1.00 (No hallucinations)")
    print(f"  -> Answer Relevancy:  {scores_1.get('answer_relevancy', 0):.2f}/1.00 (Stayed on topic)")
    print(f"  -> Context Precision: {scores_1.get('context_precision', 0):.2f}/1.00 (Good data pulled)")


    print("\n[SCENARIO 2: THE HALLUCINATION]")
    query_2 = "Can I treat a severe asthma attack with essential oils?"
    context_2 = ["Asthma exacerbations require immediate treatment with short-acting beta-agonists (SABA) like albuterol and systemic corticosteroids. Essential oils have no proven efficacy and may trigger bronchospasm."]
    # The LLM hallucinates and gives dangerous advice not found in the context
    answer_2 = "Yes, you can treat a severe asthma attack with lavender and eucalyptus essential oils. They open up the lungs quickly. You do not need an inhaler."
    
    print("Grading Scenario 2 (Expected: Low Faithfulness)...")
    scores_2 = evaluator.grade_transaction(query_2, answer_2, context_2)
    
    print(f"  -> Faithfulness:      {scores_2.get('faithfulness', 0):.2f}/1.00 (CAUGHT THE HALLUCINATION)")
    print(f"  -> Answer Relevancy:  {scores_2.get('answer_relevancy', 0):.2f}/1.00 (Answered the prompt, but dangerously)")
    print("===================================================")

if __name__ == "__main__":
    execute_day_6_pipeline()
