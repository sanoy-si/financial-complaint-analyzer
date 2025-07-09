import sys
import os
import pandas as pd

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.complaint_analyst.rag_pipeline import RAGPipeline

# --- Define Representative Questions for Evaluation ---
EVALUATION_QUESTIONS = [
    # --- General Overview Questions ---
    "What are the most common complaints about credit cards?",
    "Why are people unhappy with their personal loans?",

    # --- Specific "Why" Questions ---
    "Why are loan applications being denied after being pre-approved?",
    "What are the main problems with money transfers?",

    # --- Problem Solving Questions ---
    "How can we reduce complaints about incorrect information on credit reports?",
    "What issues are customers facing with closing their savings accounts?",

    # --- Comparative Questions ---
    "Are billing disputes more common for credit cards or personal loans?",
    
    # --- "Gotcha" Question (to test the "I don't know" response) ---
    "What are the best investment strategies for young adults?",
]

def run_evaluation():
    print("Starting RAG pipeline evaluation...")
    rag_pipeline = RAGPipeline()
    
    evaluation_results = []

    for question in EVALUATION_QUESTIONS:
        print(f"\nProcessing question: '{question}'")
        result = rag_pipeline.answer_question(question)
        evaluation_results.append({
            "Question": question,
            "Generated Answer": result['answer'],
            "Source 1": result['sources'][0].page_content if len(result['sources']) > 0 else "N/A",
            "Source 1 Metadata": result['sources'][0].metadata if len(result['sources']) > 0 else "N/A",
        })

    # --- Create the Markdown Table ---
    print("\n\n--- EVALUATION REPORT ---")
    header = "| Question | Generated Answer | Key Source Snippet | Quality Score (1-5) | Comments/Analysis |\n|---|---|---|---|---|"
    print(header)

    for res in evaluation_results:
        answer_for_table = res["Generated Answer"].replace('\n', ' ').strip()
        source_for_table = res["Source 1"].replace('\n', ' ').strip()[:150] + "..."
        
        row = f"| {res['Question']} | {answer_for_table} | `{source_for_table}` | | |"
        print(row)

if __name__ == '__main__':
    run_evaluation()