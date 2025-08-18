# src/complaint_analyst/mock_rag_pipeline.py

import time
from typing import List, Dict, Any

# This is a dummy Document class to mimic LangChain's Document object
class MockDocument:
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

class MockRAGPipeline:
    """
    A mock RAG pipeline that simulates the behavior of the real pipeline
    for rapid UI development and testing, without loading any models.
    """
    def __init__(self):
        print("✅ Initializing MOCK RAG Pipeline...")
        # No heavy models to load, initialization is instant.
        time.sleep(1) # Simulate a tiny bit of loading time
        print("✅ MOCK RAG Pipeline is ready!")

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Simulates answering a question by returning a canned response and fake sources.
        """
        print(f"🧠 MOCK: Received question: '{question}'")
        print("🧠 MOCK: Simulating retrieval and generation...")
        time.sleep(2) # Simulate the time it takes to generate an answer

        # Create a canned, generic answer
        mock_answer = f"This is a simulated answer for the question: '{question}'. " \
                      "The mock pipeline is working correctly. Based on the retrieved context, " \
                      "customers frequently mention issues with billing and unexpected fees."

        # Create fake source documents to test the UI display
        mock_sources = [
            MockDocument(
                page_content="I was charged an overdraft fee that I did not expect. My account balance was positive when I made the purchase, but the transaction posted later. This is unfair.",
                metadata={
                    "complaint_id": "MOCK-001", "product": "Savings Account",
                    "issue": "Unexpected fees", "company": "Mock Bank",
                    "state": "CA", "source_text": "Full original text for complaint MOCK-001..."
                }
            ),
            MockDocument(
                page_content="My credit card statement shows a 'service charge' of {$25.00} that I never agreed to. I called customer service, and they could not explain what this charge was for.",
                metadata={
                    "complaint_id": "MOCK-002", "product": "Credit Card",
                    "issue": "Billing disputes", "company": "Mock Credit Inc.",
                    "state": "NY", "source_text": "Full original text for complaint MOCK-002..."
                }
            )
        ]

        return {
            "answer": mock_answer,
            "sources": mock_sources
        }