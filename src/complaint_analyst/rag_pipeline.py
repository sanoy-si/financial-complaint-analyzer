import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any

# --- Configuration ---
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.1'

class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print(f"Loading vector store from {VECTOR_STORE_PATH}...")
        self.vector_store = FAISS.load_local(VECTOR_STORE_PATH, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})  
        print("Vector store loaded successfully.")

        print(f"Loading LLM: {LLM_MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,  
            temperature=0.7,  
            top_p=0.95,  
            do_sample=True,
        )
        print("RAG Pipeline initialized successfully.")

    def _format_context(self, docs: List[Any]) -> str:
        return "\n\n".join([f"Complaint Chunk {i+1}:\n" + doc.page_content for i, doc in enumerate(docs)])

    def answer_question(self, question: str) -> Dict[str, Any]:
        retrieved_docs = self.retriever.get_relevant_documents(question)
        context_str = self._format_context(retrieved_docs)
        prompt_template = f"""
        [INST]
        You are an expert financial analyst assistant. Your task is to answer questions about customer complaints.
        Use ONLY the following retrieved complaint excerpts to formulate your answer.
        Do not use any prior knowledge. If the context doesn't contain the answer, state that you don't have enough information.
        Your answer should be concise and directly address the user's question.

        CONTEXT:
        {context_str}
        
        QUESTION:
        {question}
        [/INST]
        Answer:
        """
        print("Generating answer...")
        generated_output = self.llm_pipeline(prompt_template)
        answer = generated_output[0]['generated_text'].split('[/INST]')[1].strip()
        
        return {
            "answer": answer,
            "sources": retrieved_docs
        }

if __name__ == '__main__':
    # This is a simple test to see if the pipeline initializes correctly.
    try:
        rag_pipeline = RAGPipeline()
        print("\n--- RAG Pipeline Test ---")
        test_question = "Why are people having issues with their credit card rewards?"
        result = rag_pipeline.answer_question(test_question)
        print(f"\nQuestion: {test_question}")
        print(f"\nAnswer: {result['answer']}")
        print("\n--- Sources Used ---")
        for i, doc in enumerate(result['sources']):
            print(f"Source {i+1} (Complaint ID: {doc.metadata['complaint_id']}):")
            print(doc.page_content)
            print("-" * 20)

    except Exception as e:
        print(f"An error occurred: {e}")