import pandas as pd
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
import os

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed/filtered_complaints.csv'
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# --- Chunking Parameters ---
CHUNK_SIZE = 1000   
CHUNK_OVERLAP = 200

def create_vector_store():
    print(f"Loading processed data from {PROCESSED_DATA_PATH}...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}.")
        print("Please run the data processing notebook (Task 1) first.")
        return
        
    df.dropna(subset=['narrative_cleaned'], inplace=True)
    print(f"Loaded {len(df)} complaints.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    print("Creating document chunks with metadata...")
    documents = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Chunking Narratives"):
        text_chunks = text_splitter.split_text(row['narrative_cleaned'])
        for chunk in text_chunks:
            metadata = {
                "complaint_id": row['complaint_id'],
                "product": row['product'],
                "issue": row['issue'],
                "company": row['company'],
                "state": row['state'],
                "source_text": row['narrative'] 
            }
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)
            
    print(f"Created {len(documents)} text chunks.")

    if not documents:
        print("No documents were created. Please check the input data.")
        return

    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Creating and saving the FAISS vector store...")
    print("This may take some time depending on the number of documents...")
    
    vector_store = FAISS.from_documents(documents, embeddings)
    
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    
    vector_store.save_local(VECTOR_STORE_PATH)
    
    print("-" * 50)
    print("Vector store created successfully!")
    print(f"Saved to: {VECTOR_STORE_PATH}")
    print(f"Number of indexed vectors: {vector_store.index.ntotal}")
    print("-" * 50)

if __name__ == '__main__':
    create_vector_store()