# CrediTrust Financial - Complaint Analyst AI

This project is an internal AI tool designed to transform raw, unstructured customer complaint data into a strategic asset for CrediTrust Financial. It provides a simple chat interface for internal stakeholders (Product, Support, Compliance) to ask plain-English questions about customer complaints and receive synthesized, evidence-backed answers in seconds.

## Problem

Internal teams at CrediTrust face significant bottlenecks in understanding customer feedback:
- **Product Managers** struggle to identify emerging issues.
- **Customer Support** is overwhelmed by complaint volume.
- **Compliance & Risk** teams are reactive to potential issues.

This tool aims to solve these problems by leveraging Retrieval-Augmented Generation (RAG).

## Solution Architecture (RAG)

The system works in three steps:
1.  **Retrieve:** When a user asks a question, the system searches a specialized vector database to find the most relevant customer complaint narratives.
2.  **Augment:** The user's question and the retrieved complaints are combined into a detailed prompt.
3.  **Generate:** A Large Language Model (LLM) receives the prompt and generates a concise, human-readable answer based *only* on the provided complaints.


## Project Setup

### Prerequisites
- Python 3.10+
- Poetry
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sanoy-si/financial-complaint-analyzer.git
    cd credetrust-complaint-analyzer
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```
   

3.  **Download the data:**
    Download the dataset from the [CFPB Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?size=0&data_received_max=2023-11-20&data_received_min=2023-08-21&field=all&format=csv) and save it as `complaints.csv` inside the `data/raw/` directory.

---

## How to Run the Project

The project is divided into several steps. Run them in order.

### 1. Data Preprocessing
Run the EDA notebook to understand and clean the data. This will generate the `filtered_complaints.csv` file needed for the next step.
```bash
poetry run jupyter notebook notebooks/01_eda_and_preprocessing.ipynb
```

### 2. Create the Vector Store 
Run the indexing script. This will read the processed data, create embeddings, and save the vector store.
```bash
poetry run python -m src.complaint_analyst.vector_store
```
*(Note: You can run a script inside a package using the `-m` flag)*

### 3. Run the Chat Application 
Launch the Streamlit web application.
```bash
poetry run streamlit run app.py
```
Open your browser to the local URL provided by Streamlit (usually `http://localhost:8501`).

---
## Project Structure
```
credetrust-complaint-analyzer/
├── .github/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── complaint_analyst/
├── vector_store/
├── app.py
├── README.md
└── pyproject.toml
```