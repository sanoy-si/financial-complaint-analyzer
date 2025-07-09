import streamlit as st
import time

# --- IMPORTANT: The Control Switch ---
USE_REAL_PIPELINE = False

# --- Conditional Import of the Pipeline ---
if USE_REAL_PIPELINE:
    # This will try to import the real pipeline. It will be slow to load.
    from src.complaint_analyst.rag_pipeline import RAGPipeline as ComplaintPipeline
else:
    # This imports our fast, simulated pipeline.
    from src.complaint_analyst.mock_rag_pipeline import MockRAGPipeline as ComplaintPipeline


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Financial Complaint Analyzer",
    page_icon="💬",
    layout="centered"
)


# --- Caching the Pipeline ---
@st.cache_resource
def load_pipeline():
    """Loads and caches the complaint analysis pipeline."""
    if USE_REAL_PIPELINE:
        st.write("Loading REAL RAG Pipeline... (This may take several minutes)")
    else:
        st.write("Loading MOCK RAG Pipeline for demonstration.")
    
    pipeline = ComplaintPipeline()
    return pipeline

pipeline = load_pipeline()


# --- Application UI ---
st.title("Financial Complaint Analyzer 💬")
st.markdown(
    "Ask questions about customer complaints and get synthesized, evidence-backed answers."
)

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I'm here to help you analyze customer complaints. What would you like to know?"
    }]

# --- "Clear Chat" Button ---
def clear_chat_history():
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I'm here to help you analyze customer complaints. What would you like to know?"
    }]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("e.g., Why are people unhappy with their credit cards?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Analyzing complaints and generating an answer..."):
            result = pipeline.answer_question(prompt)
            
            answer = result['answer']
            sources = result['sources']
            
            # --- Display the Answer and Sources ---
            full_response_content = answer

            if sources:
                full_response_content += "\n\n<br>**Sources Used:**\n"
                for i, doc in enumerate(sources):
                    source_text = doc.page_content.replace('\n', ' ')
                    metadata = doc.metadata
                    with st.expander(f"Source {i+1}: Complaint ID {metadata.get('complaint_id', 'N/A')}"):
                        st.write(f"**Product:** {metadata.get('product', 'N/A')}")
                        st.write(f"**Issue:** {metadata.get('issue', 'N/A')}")
                        st.write(f"**Excerpt:** *{source_text}*")
            
            time.sleep(1)
            message_placeholder.markdown(full_response_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response_content})