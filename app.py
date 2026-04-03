import streamlit as st

from src.config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME
from src.rag_pipeline import get_chunk_count, load_client, load_model, rag_stream


st.set_page_config(page_title="RAG Chatbot", layout="wide")


@st.cache_resource
def init_model():
    return load_model()


@st.cache_resource
def init_client():
    return load_client()


model = init_model()
client = init_client()

st.title("AI RAG Chatbot")

sample_questions = [
    "What happens if I violate eBay policies?",
    "Can eBay suspend or terminate my account?",
    "Can I share my login credentials with third parties?",
]

st.sidebar.header("System Info")
st.sidebar.write(f"Model: {GENERATION_MODEL_NAME} (Ollama)")
st.sidebar.write(f"Embedding: {EMBEDDING_MODEL_NAME}")
st.sidebar.write(f"Vector DB: Qdrant local / `{COLLECTION_NAME}`")
st.sidebar.write(f"Indexed chunks: {get_chunk_count(client)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander("Source Chunks"):
                for i, chunk in enumerate(msg["chunks"]):
                    st.markdown(f"**Chunk {i + 1}:**")
                    st.write(chunk[:500])

st.subheader("Suggested Questions")
cols = st.columns(3)

selected_question = None
for i, question in enumerate(sample_questions):
    if cols[i].button(question):
        selected_question = question

typed_question = st.chat_input("Ask a question...")
query = selected_question or typed_question

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        token_stream, chunks = rag_stream(query, model, client)

        full_response = ""
        for token in token_stream:
            full_response += token
            response_placeholder.markdown(full_response)

        full_response = full_response.strip()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "chunks": chunks,
        }
    )

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
