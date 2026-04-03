from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    VECTOR_DB_PATH,
)
from src.generator import generate_answer, stream_answer
from src.retriever import retrieve


def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def load_client() -> QdrantClient:
    return QdrantClient(path=str(VECTOR_DB_PATH))


def get_chunk_count(client) -> int:
    return client.count(collection_name=COLLECTION_NAME).count


def rag_query(query, model, client, top_k=DEFAULT_TOP_K):
    chunks = retrieve(query=query, model=model, client=client, top_k=top_k)
    answer = generate_answer(query, chunks)
    return answer, chunks


def rag_stream(query, model, client, top_k=DEFAULT_TOP_K):
    chunks = retrieve(query=query, model=model, client=client, top_k=top_k)
    return stream_answer(query, chunks), chunks
