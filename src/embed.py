import json

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    VECTOR_DB_PATH,
)


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def init_qdrant():
    client = QdrantClient(path=str(VECTOR_DB_PATH))
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE,
        ),
    )
    return client


def embed_and_store(chunks, model, client):
    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    points = [
        PointStruct(id=i, vector=vector.tolist(), payload={"text": chunk["text"]})
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings))
    ]

    print("Uploading to Qdrant...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Stored {len(points)} vectors in Qdrant.")


if __name__ == "__main__":
    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)

    print("Loading model...")
    model = load_model()

    print("Initializing Qdrant...")
    client = init_qdrant()

    embed_and_store(chunks, model, client)
    client.close()
