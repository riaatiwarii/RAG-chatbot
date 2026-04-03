from src.config import COLLECTION_NAME, DEFAULT_TOP_K


def retrieve(query, model, client, top_k=DEFAULT_TOP_K):
    if not query or not query.strip():
        return []

    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return [hit.payload.get("text", "") for hit in results if hit.payload]
