import ollama

from src.config import GENERATION_MODEL_NAME, GENERATION_TEMPERATURE


def build_prompt(query, chunks):
    context = "\n\n".join(chunks)

    prompt = f"""
You are a legal assistant.

Answer the question using ONLY the provided context.

STRICT RULES:
- Do NOT use phrases like "according to the context".
- Do NOT give short or vague answers.
- You MUST provide a structured answer.

FORMAT (MANDATORY):
1. First line: Direct answer (1 sentence)
2. Then: bullet points with supporting details (2-4 points)

If answer is not present, say: "Not found in document."

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt.strip()


def stream_answer(query, chunks):
    if not chunks:
        yield "Not found in document."
        return

    response_stream = ollama.chat(
        model=GENERATION_MODEL_NAME,
        messages=[{"role": "user", "content": build_prompt(query, chunks)}],
        stream=True,
        options={"temperature": GENERATION_TEMPERATURE},
    )

    for chunk in response_stream:
        token = chunk["message"]["content"]
        if token:
            yield token


def generate_answer(query, chunks):
    return "".join(stream_answer(query, chunks)).strip()
