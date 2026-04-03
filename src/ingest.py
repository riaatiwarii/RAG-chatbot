import json
import os
import re

import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader

from src.config import CHUNKS_PATH, SOURCE_PDF_PATH


def ensure_tokenizer() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text.append(extracted)

    return "\n".join(text)


def clean_text(text: str) -> str:
    text = re.sub(r"\n(?=[a-z])", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"Page \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, min_words=80, max_words=150):
    ensure_tokenizer()
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())

        if word_count + sentence_len > max_words:
            if word_count >= min_words:
                chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0

        current_chunk.append(sentence)
        word_count += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def save_chunks(chunks, path=CHUNKS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(chunks)} chunks -> {path}")


if __name__ == "__main__":
    print("Loading PDF...")
    raw_text = load_pdf(str(SOURCE_PDF_PATH))

    print("Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("\nSample text preview:\n")
    print(cleaned_text[:500])
    print("\n---\n")

    print("Chunking text...")
    chunks = chunk_text(cleaned_text, min_words=80, max_words=150)

    print(f"Total chunks created: {len(chunks)}")

    print("Saving chunks...")
    save_chunks(chunks)
