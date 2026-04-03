from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = PROJECT_ROOT / "chunks"
VECTOR_DB_PATH = PROJECT_ROOT / "vectordb"

SOURCE_PDF_PATH = DATA_DIR / "AI Training Document.pdf"
CHUNKS_PATH = CHUNKS_DIR / "chunks.json"

COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "mistral"
DEFAULT_TOP_K = 5
EMBEDDING_DIMENSION = 384
GENERATION_TEMPERATURE = 0.2
