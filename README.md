# Document-Grounded RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions based on PDF documents using semantic search and streaming responses. This project demonstrates core RAG concepts including document chunking, semantic embeddings, vector database indexing, and prompt-based generation with an open-source LLM.

Demo link - https://drive.google.com/file/d/1t6cqD2asWjSVTVDVc41GP6WFSfw0l3Ef/view?usp=sharing

**Disclaimer:** This is an educational implementation created for a Junior AI Engineer assignment to demonstrate RAG architecture principles. It is not a production-grade system.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Technical Stack](#technical-stack)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Key Components](#key-components)
- [Model & Configuration Choices](#model--configuration-choices)
- [Limitations & Known Issues](#limitations--known-issues)
- [Sample Queries](#sample-queries)

---

## Features

- **Semantic Search**: Uses pre-trained sentence embeddings to find relevant document chunks
- **Sentence-Aware Chunking**: Splits documents intelligently using NLTK sentence tokenization (80–150 words per chunk)
- **Streaming Responses**: Generates and displays LLM responses token-by-token in real-time
- **Local Vector Database**: Qdrant vector search engine for efficient retrieval
- **Document Transparency**: Shows the source chunks used to generate each answer
- **Chat History**: Maintains conversation context in the Streamlit UI
- **Suggested Questions**: Quick-access example queries to explore the chatbot

---

## Architecture Overview

The RAG pipeline follows these steps:

```
User Question
     ↓
[Embedding] → Query vector
     ↓
[Qdrant Retrieval] → Top-k relevant chunks
     ↓
[Prompt Injection] → Context + Query + Instructions
     ↓
[LLM Generation (Mistral)] → Streaming response
     ↓
[Streamlit UI] → Display answer + source chunks
```

### Data Pipeline

1. **Ingestion** (`ingest.py`): Extract and clean text from PDF using pypdf
2. **Chunking**: Split document into sentence-aware chunks (80–150 words)
3. **Embedding** (`embed.py`): Convert chunks to 384-dimensional vectors using SentenceTransformer
4. **Indexing**: Store vectors in local Qdrant collection with cosine distance metric
5. **Retrieval** (`retriever.py`): Embed user query and search for top-k matching chunks
6. **Generation** (`generator.py`): Construct prompt with retrieved context and generate response via Mistral + Ollama
7. **UI** (`app.py`): Display results and chat history in Streamlit interface

---

## Project Structure

```
ai_task/
├── app.py                      # Streamlit UI application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── AI Training Document.pdf  # Source document for Q&A
├── chunks/
│   └── chunks.json            # Pre-processed document chunks
├── vectordb/                  # Local Qdrant database directory
├── notebooks/
│   └── rag_workflow.ipynb     # Exploration and testing notebook
└── src/
    ├── config.py              # Configuration settings
    ├── ingest.py              # PDF text extraction & cleaning
    ├── embed.py               # Embedding & vector indexing
    ├── retriever.py           # Query embedding & chunk retrieval
    ├── generator.py           # LLM prompt construction & generation
    └── rag_pipeline.py        # RAG orchestration utilities
```

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Mistral 7B (via Ollama) | Response generation |
| **Embedding Model** | SentenceTransformer (all-MiniLM-L6-v2) | 384-dim semantic vectors |
| **Vector Database** | Qdrant (local mode) | Fast cosine similarity search |
| **PDF Processing** | pypdf | Text extraction from documents |
| **Tokenization** | NLTK | Sentence-aware document chunking |
| **UI Framework** | Streamlit | Interactive web interface |
| **Orchestration** | Custom Python modules | RAG pipeline coordination |

---

## Setup & Installation

### Prerequisites

- **Python 3.9+**
- **Ollama** with Mistral model installed ([download](https://ollama.ai))
  - After installing Ollama, run: `ollama pull mistral`
- **pip** (Python package manager)

### Installation Steps

1. **Clone/Extract the project**
   ```bash
   cd ai_task
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Ollama is running**
   ```bash
   ollama serve  # Run in a separate terminal
   ```

---

## Running the Application

The application runs in three stages:

### Stage 1: Ingest Documents
Extracts text from PDF and creates sentence-aware chunks:
```bash
python -m src.ingest
```
**Output:** `chunks/chunks.json` containing ~200–500 chunks depending on document size

### Stage 2: Create Vector Index
Embeds chunks and stores them in Qdrant:
```bash
python -m src.embed
```
**Output:** Local Qdrant database at `vectordb/` with indexed chunks
**Time:** ~2–5 minutes depending on document size and CPU

### Stage 3: Launch Chatbot UI
Starts the Streamlit interactive interface:
```bash
streamlit run app.py
```
**Access:** Open browser to `http://localhost:8501`

---

## Usage Guide

### Starting a Chat Session

1. Open the Streamlit app in your browser
2. View the sidebar showing:
   - **LLM Model**: mistral
   - **Embedding Model**: all-MiniLM-L6-v2
   - **Vector Database**: Qdrant (local)
   - **Indexed Chunks**: Total chunks in vector database

### Asking Questions

- **Type questions** in the chat input box at the bottom
- **Click suggested questions** to quickly test the chatbot
- **View streaming responses** as tokens are generated in real-time
- **Expand "📎 Source Chunks"** to see which document excerpts were used

### Example Workflows

**Single-turn query:**
```
User: "What happens if I violate the payment terms?"
→ Chatbot searches for relevant chunks
→ Generates answer grounded in document
→ Shows source chunks below answer
```

**Multi-turn conversation:**
```
User: "Can I share my account credentials?"
→ Chatbot answers first question
User: "What about with my family?"
→ Chatbot maintains context in chat history
```

### Clear Chat History

Click the **"Clear Chat"** button in the sidebar to reset conversation and start fresh.

---

## Key Components

### `ingest.py`
- Loads PDF using `pypdf.PdfReader`
- Removes headers, footers, and whitespace with regex
- Tokenizes into sentences using `nltk.sent_tokenize`
- Groups sentences into chunks of 80–150 words
- Saves to `chunks/chunks.json`

### `embed.py`
- Loads pre-trained `SentenceTransformer` (all-MiniLM-L6-v2)
- Reads chunks from JSON
- Generates 384-dimensional embeddings
- Initializes Qdrant collection with cosine distance
- Uploads all chunks and embeddings to `vectordb/`

### `retriever.py`
- Converts user query to embedding using same model as indexing
- Searches Qdrant with `search()` method (default: top-5 chunks)
- Returns ranked list of relevant text chunks and similarity scores

### `generator.py`
- Constructs instruction-based prompt:
  ```
  Context:
  [Retrieved chunks here]
  
  Question: [User query]
  
  Instructions:
  - Answer only using the provided context
  - If answer not found, say "Not found in document."
  - Be concise and accurate
  ```
- Calls Ollama chat endpoint with Mistral model
- Supports token-by-token streaming for live display

### `rag_pipeline.py`
- Loads embedding model and Qdrant client on initialization
- Provides `rag_query()` for single-turn generation
- Provides `rag_stream()` for token streaming
- Exposes `get_chunk_count()` for UI statistics

### `app.py`
- Streamlit interface with caching for models and clients
- Sidebar with configuration display and clear chat button
- Chat input area with history maintained in `st.session_state`
- Streaming response display with live token updates
- Source chunk expander for transparency
- Suggested questions for quick exploration

---

## Model & Configuration Choices

### Why Mistral 7B?

- **Efficient**: Runs on CPU without excessive latency
- **Multilingual**: Handles multiple languages well
- **Open Source**: No API fees or external dependencies beyond Ollama
- **Instruction-Following**: Naturally follows prompt instructions to answer from context

### Why SentenceTransformer (all-MiniLM-L6-v2)?

- **Lightweight**: Only 22MB, fast inference on CPU (~10–50ms per sentence)
- **High Quality**: MTEB benchmark scores competitive with larger models
- **Pre-trained**: Semantic understanding of English text without fine-tuning
- **384 Dimensions**: Good balance between expressiveness and storage/speed

### Why Qdrant?

- **Local Mode**: No network dependencies, full privacy
- **Fast**: Optimized vector search with SIMD operations
- **Flexible**: Supports multiple distance metrics (cosine, Euclidean, dot product)
- **Simple Setup**: No database server configuration needed

### Chunk Size: 80–150 Words

- **Trade-off Balance**: Smaller chunks (40–80 words) lose context; larger (200+ words) dilute relevance
- **Sentence Boundaries**: Respects document structure for coherence
- **Retrieval Precision**: Allows top-k to return diverse, non-redundant results

### Prompt Format: Instruction Style

The prompt uses explicit instructions to minimize hallucination:
```
You are a helpful assistant. Answer questions ONLY using the provided context.
If the answer is not in the context, respond with "Not found in document."
```

---

## Limitations & Known Issues

### Hallucination Risk

- **Issue**: LLM may generate plausible but incorrect answers if retrieved context is insufficient
- **Mitigation**: Prompt explicitly instructs "Only use provided context"
- **Note**: Not foolproof; always validate critical information against source

### Retrieval Limitations

- **Semantic Gaps**: If user question uses terminology different from document, retrieval may fail
- **Limited Context**: Top-5 chunks may miss important nuances from longer documents
- **Ambiguous Queries**: Questions without clear keywords may not retrieve relevant sections

### CPU Inference Latency

- **First Response**: ~3–8 seconds (model loading on Ollama)
- **Subsequent Responses**: ~2–5 seconds per response
- **Streaming**: Tokens appear at ~20–50ms intervals depending on CPU
- **Note**: GPU acceleration (CUDA/Metal) would significantly improve speed

### Context Window Limitations

- **Mistral 7B**: 8K token context window
- **Injected Context**: Top-5 chunks (~1500–2000 tokens) leaves room for prompt and response
- **Very Long Documents**: May need to increase top-k chunks or implement multi-hop retrieval

### Chunk Redundancy

- **Overlap**: Sentence-aware chunking may create semantic overlap between adjacent chunks
- **Result**: Multiple chunks retrieved for same concept may not add new information

---

## Sample Queries

Test the chatbot with these example questions based on typical policy documents:

### Expected Success Cases

1. **"What happens if I violate eBay policies?"**
   - ✓ Clear policy question
   - ✓ Likely well-covered in document
   - Expected: Explanation of consequences (suspension, account termination, etc.)

2. **"Can eBay suspend or terminate my account?"**
   - ✓ Direct factual question
   - ✓ Standard in policy documents
   - Expected: Yes, under which conditions

3. **"Can I share my login credentials with third parties?"**
   - ✓ Common user question
   - ✓ Usually explicitly addressed
   - Expected: No, with explanation of risks

4. **"What are the prohibited activities?"**
   - ✓ Broad but document-grounded
   - ✓ Core policy content
   - Expected: List or categories of violations

5. **"How long is the return window?"**
   - ✓ Specific factual question
   - ✓ Data-driven answer
   - Expected: Specific timeframe if in document

### Expected Failure Cases

6. **"How do I contact customer support?" (if not in PDF)**
   - ✗ Out of document scope
   - Expected: "Not found in document."

7. **"What are the best eBay sellers?" (subjective/external knowledge)**
   - ✗ Requires knowledge beyond document
   - Expected: "Not found in document."

8. **"How is blockchain used in eBay?" (if document doesn't mention)**
   - ✗ Topic not covered
   - Expected: "Not found in document."

---

## Troubleshooting

### Ollama Connection Error
```
ConnectionError: Failed to connect to Ollama
```
**Solution**: Ensure Ollama is running in a separate terminal (`ollama serve`)

### No Chunks Retrieved
```
Query returned empty results
```
**Solution**: 
- Verify `vectordb/` directory exists and contains data
- Re-run `python -m src.embed` to rebuild index
- Check if document was properly ingested with `python -m src.ingest`

### Memory Error During Embedding
```
MemoryError: Unable to allocate array
```
**Solution**: 
- Reduce batch size in `embed.py` (default: 32, try 8 or 4)
- Process document in smaller chunks
- Use a lower-dimensional embedding model

### Slow Response Times
**Solutions**:
- Ensure Ollama is running with sufficient resources
- Use GPU if available (`ollama run mistral --gpu`)
- Reduce top-k retrieval count in `retriever.py`

### Streamlit "Port Already in Use"
```
streamlit run app.py --logger.level=debug --server.port 8502
```

---

## Future Enhancements

- **Hybrid Retrieval**: Combine dense (semantic) and sparse (keyword) search
- **Multi-Document Support**: Upload and index multiple PDFs simultaneously
- **Fine-Tuned Embeddings**: Retrain embedding model on domain-specific corpus
- **Query Expansion**: Automatically generate synonym queries for better recall
- **Reranking**: Use cross-encoder to re-rank retrieved chunks by relevance
- **Chat Summarization**: Compress long chat histories for context management
- **Logging & Evaluation**: Track query success rates and failure patterns
- **GPU Acceleration**: Integrate CUDA/Metal for faster inference
- **Deployed API**: REST API for programmatic access beyond Streamlit

---

## References

- [Anthropic RAG Best Practices](https://www.anthropic.com/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Qdrant Vector Database](https://qdrant.tech/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Ollama Documentation](https://github.com/ollama/ollama)

---
