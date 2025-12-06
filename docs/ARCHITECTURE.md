# System Architecture

## Overview

The Smart Internship Assistant uses a Retrieval-Augmented Generation (RAG) architecture to answer questions based on uploaded PDF documents.

## Data Flow

### 1. Document Upload Flow
```
User → FastAPI Upload Endpoint
    → PDF Validation
    → Text Extraction (PyPDF2/pdfplumber)
    → Text Chunking (overlap strategy)
    → Embedding Generation (sentence-transformers)
    → ChromaDB Storage (with metadata)
    → Return Document ID
```

### 2. Question-Answer Flow
```
User → FastAPI Ask Endpoint
    → Question Validation
    → Question Embedding (same model)
    → Similarity Search (ChromaDB)
    → Retrieve Top K Chunks
    → Context Assembly
    → LLM Prompt Construction
    → OpenAI API Call
    → Answer Post-Processing
    → Return Answer + Citations
```

## Components

### FastAPI Application (`app/main.py`)
- HTTP server and routing
- Request/response handling
- Middleware (CORS, error handling)
- Dependency injection

### Document Processor (`app/document_processor.py`)
- PDF text extraction
- Text cleaning and normalization
- Intelligent chunking with overlap
- Metadata extraction

### Embeddings Module (`app/embeddings.py`)
- Embedding model initialization
- Batch embedding generation
- ChromaDB client management
- Vector storage and retrieval

### RAG Engine (`app/rag_engine.py`)
- Semantic search orchestration
- Context assembly
- LLM integration
- Answer generation

## Technology Choices

### Embedding Model: all-MiniLM-L6-v2
- **Pros**: Fast, free, runs locally, 384-dimension vectors
- **Cons**: Lower quality than larger models
- **Use Case**: Development and small-scale deployment

### Vector Database: ChromaDB
- **Pros**: Simple, embedded, persistent storage
- **Cons**: Not ideal for large-scale production
- **Use Case**: Prototyping and small to medium datasets

### LLM: OpenAI GPT-3.5-turbo
- **Pros**: Fast, cost-effective, good quality
- **Cons**: Requires API key, costs money
- **Alternative**: GPT-4 for better quality

## Database Schema

### ChromaDB Collection: `company_knowledge`

**Stored for each chunk:**
- `id`: Unique chunk identifier
- `embedding`: 384-dimension vector
- `document`: The actual text chunk
- `metadata`:
  - `document_id`: Source document UUID
  - `document_name`: Original filename
  - `page_number`: Page in source PDF
  - `chunk_index`: Position in document
  - `upload_date`: ISO timestamp
  - `total_chunks`: Total chunks in document

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/upload` | Upload PDF document |
| POST | `/api/ask` | Ask question |
| GET | `/api/documents` | List all documents |
| GET | `/api/documents/{id}` | Get document details |
| DELETE | `/api/documents/{id}` | Delete document |
| GET | `/api/health` | Health check |

## Configuration

All configuration via environment variables (`.env` file):
- API keys
- Model names
- Chunk sizes
- Retrieval parameters
- Path configurations

## Future Enhancements

- Support for DOCX, TXT files
- Multi-language support
- User authentication
- Document categorization
- Conversation memory
- Web UI