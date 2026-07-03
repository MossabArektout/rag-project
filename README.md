# 🤖 Smart Internship Assistant

A RAG (Retrieval-Augmented Generation) powered Q&A system for company knowledge bases. Upload PDF documents and ask questions to get AI-powered answers with source citations.

## 📚 About This Project

This is a **learning project** developed to understand and implement RAG (Retrieval-Augmented Generation) systems. It demonstrates key concepts including:
- Document processing and text extraction
- Semantic embeddings and vector storage
- Similarity search and retrieval
- LLM integration for natural language generation
- Full-stack application development with FastAPI and Streamlit

**Developer:** [Mossab Arektout](https://github.com/MossabArektout)

## ✨ Features

- 📄 **PDF Document Processing** - Upload and process company documents
- 🔍 **Semantic Search** - Find relevant information using AI-powered embeddings
- 🤖 **AI-Powered Answers** - Natural language responses powered by Google Gemini
- 📊 **Source Citations** - View which documents and pages were used
- 🎯 **Confidence Scoring** - See how confident the system is in its answers
- 🌐 **Web Interface** - User-friendly Streamlit UI
- 🔌 **REST API** - FastAPI backend for programmatic access

## 🏗️ Architecture

```
┌─────────────┐
│   Streamlit │ ◄── User Interface
│   Frontend  │
└──────┬──────┘
       │
┌──────▼──────┐
│   FastAPI   │ ◄── REST API
│   Backend   │
└──────┬──────┘
       │
┌──────▼──────────────────────────┐
│         RAG Pipeline             │
├──────────────────────────────────┤
│ 1. Document Processing (PyPDF2) │
│ 2. Text Chunking                │
│ 3. Embeddings (Sentence-BERT)   │
│ 4. Vector Storage (ChromaDB)    │
│ 5. Retrieval & Search           │
│ 6. Answer Generation (Gemini)   │
└──────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API Key (free tier available)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/MossabArektout/rag-project.git
cd rag-project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API key**

Create a `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
```

Edit `.env` and add your Google Gemini API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

Get a free API key at: https://makersuite.google.com/app/apikey

### Running the Application

**Option 1: Automatic (Recommended)**
```bash
python run_app.py
```

**Option 2: Manual**

Terminal 1 - API Server:
```bash
uvicorn app.main:app --reload
```

Terminal 2 - Streamlit UI:
```bash
streamlit run streamlit_app.py
```

### Access the Application

- **Web Interface:** http://localhost:8501
- **API Documentation:** http://localhost:8000/docs

## 📖 Usage

1. **Upload Documents**
   - Click "Browse files" in the sidebar
   - Select a PDF document
   - Wait for processing

2. **Ask Questions**
   - Type your question in the input box
   - Click "Ask Question"
   - View AI-generated answer with sources

3. **Manage Documents**
   - View all uploaded documents in sidebar
   - Delete documents you no longer need

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern web framework
- **ChromaDB** - Vector database for embeddings
- **Sentence Transformers** - Generate semantic embeddings
- **PyPDF2 & pdfplumber** - PDF text extraction

### Frontend
- **Streamlit** - Interactive web interface

### AI/ML
- **Google Gemini** - LLM for answer generation
- **all-MiniLM-L6-v2** - Embedding model

## 📂 Project Structure

```
smart-internship-assistant/
├── app/
│   ├── main.py              # FastAPI application
│   ├── llm_client.py        # Gemini API client
│   ├── rag_engine.py        # RAG pipeline orchestration
│   ├── document_processor.py # PDF processing
│   ├── embeddings.py        # Embedding generation
│   ├── vector_db.py         # ChromaDB interface
│   └── routers/
│       ├── documents.py     # Document endpoints
│       └── qa.py            # Q&A endpoints
├── config/
│   ├── settings.py          # Configuration
│   └── logger.py            # Logging setup
├── streamlit_app.py         # Streamlit frontend
├── run_app.py               # Application launcher
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables
```

## ⚙️ Configuration

Key settings in `.env`:

```bash
# API Keys
GOOGLE_API_KEY=your_key_here

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.3
TOP_K_RESULTS=5

# LLM Configuration
LLM_MODEL=gemini-1.5-pro-latest
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
```

## 📝 API Endpoints

### Documents
- `POST /api/documents/upload` - Upload PDF
- `GET /api/documents/` - List all documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete document

### Q&A
- `POST /api/qa/ask` - Ask a question

### Health
- `GET /api/health` - Check API status

Full API documentation: http://localhost:8000/docs

## 🐛 Troubleshooting

**API won't start:**
- Check if port 8000 is already in use
- Ensure all dependencies are installed
- Check logs in `./logs/app.log`

**No answers generated:**
- Verify Google API key is set in `.env`
- Check API key is valid at https://makersuite.google.com
- Ensure you haven't exceeded free tier limits

**Upload fails:**
- Check file is a valid PDF (not encrypted)
- Ensure file size is under 10MB
- Try a different PDF file

## 🤝 Contributing

This is a learning project, but suggestions and improvements are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests
- Share your feedback

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Mossab Arektout**
- GitHub: [@MossabArektout](https://github.com/MossabArektout)

## 🙏 Acknowledgments

- Built as a learning project to understand RAG systems
- Inspired by modern AI-powered document search solutions
- Uses Google Gemini for natural language generation
- ChromaDB for vector storage
- Sentence Transformers for embeddings

---

⭐ **If you find this project helpful for learning RAG systems, please give it a star!**
