# Smart Internship Assistant - User Guide

## Getting Started

### Starting the Application

#### Method 1: Automatic (Recommended)
```bash
python run_app.py
```
This will start both the API and Streamlit interface automatically.

#### Method 2: Manual
Open two terminals:

**Terminal 1 (API Server):**
```bash
uvicorn app.main:app --reload
```

**Terminal 2 (Streamlit Interface):**
```bash
streamlit run streamlit_app.py
```

### Accessing the Application

Once started, the application will be available at:
- **Streamlit Interface:** http://localhost:8501
- **API Documentation:** http://localhost:8000/docs

## Using the Interface

### 1. Upload Documents

1. Click the **"Browse files"** button in the sidebar
2. Select a PDF file from your computer
3. Click **"Upload Document"**
4. Wait for processing (you'll see a progress spinner)
5. Success! The document is now in your knowledge base

### 2. Ask Questions

1. Type your question in the question input box
2. (Optional) Use advanced options to:
   - Adjust number of sources retrieved
   - Filter by specific documents
3. Click **"Ask Question"**
4. View the AI-generated answer with sources

### 3. Manage Documents

- View all uploaded documents in the sidebar
- Click on a document to see details
- Delete documents you no longer need

## Features

### Answer Quality Indicators

- 🟢 **High Confidence (70%+):** Very reliable answer
- 🟡 **Medium Confidence (40-70%):** Good answer, may need verification
- 🔴 **Low Confidence (<40%):** Limited information available

### Source Citations

Each answer includes:
- Document name
- Page number
- Relevance score
- Excerpt from the source text

## Tips for Best Results

1. **Upload Quality Documents:** Clear, well-formatted PDFs work best
2. **Ask Specific Questions:** More specific questions get better answers
3. **Use Multiple Sources:** Upload several related documents for comprehensive answers
4. **Check Sources:** Always verify important information using the provided sources

## Troubleshooting

### API Not Running
**Error:** "API Server is not running!"

**Solution:** Make sure the API server is started:
```bash
uvicorn app.main:app --reload
```

### No Documents Found
**Issue:** Can't ask questions

**Solution:** Upload at least one PDF document first

### Upload Failed
**Possible Causes:**
- File is too large (>10MB by default)
- File is not a valid PDF
- File is encrypted/password protected

**Solution:** Try a different file or check file properties

## API Configuration

To use OpenAI for better answers, add your API key to `.env`:
```
OPENAI_API_KEY=your-key-here
```

Without an API key, the system will return retrieved context without AI-generated answers.