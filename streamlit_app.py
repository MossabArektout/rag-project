"""
Streamlit Interface for Smart Internship Assistant
"""
import streamlit as st
import requests
from pathlib import Path
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Smart Internship Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
    }
    .confidence-medium {
        color: #ffc107;
    }
    .confidence-low {
        color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def upload_document(file):
    """Upload document to API"""
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(
            f"{API_BASE_URL}/api/documents/upload",
            files=files,
            timeout=60
        )
        
        if response.status_code == 201:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, {"detail": str(e)}


def get_documents():
    """Get list of all documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/documents/")
        if response.status_code == 200:
            return response.json()
        return {"total_documents": 0, "documents": []}
    except:
        return {"total_documents": 0, "documents": []}


def delete_document(doc_id):
    """Delete a document"""
    try:
        response = requests.delete(f"{API_BASE_URL}/api/documents/{doc_id}")
        return response.status_code == 200
    except:
        return False


def ask_question(question, doc_ids=None, top_k=5):
    """Ask a question"""
    try:
        payload = {
            "question": question,
            "document_ids": doc_ids,
            "top_k": top_k
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/qa/ask",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, {"detail": str(e)}


def format_confidence(score):
    """Format confidence score with color"""
    if score >= 0.7:
        color_class = "confidence-high"
        emoji = "🟢"
    elif score >= 0.4:
        color_class = "confidence-medium"
        emoji = "🟡"
    else:
        color_class = "confidence-low"
        emoji = "🔴"
    
    return f'{emoji} <span class="{color_class}">{score:.1%}</span>'


def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🤖 Smart Internship Assistant")
    st.markdown("### RAG-Powered Q&A System for Company Knowledge Bases")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API Server is not running! Please start it with: `uvicorn app.main:app --reload`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Get current documents
        docs_data = get_documents()
        total_docs = docs_data.get("total_documents", 0)
        
        st.metric("Total Documents", total_docs)
        
        # Upload section
        st.subheader("Upload New Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload PDF documents to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Upload Document", type="primary", use_container_width=True):
                with st.spinner("Processing document... This may take a moment."):
                    success, result = upload_document(uploaded_file)
                    
                    if success:
                        st.success(f"✅ Document uploaded successfully!")
                        st.json({
                            "Document ID": result["document_id"][:8] + "...",
                            "Name": result["metadata"]["document_name"],
                            "Chunks": result["metadata"]["total_chunks"],
                            "Pages": result["metadata"]["page_count"]
                        })
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {result.get('detail', 'Unknown error')}")
        
        st.divider()
        
        # Document list
        st.subheader("Uploaded Documents")
        
        if total_docs == 0:
            st.info("No documents uploaded yet. Upload your first document above!")
        else:
            documents = docs_data.get("documents", [])
            
            for doc in documents:
                with st.expander(f"📄 {doc['document_name']}", expanded=False):
                    st.write(f"**ID:** {doc['document_id'][:8]}...")
                    st.write(f"**Chunks:** {doc['total_chunks']}")
                    st.write(f"**Uploaded:** {doc['upload_date'][:10]}")
                    
                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{doc['document_id']}",
                        use_container_width=True
                    ):
                        if delete_document(doc['document_id']):
                            st.success("Document deleted!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Failed to delete document")
    
    # Main content area
    if total_docs == 0:
        st.info("👈 Upload some documents using the sidebar to get started!")
        
        st.markdown("""
        ### How to use this system:
        
        1. **Upload Documents**: Use the sidebar to upload PDF files containing your company knowledge
        2. **Ask Questions**: Once documents are uploaded, you can ask questions about their content
        3. **Get AI Answers**: The system will retrieve relevant information and generate natural language answers
        4. **View Sources**: See which documents and pages were used to answer your question
        
        ### Features:
        - 🔍 Semantic search across all uploaded documents
        - 🤖 AI-powered natural language answers
        - 📊 Confidence scoring for answers
        - 📚 Source citations with page numbers
        - ⚡ Fast document processing and retrieval
        """)
    else:
        # Q&A Interface
        st.header("💬 Ask a Question")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What does the company do? What are our main products?",
            help="Ask anything about your uploaded documents"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                top_k = st.slider(
                    "Number of sources to retrieve",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="How many relevant chunks to retrieve"
                )
            
            with col2:
                # Document filter
                doc_names = [doc['document_name'] for doc in documents]
                selected_docs = st.multiselect(
                    "Filter by documents (optional)",
                    options=doc_names,
                    help="Leave empty to search all documents"
                )
                
                # Get doc IDs for selected names
                selected_doc_ids = None
                if selected_docs:
                    selected_doc_ids = [
                        doc['document_id'] 
                        for doc in documents 
                        if doc['document_name'] in selected_docs
                    ]
        
        # Ask button
        if st.button("🔍 Ask Question", type="primary", disabled=not question):
            if question:
                with st.spinner("Thinking... 🤔"):
                    success, result = ask_question(
                        question,
                        doc_ids=selected_doc_ids,
                        top_k=top_k
                    )
                    
                    if success:
                        # Display answer
                        st.markdown("### 💡 Answer")
                        st.markdown(f"**{result['answer']}**")
                        
                        # Metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Confidence",
                                f"{result['confidence_score']:.1%}"
                            )
                        with col2:
                            st.metric(
                                "Sources Used",
                                len(result['sources'])
                            )
                        with col3:
                            st.metric(
                                "Processing Time",
                                f"{result['processing_time_seconds']:.2f}s"
                            )
                        
                        # Sources
                        st.markdown("### 📚 Sources")
                        
                        for i, source in enumerate(result['sources'], 1):
                            with st.container():
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i}: {source['document_name']}</strong> (Page {source['page_number']})
                                    <br/>
                                    Relevance: {format_confidence(source['similarity_score'])}
                                    <br/><br/>
                                    <em>{source['chunk_text'][:300]}{'...' if len(source['chunk_text']) > 300 else ''}</em>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Error: {result.get('detail', 'Unknown error')}")
        
        st.divider()
        
        # Example questions
        st.markdown("### 💭 Example Questions")
        
        example_questions = [
            "What is the company's mission?",
            "What products or services do we offer?",
            "What are our company values?",
            "Tell me about our technology stack",
            "What makes our company unique?"
        ]
        
        cols = st.columns(len(example_questions))
        for col, eq in zip(cols, example_questions):
            with col:
                if st.button(eq, use_container_width=True):
                    st.session_state.example_q = eq
                    st.rerun()
        
        # Handle example question click
        if hasattr(st.session_state, 'example_q'):
            question = st.session_state.example_q
            delattr(st.session_state, 'example_q')


if __name__ == "__main__":
    main()