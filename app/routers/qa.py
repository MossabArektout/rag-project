from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.models import QuestionRequest, AnswerResponse, SourceCitation
from app.storage_manager import DocumentStorageManager
from app.rag_engine import RAGEngine

router = APIRouter(
    prefix="/api/qa",
    tags=["question-answering"]
)

# Initialize components
storage_manager = DocumentStorageManager()
rag_engine = RAGEngine(
    vector_db=storage_manager.vector_db,
    embedding_generator=storage_manager.embedding_generator
)


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question based on uploaded documents using RAG
    
    - **question**: The question to ask
    - **document_ids**: Optional list of document IDs to search
    - **top_k**: Number of relevant chunks to retrieve (default: 5)
    
    Returns an AI-generated answer with source citations
    """
    try:
        logger.info(f"Received question: {request.question}")
        
        # Validate that we have documents
        stats = storage_manager.get_database_stats()
        if stats["total_documents"] == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents uploaded yet. Please upload documents first."
            )
        
        # Use RAG engine to answer question
        result = rag_engine.answer_question(
            question=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids,
            use_reranking=False,  # Can make this configurable
            include_context_window=False
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("error", "Failed to answer question")
            )
        
        # Format sources
        sources = [
            SourceCitation(
                document_name=source["document_name"],
                page_number=source["page_number"],
                chunk_text=source["chunk_text"],
                similarity_score=round(source["similarity_score"], 3)
            )
            for source in result["sources"]
        ]
        
        # Create response
        response = AnswerResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            confidence_score=result["confidence_score"],
            processing_time_seconds=result["processing_time"]
        )
        
        logger.success(f"Question answered successfully")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}"
        )