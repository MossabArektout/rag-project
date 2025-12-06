from fastapi import APIRouter, HTTPException, status
from loguru import logger
from typing import List
import time

from app.models import QuestionRequest, AnswerResponse, SourceCitation
from app.storage_manager import DocumentStorageManager
from config.settings import settings

# We'll implement LLM integration in Phase 5
# For now, this is a placeholder that returns retrieved context

router = APIRouter(
    prefix="/api/qa",
    tags=["question-answering"]
)

# Initialize storage manager
storage_manager = DocumentStorageManager()


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question based on uploaded documents
    
    - **question**: The question to ask
    - **document_ids**: Optional list of document IDs to search (searches all if not provided)
    - **top_k**: Number of relevant chunks to retrieve (default: 5)
    
    Returns an answer with source citations
    """
    try:
        start_time = time.time()
        
        logger.info(f"Received question: {request.question}")
        
        # Validate that we have documents
        stats = storage_manager.get_database_stats()
        if stats["total_documents"] == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents uploaded yet. Please upload documents first."
            )
        
        # Generate embedding for the question
        logger.info("Generating question embedding")
        question_embedding = storage_manager.embedding_generator.generate_embedding(
            request.question
        )
        
        # Prepare metadata filter if document_ids provided
        where_filter = None
        if request.document_ids:
            # Verify documents exist
            for doc_id in request.document_ids:
                doc_info = storage_manager.get_document_info(doc_id)
                if "error" in doc_info:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Document not found: {doc_id}"
                    )
            
            # ChromaDB doesn't support 'in' operator directly in current version
            # For now, we'll search all and filter in code
            # In production, you might want to do multiple queries
            logger.info(f"Filtering by documents: {request.document_ids}")
        
        # Search vector database
        logger.info(f"Searching for top {request.top_k} relevant chunks")
        results = storage_manager.vector_db.query(
            query_embeddings=[question_embedding],
            n_results=request.top_k or settings.top_k_results,
            where=where_filter
        )
        
        # Process results
        if not results["ids"][0]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant information found for your question"
            )
        
        # Filter by document_ids if provided
        filtered_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            if request.document_ids:
                if metadata["document_id"] in request.document_ids:
                    filtered_results["documents"][0].append(doc)
                    filtered_results["metadatas"][0].append(metadata)
                    filtered_results["distances"][0].append(distance)
            else:
                filtered_results["documents"][0].append(doc)
                filtered_results["metadatas"][0].append(metadata)
                filtered_results["distances"][0].append(distance)
        
        if not filtered_results["documents"][0]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant information found in the specified documents"
            )
        
        # Create source citations
        sources = []
        for doc, metadata, distance in zip(
            filtered_results["documents"][0],
            filtered_results["metadatas"][0],
            filtered_results["distances"][0]
        ):
            similarity = 1 - distance  # Convert distance to similarity
            
            # Only include if above threshold
            if similarity >= settings.similarity_threshold:
                citation = SourceCitation(
                    document_name=metadata["document_name"],
                    page_number=metadata.get("page_number"),
                    chunk_text=doc,
                    similarity_score=round(similarity, 3)
                )
                sources.append(citation)
        
        if not sources:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No results above similarity threshold ({settings.similarity_threshold})"
            )
        
        # For now, create a simple answer from retrieved context
        # In Phase 5, we'll use an LLM to generate proper answers
        context_summary = "\n\n".join([
            f"[From {source.document_name}, Page {source.page_number}]: {source.chunk_text[:200]}..."
            for source in sources[:3]
        ])
        
        placeholder_answer = (
            f"Based on the documents, here are the most relevant passages:\n\n"
            f"{context_summary}\n\n"
            f"(Note: This is a placeholder response. LLM integration will be added in Phase 5 "
            f"to provide natural language answers.)"
        )
        
        # Calculate confidence (average similarity of top sources)
        avg_similarity = sum(s.similarity_score for s in sources) / len(sources)
        
        processing_time = time.time() - start_time
        
        response = AnswerResponse(
            question=request.question,
            answer=placeholder_answer,
            sources=sources,
            confidence_score=round(avg_similarity, 3),
            processing_time_seconds=round(processing_time, 3)
        )
        
        logger.success(f"Question answered in {processing_time:.2f}s with {len(sources)} sources")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}"
        )