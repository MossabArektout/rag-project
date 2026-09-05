from typing import Dict, List, Optional
from loguru import logger
import time

from app.vector_db import VectorDatabase
from app.embeddings import EmbeddingGenerator
from app.retrieval import AdvancedRetriever
from app.context_builder import ContextBuilder
from app.llm_client import LLMClient
from app.answer_processor import AnswerProcessor
from config.settings import settings


class RAGEngine:
    """Complete RAG engine orchestrating retrieval and generation"""
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_generator: EmbeddingGenerator
    ):
        """
        Initialize RAG engine
        
        Args:
            vector_db: Vector database instance
            embedding_generator: Embedding generator instance
        """
        logger.info("Initializing RAG Engine")
        
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        
        # Initialize components
        self.retriever = AdvancedRetriever(vector_db, embedding_generator)
        self.context_builder = ContextBuilder(max_context_length=4000)
        self.llm_client = LLMClient()
        self.answer_processor = AnswerProcessor()
        
        logger.success("✓ RAG Engine initialized")
    
    def answer_question(
        self,
        question: str,
        top_k: int = None,
        document_ids: Optional[List[str]] = None,
        use_reranking: bool = False,
        include_context_window: bool = False
    ) -> Dict:
        """
        Complete RAG pipeline: Retrieve → Augment → Generate
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            document_ids: Optional document filter
            use_reranking: Whether to use re-ranking
            include_context_window: Whether to include surrounding chunks
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        logger.info(f"Processing question: {question}")
        
        try:
            # Step 1: Retrieve relevant chunks
            logger.info("Step 1/4: Retrieving relevant chunks")
            
            if use_reranking:
                chunks = self.retriever.retrieve_with_reranking(
                    query=question,
                    top_k=top_k,
                    document_ids=document_ids
                )
            elif include_context_window:
                chunks = self.retriever.retrieve_with_context(
                    query=question,
                    top_k=top_k,
                    context_window=1
                )
            else:
                chunks = self.retriever.retrieve(
                    query=question,
                    top_k=top_k,
                    document_ids=document_ids
                )
            
            if not chunks:
                logger.warning("No relevant chunks found")
                return {
                    "success": False,
                    "error": "No relevant information found",
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or upload more documents.",
                    "sources": [],
                    "confidence_score": 0.0,
                    "processing_time": time.time() - start_time
                }
            
            logger.info(f"Retrieved {len(chunks)} chunks")
            
            # Step 2: Build context
            logger.info("Step 2/4: Building context")
            context = self.context_builder.build_context(
                chunks,
                include_metadata=True,
                deduplicate=True
            )
            
            # Step 3: Generate answer
            logger.info("Step 3/4: Generating answer")
            
            if not self.llm_client.check_availability():
                # Fallback: return context without LLM
                logger.warning("LLM not available, returning context only")
                answer = self._create_fallback_answer(question, chunks)
            else:
                answer = self.llm_client.generate_answer(
                    question=question,
                    context=context
                )
                
                if not answer:
                    logger.error("LLM failed to generate answer")
                    answer = self._create_fallback_answer(question, chunks)
            
            # Step 4: Post-process answer
            logger.info("Step 4/4: Processing answer")
            processed = self.answer_processor.process_answer(
                answer=answer,
                chunks=chunks,
                question=question
            )
            
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "success": True,
                "question": question,
                "answer": processed["answer"],
                "sources": self._format_sources(chunks),
                "confidence_score": processed["confidence_score"],
                "processing_time": round(processing_time, 3),
                "metadata": {
                    "num_chunks_retrieved": len(chunks),
                    "answer_length": processed["answer_length"],
                    "is_valid": processed["is_valid"],
                    "validation_message": processed["validation_message"],
                    "used_llm": self.llm_client.check_availability()
                }
            }
            
            logger.success(f"Question answered in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "An error occurred while processing your question.",
                "sources": [],
                "confidence_score": 0.0,
                "processing_time": time.time() - start_time
            }
    
    def _create_fallback_answer(self, question: str, chunks: List[Dict]) -> str:
        """
        Create fallback answer when LLM is not available
        
        Args:
            question: User's question
            chunks: Retrieved chunks
            
        Returns:
            Fallback answer
        """
        answer_parts = [
            "Based on the retrieved documents, here are the most relevant passages:\n"
        ]
        
        for i, chunk in enumerate(chunks[:3], 1):
            metadata = chunk["metadata"]
            text = chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"]
            
            answer_parts.append(
                f"\n{i}. From {metadata['document_name']} (Page {metadata.get('page_number', 'N/A')}):\n{text}\n"
            )
        
        answer_parts.append(
            "\n(Note: This response shows raw retrieved content. "
            "For natural language answers, please configure a Groq API key in the .env file.)"
        )
        
        return "\n".join(answer_parts)
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """
        Format chunks as source citations
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            List of formatted sources
        """
        sources = []
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            source = {
                "document_name": metadata["document_name"],
                "page_number": metadata.get("page_number"),
                "chunk_text": chunk["text"],
                "similarity_score": chunk.get("similarity_score", 0)
            }
            sources.append(source)
        
        return sources