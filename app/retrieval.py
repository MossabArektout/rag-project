from typing import List, Dict, Optional, Tuple
from loguru import logger
from app.vector_db import VectorDatabase
from app.embeddings import EmbeddingGenerator
from config.settings import settings


class AdvancedRetriever:
    """Advanced retrieval strategies for RAG"""
    
    def __init__(self, vector_db: VectorDatabase, embedding_generator: EmbeddingGenerator):
        """
        Initialize retriever
        
        Args:
            vector_db: Vector database instance
            embedding_generator: Embedding generator instance
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        document_ids: Optional[List[str]] = None,
        similarity_threshold: float = None
    ) -> List[Dict]:
        """
        Basic retrieval with filtering
        
        Args:
            query: Search query
            top_k: Number of results to return
            document_ids: Optional list of document IDs to filter
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieved chunks with metadata
        """
        k = top_k or settings.top_k_results
        threshold = similarity_threshold or settings.similarity_threshold
        
        logger.info(f"Retrieving top {k} chunks for query: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search vector database
        results = self.vector_db.query(
            query_embeddings=[query_embedding],
            n_results=k * 2,  # Get more than needed for filtering
            where=None
        )
        
        # Process and filter results
        retrieved_chunks = []
        
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1 - distance
            
            # Apply filters
            if similarity < threshold:
                continue
            
            if document_ids and metadata["document_id"] not in document_ids:
                continue
            
            chunk = {
                "text": doc,
                "metadata": metadata,
                "similarity_score": round(similarity, 4)
            }
            retrieved_chunks.append(chunk)
            
            # Stop when we have enough
            if len(retrieved_chunks) >= k:
                break
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks above threshold {threshold}")
        
        return retrieved_chunks
    
    def retrieve_with_reranking(
        self,
        query: str,
        top_k: int = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Retrieve with simple re-ranking based on query term overlap
        
        Args:
            query: Search query
            top_k: Number of results to return
            document_ids: Optional list of document IDs to filter
            
        Returns:
            List of re-ranked chunks
        """
        k = top_k or settings.top_k_results
        
        # First, get more candidates than needed
        candidates = self.retrieve(
            query=query,
            top_k=k * 3,
            document_ids=document_ids,
            similarity_threshold=settings.similarity_threshold * 0.8  # Lower threshold
        )
        
        if not candidates:
            return []
        
        logger.info(f"Re-ranking {len(candidates)} candidates")
        
        # Simple re-ranking: boost chunks with exact query terms
        query_terms = set(query.lower().split())
        
        for chunk in candidates:
            text_terms = set(chunk["text"].lower().split())
            overlap = len(query_terms & text_terms)
            
            # Boost score based on term overlap
            boost = 1 + (overlap * 0.1)
            chunk["reranked_score"] = chunk["similarity_score"] * boost
        
        # Sort by re-ranked score
        candidates.sort(key=lambda x: x["reranked_score"], reverse=True)
        
        # Return top k
        return candidates[:k]
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = None,
        context_window: int = 1
    ) -> List[Dict]:
        """
        Retrieve chunks with surrounding context
        
        Args:
            query: Search query
            top_k: Number of results to return
            context_window: Number of adjacent chunks to include
            
        Returns:
            List of chunks with context
        """
        k = top_k or settings.top_k_results
        
        # Get initial chunks
        chunks = self.retrieve(query=query, top_k=k)
        
        if not chunks or context_window == 0:
            return chunks
        
        logger.info(f"Retrieving context (window={context_window}) for {len(chunks)} chunks")
        
        # For each chunk, try to get surrounding chunks
        enriched_chunks = []
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            doc_id = metadata["document_id"]
            chunk_idx = metadata["chunk_index"]
            
            # Get all chunks from the same document
            doc_chunks = self.vector_db.get_document_chunks(doc_id)
            
            # Find the current chunk and its neighbors
            context_text = chunk["text"]
            
            # Add before context
            for i in range(max(0, chunk_idx - context_window), chunk_idx):
                for meta in doc_chunks["metadatas"]:
                    if meta["chunk_index"] == i:
                        idx = doc_chunks["metadatas"].index(meta)
                        context_text = doc_chunks["documents"][idx] + "\n\n" + context_text
                        break
            
            # Add after context
            for i in range(chunk_idx + 1, chunk_idx + context_window + 1):
                for meta in doc_chunks["metadatas"]:
                    if meta["chunk_index"] == i:
                        idx = doc_chunks["metadatas"].index(meta)
                        context_text = context_text + "\n\n" + doc_chunks["documents"][idx]
                        break
            
            enriched_chunk = chunk.copy()
            enriched_chunk["text_with_context"] = context_text
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks