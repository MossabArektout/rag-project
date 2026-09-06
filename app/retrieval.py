from typing import List, Dict, Optional, Tuple
from loguru import logger
from app.vector_db import VectorDatabase
from app.embeddings import EmbeddingGenerator
from app.bm25_index import BM25Index
from app.reranker import CrossEncoderReranker
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
        self.bm25_index = BM25Index(vector_db)
        self._cross_encoder: Optional[CrossEncoderReranker] = None

    @property
    def cross_encoder(self) -> CrossEncoderReranker:
        """Lazily load the cross-encoder, since it's only needed when reranking is enabled"""
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoderReranker(settings.cross_encoder_model)
        return self._cross_encoder
    
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
    
    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = None,
        document_ids: Optional[List[str]] = None,
        candidate_pool_size: int = None,
        rrf_k: int = None
    ) -> List[Dict]:
        """
        Hybrid retrieval: combine semantic search (ChromaDB embeddings) with
        keyword search (BM25), merged via Reciprocal Rank Fusion (RRF).

        Semantic search alone misses exact keyword/code matches (e.g. "Section 4.2"),
        since embeddings compress meaning rather than literal terms. BM25 alone misses
        paraphrases and synonyms. RRF combines both rankings without needing to
        normalize/compare their very different score scales directly.

        Args:
            query: Search query
            top_k: Number of final results to return
            document_ids: Optional list of document IDs to filter
            candidate_pool_size: How many candidates to pull from each method before fusion
            rrf_k: RRF constant (higher = flatter weighting of rank position)

        Returns:
            List of fused chunks, ranked by combined RRF score
        """
        k = top_k or settings.top_k_results
        pool = candidate_pool_size or settings.hybrid_candidate_pool_size
        rrf_constant = rrf_k or settings.rrf_k

        # Semantic candidates - no similarity threshold filtering here, since RRF
        # needs the full rank ordering rather than a hard cutoff
        query_embedding = self.embedding_generator.generate_embedding(query)
        raw_results = self.vector_db.query(query_embeddings=[query_embedding], n_results=pool)

        semantic_ranked: List[Tuple[str, Dict]] = []
        for chunk_id, doc, metadata, distance in zip(
            raw_results["ids"][0],
            raw_results["documents"][0],
            raw_results["metadatas"][0],
            raw_results["distances"][0]
        ):
            if document_ids and metadata["document_id"] not in document_ids:
                continue
            semantic_ranked.append((chunk_id, {
                "text": doc,
                "metadata": metadata,
                "similarity_score": round(1 - distance, 4)
            }))

        # Keyword candidates via BM25
        keyword_ranked: List[Tuple[str, Dict]] = []
        for item in self.bm25_index.search(query, top_k=pool):
            if document_ids and item["metadata"]["document_id"] not in document_ids:
                continue
            keyword_ranked.append((item["chunk_id"], item))

        # Reciprocal Rank Fusion: score = sum of 1/(rrf_k + rank) across both lists
        fused_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict] = {}

        for rank, (chunk_id, chunk) in enumerate(semantic_ranked, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_constant + rank)
            chunk_lookup.setdefault(chunk_id, chunk)

        for rank, (chunk_id, chunk) in enumerate(keyword_ranked, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_constant + rank)
            chunk_lookup.setdefault(chunk_id, {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "similarity_score": 0.0  # keyword-only match, no semantic score available
            })

        if not fused_scores:
            logger.warning("Hybrid search found no candidates from either method")
            return []

        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:k]

        results = []
        for chunk_id in ranked_ids:
            chunk = dict(chunk_lookup[chunk_id])
            chunk["rrf_score"] = round(fused_scores[chunk_id], 6)
            results.append(chunk)

        logger.info(
            f"Hybrid search: {len(semantic_ranked)} semantic + {len(keyword_ranked)} keyword "
            f"candidates -> {len(results)} fused results"
        )

        return results

    def retrieve_hybrid_reranked(
        self,
        query: str,
        top_k: int = None,
        document_ids: Optional[List[str]] = None,
        rerank_pool_size: int = None
    ) -> List[Dict]:
        """
        Hybrid retrieval followed by cross-encoder re-ranking.

        RRF fuses semantic + keyword rankings cheaply but only looks at rank
        position, not the actual query-chunk relevance. The cross-encoder is far
        more accurate but too slow to run over the whole corpus, so it only
        re-scores the RRF candidate pool (default top 20) to pick the final top k.

        Args:
            query: Search query
            top_k: Number of final results to return
            document_ids: Optional list of document IDs to filter
            rerank_pool_size: How many RRF candidates to feed into the cross-encoder

        Returns:
            List of chunks re-ranked by cross-encoder score
        """
        k = top_k or settings.top_k_results
        pool_size = rerank_pool_size or settings.rerank_candidate_pool_size

        candidates = self.retrieve_hybrid(
            query=query,
            top_k=pool_size,
            document_ids=document_ids
        )

        if not candidates:
            return []

        logger.info(f"Re-ranking {len(candidates)} RRF candidates with cross-encoder")

        return self.cross_encoder.rerank(query, candidates, top_k=k)

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