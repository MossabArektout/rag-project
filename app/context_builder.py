from typing import List, Dict
from loguru import logger


class ContextBuilder:
    """Build context for LLM prompts from retrieved chunks"""
    
    def __init__(self, max_context_length: int = 4000):
        """
        Initialize context builder
        
        Args:
            max_context_length: Maximum characters in context
        """
        self.max_context_length = max_context_length
    
    def build_context(
        self,
        chunks: List[Dict],
        include_metadata: bool = True,
        deduplicate: bool = True
    ) -> str:
        """
        Build context string from retrieved chunks
        
        Args:
            chunks: List of retrieved chunks
            include_metadata: Whether to include source metadata
            deduplicate: Whether to remove duplicate content
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        logger.info(f"Building context from {len(chunks)} chunks")
        
        # Deduplicate if requested
        if deduplicate:
            chunks = self._deduplicate_chunks(chunks)
            logger.info(f"After deduplication: {len(chunks)} chunks")
        
        # Sort by similarity score (highest first)
        chunks = sorted(chunks, key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        # Build context string
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            # Get text (prefer version with context if available)
            text = chunk.get("text_with_context", chunk["text"])
            
            # Format with metadata
            if include_metadata:
                metadata = chunk["metadata"]
                source_info = f"[Source {i}: {metadata['document_name']}, Page {metadata.get('page_number', 'N/A')}]"
                chunk_text = f"{source_info}\n{text}"
            else:
                chunk_text = text
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > self.max_context_length:
                logger.warning(f"Context length limit reached. Including {i-1}/{len(chunks)} chunks")
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"Built context with {len(context_parts)} chunks ({len(context)} characters)")
        
        return context
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Remove duplicate or highly similar chunks
        
        Args:
            chunks: List of chunks
            
        Returns:
            Deduplicated list
        """
        if not chunks:
            return chunks
        
        unique_chunks = []
        seen_texts = set()
        
        for chunk in chunks:
            text = chunk["text"]
            
            # Use first 100 characters as fingerprint
            fingerprint = text[:100].strip()
            
            if fingerprint not in seen_texts:
                unique_chunks.append(chunk)
                seen_texts.add(fingerprint)
        
        return unique_chunks
    
    def format_sources(self, chunks: List[Dict]) -> str:
        """
        Format source citations
        
        Args:
            chunks: List of chunks with metadata
            
        Returns:
            Formatted citations string
        """
        if not chunks:
            return "No sources available."
        
        citations = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            similarity = chunk.get("similarity_score", 0)
            
            citation = (
                f"{i}. {metadata['document_name']} "
                f"(Page {metadata.get('page_number', 'N/A')}) "
                f"- Relevance: {similarity:.1%}"
            )
            citations.append(citation)
        
        return "\n".join(citations)