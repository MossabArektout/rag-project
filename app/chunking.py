from typing import List, Dict
from dataclasses import dataclass
from loguru import logger
from config.settings import settings
import re


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int


class TextChunker:
    """Intelligent text chunking with overlap"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = 100
    ):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Sentence boundary markers
        self.sentence_endings = re.compile(r'[.!?]\s+')
    
    def find_sentence_boundaries(self, text: str) -> List[int]:
        """
        Find sentence ending positions in text
        
        Returns:
            List of character positions where sentences end
        """
        boundaries = [0]  # Start of text
        
        for match in self.sentence_endings.finditer(text):
            boundaries.append(match.end())
        
        boundaries.append(len(text))  # End of text
        
        return boundaries
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunk text by sentences, respecting chunk_size and overlap
        
        Returns:
            List of text chunks
        """
        if not text or len(text) < self.min_chunk_size:
            return [text] if text else []
        
        # Find sentence boundaries
        boundaries = self.find_sentence_boundaries(text)
        
        chunks = []
        current_chunk_start = 0
        current_chunk_end = 0
        
        for i in range(1, len(boundaries)):
            # Check if adding this sentence would exceed chunk_size
            potential_end = boundaries[i]
            chunk_length = potential_end - current_chunk_start
            
            if chunk_length >= self.chunk_size:
                # Create chunk up to previous boundary
                if current_chunk_end > current_chunk_start:
                    chunk_text = text[current_chunk_start:current_chunk_end].strip()
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    overlap_start = max(
                        current_chunk_start,
                        current_chunk_end - self.chunk_overlap
                    )
                    current_chunk_start = overlap_start
                    current_chunk_end = potential_end
                else:
                    # First sentence itself is too long
                    chunk_text = text[current_chunk_start:potential_end].strip()
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk_start = potential_end
                    current_chunk_end = potential_end
            else:
                current_chunk_end = potential_end
        
        # Add final chunk
        if current_chunk_end > current_chunk_start:
            chunk_text = text[current_chunk_start:current_chunk_end].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def chunk_by_fixed_size(self, text: str) -> List[str]:
        """
        Simple fixed-size chunking with overlap (fallback method)
        
        Returns:
            List of text chunks
        """
        if not text or len(text) < self.min_chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If not at the end, try to break at a space
            if end < len(text):
                # Look for the last space within the chunk
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos + 1
            
            chunk_text = text[start:end].strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    
    def chunk_pages(
        self,
        page_texts: Dict[int, str],
        document_id: str,
        document_name: str,
        method: str = "sentences"
    ) -> List[Chunk]:
        """
        Chunk all pages from a document
        
        Args:
            page_texts: Dictionary mapping page numbers to text
            document_id: Unique document identifier
            document_name: Document filename
            method: "sentences" or "fixed" chunking method
        
        Returns:
            List of Chunk objects with metadata
        """
        logger.info(f"Chunking document '{document_name}' with {len(page_texts)} pages")
        
        all_chunks = []
        chunk_index = 0
        
        for page_num, page_text in sorted(page_texts.items()):
            if not page_text.strip():
                continue
            
            # Choose chunking method
            if method == "sentences":
                text_chunks = self.chunk_by_sentences(page_text)
            else:
                text_chunks = self.chunk_by_fixed_size(page_text)
            
            # Create Chunk objects with metadata
            for text_chunk in text_chunks:
                chunk = Chunk(
                    text=text_chunk,
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    start_char=0,  # Could calculate actual position if needed
                    end_char=len(text_chunk)
                )
                all_chunks.append(chunk)
                chunk_index += 1
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(page_texts)} pages")
        
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict:
        """
        Get statistics about chunks
        
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_characters': 0
            }
        
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }