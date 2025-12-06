from typing import Dict, List
from loguru import logger
import re


class AnswerProcessor:
    """Post-process generated answers"""
    
    def __init__(self):
        """Initialize answer processor"""
        pass
    
    def process_answer(
        self,
        answer: str,
        chunks: List[Dict],
        question: str
    ) -> Dict:
        """
        Process and enhance generated answer
        
        Args:
            answer: Raw answer from LLM
            chunks: Retrieved chunks used for context
            question: Original question
            
        Returns:
            Processed answer with metadata
        """
        logger.info("Processing answer")
        
        # Clean answer
        cleaned_answer = self._clean_answer(answer)
        
        # Calculate confidence
        confidence = self._calculate_confidence(chunks, answer)
        
        # Validate answer quality
        is_valid, validation_msg = self._validate_answer(cleaned_answer, question)
        
        result = {
            "answer": cleaned_answer,
            "confidence_score": confidence,
            "is_valid": is_valid,
            "validation_message": validation_msg,
            "answer_length": len(cleaned_answer),
            "num_sources": len(chunks)
        }
        
        return result
    
    def _clean_answer(self, answer: str) -> str:
        """
        Clean and format answer
        
        Args:
            answer: Raw answer
            
        Returns:
            Cleaned answer
        """
        # Remove excessive whitespace
        answer = re.sub(r'\s+', ' ', answer)
        
        # Remove multiple newlines
        answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)
        
        # Trim
        answer = answer.strip()
        
        return answer
    
    def _calculate_confidence(self, chunks: List[Dict], answer: str) -> float:
        """
        Calculate confidence score for the answer
        
        Args:
            chunks: Retrieved chunks
            answer: Generated answer
            
        Returns:
            Confidence score (0-1)
        """
        if not chunks:
            return 0.0
        
        # Factor 1: Average similarity of retrieved chunks
        avg_similarity = sum(c.get("similarity_score", 0) for c in chunks) / len(chunks)
        
        # Factor 2: Number of sources (more sources = higher confidence, up to a point)
        source_factor = min(len(chunks) / 5, 1.0)  # Cap at 5 sources
        
        # Factor 3: Answer length (very short answers might be less confident)
        length_factor = min(len(answer) / 200, 1.0)  # Normalize around 200 chars
        
        # Weighted combination
        confidence = (
            avg_similarity * 0.6 +
            source_factor * 0.2 +
            length_factor * 0.2
        )
        
        return round(min(confidence, 1.0), 3)
    
    def _validate_answer(self, answer: str, question: str) -> tuple:
        """
        Validate answer quality
        
        Args:
            answer: Generated answer
            question: Original question
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check if answer is too short
        if len(answer) < 20:
            return False, "Answer is too short"
        
        # Check for common failure patterns
        failure_patterns = [
            "i don't have enough information",
            "i cannot answer",
            "not available in the context",
            "i don't know",
            "no information",
            "context does not contain"
        ]
        
        answer_lower = answer.lower()
        for pattern in failure_patterns:
            if pattern in answer_lower and len(answer) < 100:
                return False, "Insufficient information in context"
        
        # Check if answer seems relevant to question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words & answer_words)
        
        if overlap < 2 and len(question_words) > 3:
            return False, "Answer may not be relevant to question"
        
        return True, "Answer validated"
    
    def add_citations(self, answer: str, chunks: List[Dict]) -> str:
        """
        Add inline citations to answer
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            Answer with citations
        """
        # For now, just append sources at the end
        # More sophisticated citation matching could be implemented
        
        if not chunks:
            return answer
        
        citations = "\n\n**Sources:**\n"
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            citations += f"{i}. {metadata['document_name']} (Page {metadata.get('page_number', 'N/A')})\n"
        
        return answer + citations