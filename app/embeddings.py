from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from loguru import logger
from config.settings import settings
import torch


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self.embedding_dimension = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load model
            self.model = SentenceTransformer(self.model_name, device=device)
            
            # Get embedding dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.success(
                f"✓ Model loaded successfully. Embedding dimension: {self.embedding_dimension}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            List of floats representing the embedding
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.embedding_dimension
        
        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (more efficient)
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            List of embeddings
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Filter out empty texts but keep track of indices
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                logger.warning("No valid texts to embed")
                return [[0.0] * self.embedding_dimension] * len(texts)
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )
            
            # Create result list with proper ordering
            result = [[0.0] * self.embedding_dimension] * len(texts)
            for idx, embedding in zip(valid_indices, embeddings):
                result[idx] = embedding.tolist()
            
            logger.success(f"✓ Generated {len(embeddings)} embeddings")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def compute_similarity(
        self, 
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Convert to numpy arrays if needed
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": str(self.model.device),
            "max_seq_length": self.model.max_seq_length
        }