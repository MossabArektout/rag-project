from typing import Dict, List
from sentence_transformers import CrossEncoder
from loguru import logger
import torch


class CrossEncoderReranker:
    """Re-ranks candidate chunks with a cross-encoder.

    Unlike the bi-encoder used for retrieval (query and chunk embedded separately,
    then compared), a cross-encoder scores the (query, chunk) pair jointly, which
    is far more accurate but too slow to run over the whole corpus. So it only
    runs over a small candidate pool (e.g. the RRF top 20) to pick the final top k.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CrossEncoder(self.model_name, device=device)
            logger.success(f"✓ Cross-encoder loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise

    def rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        """
        Re-score candidate chunks against the query and return the top k.

        Args:
            query: Search query
            chunks: Candidate chunks (each must have a "text" field)
            top_k: Number of chunks to return after re-ranking

        Returns:
            Chunks sorted by cross-encoder score, descending, truncated to top_k.
            Each chunk gains a "cross_encoder_score" field.
        """
        if not chunks:
            return []

        pairs = [[query, chunk["text"]] for chunk in chunks]

        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error during cross-encoder re-ranking: {e}")
            return chunks[:top_k]

        for chunk, score in zip(chunks, scores):
            chunk["cross_encoder_score"] = round(float(score), 6)

        chunks.sort(key=lambda c: c["cross_encoder_score"], reverse=True)

        return chunks[:top_k]
