from typing import Dict, List
import bm25s
from loguru import logger


class BM25Index:
    """In-memory BM25 keyword search index, built from the vector database's chunks.

    ChromaDB has no full-text index, so this maintains its own. Since document
    uploads/deletes can happen through a different DocumentStorageManager instance
    than the one this index was built from (see app/routers/documents.py vs
    app/routers/qa.py), staleness can't be pushed via explicit invalidation calls.
    Instead, each search cheaply compares the collection's current chunk count
    against what was last indexed and rebuilds if it changed.
    """

    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.retriever = None
        self.chunk_ids: List[str] = []
        self.corpus: Dict[str, Dict] = {}
        self._indexed_count = -1
        self.build_index()

    def build_index(self):
        """Rebuild the BM25 index from everything currently in ChromaDB"""
        try:
            all_items = self.vector_db.collection.get(include=["documents", "metadatas"])
            ids = all_items.get("ids", [])
            documents = all_items.get("documents", [])
            metadatas = all_items.get("metadatas", [])

            self.chunk_ids = ids
            self.corpus = {
                chunk_id: {"text": doc, "metadata": meta}
                for chunk_id, doc, meta in zip(ids, documents, metadatas)
            }
            self._indexed_count = len(ids)

            if not documents:
                self.retriever = None
                logger.info("BM25 index empty (no documents in collection)")
                return

            tokenized_corpus = bm25s.tokenize(documents, stopwords="en", show_progress=False)
            self.retriever = bm25s.BM25()
            self.retriever.index(tokenized_corpus, show_progress=False)

            logger.success(f"✓ BM25 index built with {len(documents)} chunks")

        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self.retriever = None

    def _refresh_if_stale(self):
        """Rebuild if the underlying collection's chunk count has changed"""
        try:
            current_count = self.vector_db.collection.count()
        except Exception as e:
            logger.error(f"Error checking collection count for BM25 freshness: {e}")
            return

        if current_count != self._indexed_count:
            logger.info(
                f"BM25 index stale (indexed={self._indexed_count}, current={current_count}), rebuilding"
            )
            self.build_index()

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Search the keyword index for the top_k most relevant chunks

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of chunks ranked by BM25 score, each with text/metadata/bm25_score
        """
        self._refresh_if_stale()

        if not self.retriever or not self.chunk_ids:
            return []

        k = min(top_k, len(self.chunk_ids))
        if k == 0:
            return []

        try:
            tokenized_query = bm25s.tokenize(query, stopwords="en", show_progress=False)
            results = self.retriever.retrieve(
                tokenized_query, corpus=self.chunk_ids, k=k, show_progress=False
            )

            ranked = []
            for chunk_id, score in zip(results.documents[0], results.scores[0]):
                item = self.corpus.get(chunk_id)
                if not item:
                    continue
                ranked.append({
                    "chunk_id": chunk_id,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "bm25_score": float(score)
                })

            return ranked

        except Exception as e:
            logger.error(f"Error during BM25 search: {e}")
            return []
