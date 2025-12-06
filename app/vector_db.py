import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional, Any
from pathlib import Path
from loguru import logger
from config.settings import settings
import uuid
from datetime import datetime


class VectorDatabase:
    """Manage ChromaDB vector database operations"""
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize ChromaDB client
        
        Args:
            persist_directory: Path to persist database
        """
        self.persist_directory = persist_directory or settings.chroma_db_path
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with persistence"""
        try:
            # Ensure directory exists
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.success(
                f"✓ ChromaDB initialized. Collection: {settings.chroma_collection_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to the vector database
        
        Args:
            texts: List of text chunks
            embeddings: List of embeddings for each text
            metadatas: List of metadata dictionaries
            ids: Optional list of custom IDs
            
        Returns:
            Success status
        """
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Validate inputs
            if not (len(texts) == len(embeddings) == len(metadatas) == len(ids)):
                raise ValueError("All input lists must have the same length")
            
            logger.info(f"Adding {len(texts)} documents to ChromaDB")
            
            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.success(f"✓ Added {len(texts)} documents to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict:
        """
        Query the vector database
        
        Args:
            query_embeddings: List of query embeddings
            n_results: Number of results to return per query
            where: Metadata filter (e.g., {"document_id": "123"})
            where_document: Document content filter
            
        Returns:
            Query results dictionary
        """
        try:
            logger.info(f"Querying ChromaDB for top {n_results} results")
            
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.info(f"✓ Retrieved {len(results['ids'][0])} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            raise
    
    def delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete all chunks from a specific document
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Deleting document: {document_id}")
            
            # Delete by metadata filter
            self.collection.delete(
                where={"document_id": document_id}
            )
            
            logger.success(f"✓ Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_document_chunks(self, document_id: str) -> Dict:
        """
        Get all chunks for a specific document
        
        Args:
            document_id: Document ID
            
        Returns:
            Dictionary with chunks and metadata
        """
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            raise
    
    def list_all_documents(self) -> List[Dict]:
        """
        List all unique documents in the database
        
        Returns:
            List of document metadata
        """
        try:
            # Get all items
            all_items = self.collection.get(
                include=["metadatas"]
            )
            
            # Extract unique documents
            documents_dict = {}
            
            for metadata in all_items["metadatas"]:
                doc_id = metadata.get("document_id")
                if doc_id and doc_id not in documents_dict:
                    documents_dict[doc_id] = {
                        "document_id": doc_id,
                        "document_name": metadata.get("document_name"),
                        "upload_date": metadata.get("upload_date"),
                        "total_chunks": metadata.get("total_chunks", 0)
                    }
            
            return list(documents_dict.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            documents = self.list_all_documents()
            
            return {
                "total_chunks": count,
                "total_documents": len(documents),
                "collection_name": settings.chroma_collection_name,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "collection_name": settings.chroma_collection_name,
                "persist_directory": self.persist_directory
            }
    
    def reset_collection(self) -> bool:
        """
        Delete and recreate the collection (USE WITH CAUTION!)
        
        Returns:
            Success status
        """
        try:
            logger.warning("Resetting collection - all data will be deleted!")
            
            # Delete collection
            self.client.delete_collection(name=settings.chroma_collection_name)
            
            # Recreate collection
            self.collection = self.client.create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.success("✓ Collection reset complete")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False