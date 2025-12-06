from typing import List, Dict, Tuple
from pathlib import Path
from loguru import logger
from datetime import datetime
import uuid

from app.document_processor import PDFProcessor
from app.text_cleaner import TextCleaner
from app.chunking import TextChunker, Chunk
from app.embeddings import EmbeddingGenerator
from app.vector_db import VectorDatabase


class DocumentStorageManager:
    """Orchestrate the complete document storage pipeline"""
    
    def __init__(self):
        """Initialize all components"""
        logger.info("Initializing Document Storage Manager")
        
        self.pdf_processor = PDFProcessor()
        self.text_cleaner = TextCleaner()
        self.chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase()
        
        logger.success("✓ All components initialized")
    
    def process_and_store_document(
        self,
        file_path: Path,
        document_id: str = None
    ) -> Tuple[bool, Dict]:
        """
        Complete pipeline: Extract → Clean → Chunk → Embed → Store
        
        Args:
            file_path: Path to PDF file
            document_id: Optional custom document ID
            
        Returns:
            Tuple of (success, metadata_dict)
        """
        try:
            logger.info(f"Starting document processing: {file_path.name}")
            
            # Generate document ID
            doc_id = document_id or str(uuid.uuid4())
            upload_date = datetime.now().isoformat()
            
            # Step 1: Validate PDF
            logger.info("Step 1/6: Validating PDF")
            is_valid, error = self.pdf_processor.validate_pdf(file_path)
            if not is_valid:
                logger.error(f"Validation failed: {error}")
                return False, {"error": error}
            
            # Step 2: Extract metadata
            logger.info("Step 2/6: Extracting metadata")
            pdf_metadata = self.pdf_processor.get_pdf_metadata(file_path)
            
            # Step 3: Extract text
            logger.info("Step 3/6: Extracting text")
            page_texts = self.pdf_processor.extract_text(file_path)
            
            if not page_texts:
                error_msg = "No text could be extracted from PDF"
                logger.error(error_msg)
                return False, {"error": error_msg}
            
            # Step 4: Clean text
            logger.info("Step 4/6: Cleaning text")
            cleaned_pages = self.text_cleaner.clean_pages(page_texts)
            
            # Step 5: Chunk text
            logger.info("Step 5/6: Chunking text")
            chunks = self.chunker.chunk_pages(
                cleaned_pages,
                document_id=doc_id,
                document_name=file_path.name,
                method="sentences"
            )
            
            if not chunks:
                error_msg = "No chunks created from document"
                logger.error(error_msg)
                return False, {"error": error_msg}
            
            # Get chunk statistics
            chunk_stats = self.chunker.get_chunk_stats(chunks)
            
            # Step 6: Generate embeddings and store
            logger.info("Step 6/6: Generating embeddings and storing")
            success = self._embed_and_store_chunks(chunks, upload_date)
            
            if not success:
                return False, {"error": "Failed to store chunks"}
            
            # Prepare metadata response
            metadata = {
                "document_id": doc_id,
                "document_name": file_path.name,
                "upload_date": upload_date,
                "total_chunks": len(chunks),
                "file_size_mb": pdf_metadata["file_size_mb"],
                "page_count": pdf_metadata["page_count"],
                "chunk_stats": chunk_stats,
                "processing_status": "completed"
            }
            
            logger.success(f"✓ Document processed successfully: {file_path.name}")
            logger.info(f"  - Pages: {pdf_metadata['page_count']}")
            logger.info(f"  - Chunks: {len(chunks)}")
            logger.info(f"  - Document ID: {doc_id}")
            
            return True, metadata
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return False, {"error": str(e)}
    
    def _embed_and_store_chunks(
        self,
        chunks: List[Chunk],
        upload_date: str
    ) -> bool:
        """
        Generate embeddings for chunks and store in vector database
        
        Args:
            chunks: List of Chunk objects
            upload_date: ISO format upload timestamp
            
        Returns:
            Success status
        """
        try:
            # Extract texts from chunks
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings (batch processing for efficiency)
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            embeddings = self.embedding_generator.generate_embeddings_batch(
                texts,
                batch_size=32,
                show_progress=True
            )
            
            # Prepare metadata for each chunk
            metadatas = []
            ids = []
            
            for chunk in chunks:
                metadata = {
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "upload_date": upload_date,
                    "total_chunks": len(chunks)
                }
                metadatas.append(metadata)
                
                # Create unique ID for each chunk
                chunk_id = f"{chunk.document_id}_{chunk.chunk_index}"
                ids.append(chunk_id)
            
            # Store in vector database
            logger.info("Storing chunks in vector database")
            self.vector_db.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.success(f"✓ Stored {len(chunks)} chunks in vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error embedding and storing chunks: {e}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Deleting document: {document_id}")
            success = self.vector_db.delete_by_document_id(document_id)
            
            if success:
                logger.success(f"✓ Document deleted: {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_document_info(self, document_id: str) -> Dict:
        """
        Get information about a stored document
        
        Args:
            document_id: Document ID
            
        Returns:
            Document metadata
        """
        try:
            chunks = self.vector_db.get_document_chunks(document_id)
            
            if not chunks["ids"]:
                return {"error": "Document not found"}
            
            # Get metadata from first chunk
            first_metadata = chunks["metadatas"][0]
            
            return {
                "document_id": document_id,
                "document_name": first_metadata.get("document_name"),
                "upload_date": first_metadata.get("upload_date"),
                "total_chunks": len(chunks["ids"]),
                "page_count": first_metadata.get("page_number", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return {"error": str(e)}
    
    def list_all_documents(self) -> List[Dict]:
        """
        List all stored documents
        
        Returns:
            List of document metadata
        """
        return self.vector_db.list_all_documents()
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Statistics dictionary
        """
        stats = self.vector_db.get_collection_stats()
        
        # Add embedding model info
        model_info = self.embedding_generator.get_model_info()
        stats.update({"embedding_model": model_info})
        
        return stats