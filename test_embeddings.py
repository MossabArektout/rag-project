"""
Test script for embedding and vector storage
"""
from pathlib import Path
from app.storage_manager import DocumentStorageManager
from loguru import logger
import time


def test_embedding_and_storage(pdf_path: str):
    """Test the complete embedding and storage pipeline"""
    
    logger.info("="*60)
    logger.info("TESTING EMBEDDING AND VECTOR STORAGE PIPELINE")
    logger.info("="*60)
    
    # Initialize storage manager
    storage_manager = DocumentStorageManager()
    
    file_path = Path(pdf_path)
    
    if not file_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Test 1: Process and store document
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Processing and storing document")
    logger.info("="*60)
    
    start_time = time.time()
    success, metadata = storage_manager.process_and_store_document(file_path)
    end_time = time.time()
    
    if success:
        logger.success(f"✓ Document processed in {end_time - start_time:.2f} seconds")
        logger.info(f"\nDocument Metadata:")
        for key, value in metadata.items():
            if key != "chunk_stats":
                logger.info(f"  {key}: {value}")
        
        logger.info(f"\nChunk Statistics:")
        if "chunk_stats" in metadata:
            for key, value in metadata["chunk_stats"].items():
                logger.info(f"  {key}: {value}")
        
        document_id = metadata["document_id"]
    else:
        logger.error(f"✗ Processing failed: {metadata.get('error')}")
        return
    
    # Test 2: Get database statistics
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Database statistics")
    logger.info("="*60)
    
    stats = storage_manager.get_database_stats()
    logger.info(f"Total documents: {stats['total_documents']}")
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Embedding model: {stats['embedding_model']['model_name']}")
    logger.info(f"Embedding dimension: {stats['embedding_model']['embedding_dimension']}")
    
    # Test 3: List all documents
    logger.info("\n" + "="*60)
    logger.info("TEST 3: List all documents")
    logger.info("="*60)
    
    documents = storage_manager.list_all_documents()
    logger.info(f"Found {len(documents)} documents:")
    for doc in documents:
        logger.info(f"  - {doc['document_name']} (ID: {doc['document_id'][:8]}...)")
    
    # Test 4: Get specific document info
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Get document information")
    logger.info("="*60)
    
    doc_info = storage_manager.get_document_info(document_id)
    logger.info(f"Document: {doc_info.get('document_name')}")
    logger.info(f"Chunks: {doc_info.get('total_chunks')}")
    logger.info(f"Upload date: {doc_info.get('upload_date')}")
    
    # Test 5: Test semantic search (preview for Phase 5)
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Test semantic search")
    logger.info("="*60)
    
    test_query = "What does the company do?"
    logger.info(f"Query: '{test_query}'")
    
    # Generate query embedding
    query_embedding = storage_manager.embedding_generator.generate_embedding(test_query)
    
    # Search vector database
    results = storage_manager.vector_db.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    logger.info(f"\nTop 3 results:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 - distance  # Convert distance to similarity
        logger.info(f"\n  Result {i} (similarity: {similarity:.3f}):")
        logger.info(f"  Page: {metadata['page_number']}")
        logger.info(f"  Text: {doc[:200]}...")
    
    logger.info("\n" + "="*60)
    logger.success("✓ ALL TESTS COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    
    return document_id


if __name__ == "__main__":
    # Test with sample PDF
    pdf_path = "uploads/sample.pdf"
    
    if not Path(pdf_path).exists():
        logger.error(f"PDF not found: {pdf_path}")
        logger.info("Run: python create_sample_pdf.py first")
    else:
        test_embedding_and_storage(pdf_path)