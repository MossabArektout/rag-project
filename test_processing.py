"""
Test script for document processing pipeline
"""
from pathlib import Path
from app.document_processor import PDFProcessor
from app.text_cleaner import TextCleaner
from app.chunking import TextChunker
from loguru import logger
import uuid


def test_document_processing(pdf_path: str):
    """Test the complete document processing pipeline"""
    
    logger.info("="*50)
    logger.info("TESTING DOCUMENT PROCESSING PIPELINE")
    logger.info("="*50)
    
    # Initialize processors
    pdf_processor = PDFProcessor()
    text_cleaner = TextCleaner()
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    
    # Convert to Path
    file_path = Path(pdf_path)
    
    # Step 1: Validate PDF
    logger.info(f"\n1. Validating PDF: {file_path.name}")
    is_valid, error = pdf_processor.validate_pdf(file_path)
    
    if not is_valid:
        logger.error(f"Validation failed: {error}")
        return
    
    logger.success("✓ PDF is valid")
    
    # Step 2: Extract metadata
    logger.info("\n2. Extracting metadata")
    metadata = pdf_processor.get_pdf_metadata(file_path)
    logger.info(f"Pages: {metadata['page_count']}")
    logger.info(f"Size: {metadata['file_size_mb']:.2f} MB")
    
    # Step 3: Extract text
    logger.info("\n3. Extracting text from PDF")
    page_texts = pdf_processor.extract_text(file_path, method="auto")
    total_chars = sum(len(text) for text in page_texts.values())
    logger.info(f"Extracted {len(page_texts)} pages")
    logger.info(f"Total characters: {total_chars:,}")
    
    # Preview first page
    if page_texts:
        first_page = list(page_texts.values())[0]
        logger.info(f"\nFirst 200 chars of page 1:\n{first_page[:200]}...")
    
    # Step 4: Clean text
    logger.info("\n4. Cleaning text")
    cleaned_pages = text_cleaner.clean_pages(page_texts)
    cleaned_chars = sum(len(text) for text in cleaned_pages.values())
    logger.info(f"Cleaned pages: {len(cleaned_pages)}")
    logger.info(f"Characters after cleaning: {cleaned_chars:,}")
    logger.info(f"Removed: {total_chars - cleaned_chars:,} characters")
    
    # Step 5: Chunk text
    logger.info("\n5. Chunking text")
    document_id = str(uuid.uuid4())
    chunks = chunker.chunk_pages(
        cleaned_pages,
        document_id=document_id,
        document_name=file_path.name,
        method="sentences"
    )
    
    # Get chunk statistics
    stats = chunker.get_chunk_stats(chunks)
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Average chunk size: {stats['avg_chunk_size']:.0f} chars")
    logger.info(f"Min chunk size: {stats['min_chunk_size']} chars")
    logger.info(f"Max chunk size: {stats['max_chunk_size']} chars")
    
    # Preview first chunk
    if chunks:
        logger.info(f"\nFirst chunk preview:")
        logger.info(f"Page: {chunks[0].page_number}")
        logger.info(f"Text: {chunks[0].text[:200]}...")
    
    logger.info("\n" + "="*50)
    logger.success("✓ DOCUMENT PROCESSING PIPELINE TEST COMPLETE")
    logger.info("="*50)
    
    return chunks, metadata


if __name__ == "__main__":
    # Test with a sample PDF
    # Replace this with your actual PDF path
    pdf_path = "uploads/sample.pdf"
    
    if not Path(pdf_path).exists():
        logger.error(f"PDF not found: {pdf_path}")
        logger.info("Please place a PDF file at the path above or change the path")
    else:
        chunks, metadata = test_document_processing(pdf_path)