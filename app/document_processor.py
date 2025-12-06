import PyPDF2
import pdfplumber
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import re


class PDFProcessor:
    """Handle PDF text extraction with multiple strategies"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def validate_pdf(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate PDF file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not file_path.exists():
                return False, "File does not exist"
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_extensions:
                return False, f"Invalid file type. Only {self.supported_extensions} supported"
            
            # Check if file is readable
            if not file_path.is_file():
                return False, "Path is not a file"
            
            # Try to open with PyPDF2
            with open(file_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    # Check if encrypted
                    if pdf_reader.is_encrypted:
                        return False, "PDF is encrypted/password protected"
                    
                    # Check if has pages
                    if len(pdf_reader.pages) == 0:
                        return False, "PDF has no pages"
                    
                except PyPDF2.errors.PdfReadError as e:
                    return False, f"Invalid or corrupted PDF: {str(e)}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating PDF: {e}")
            return False, f"Validation error: {str(e)}"
    
    def extract_text_pypdf2(self, file_path: Path) -> Dict[int, str]:
        """
        Extract text using PyPDF2 (fast but less accurate)
        
        Returns:
            Dictionary mapping page numbers to text content
        """
        page_texts = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    try:
                        text = page.extract_text()
                        page_texts[page_num] = text if text else ""
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        page_texts[page_num] = ""
            
            logger.info(f"Extracted {len(page_texts)} pages using PyPDF2")
            return page_texts
            
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            raise
    
    def extract_text_pdfplumber(self, file_path: Path) -> Dict[int, str]:
        """
        Extract text using pdfplumber (slower but more accurate, handles tables)
        
        Returns:
            Dictionary mapping page numbers to text content
        """
        page_texts = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text()
                        page_texts[page_num] = text if text else ""
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        page_texts[page_num] = ""
            
            logger.info(f"Extracted {len(page_texts)} pages using pdfplumber")
            return page_texts
            
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            raise
    
    def extract_text(self, file_path: Path, method: str = "auto") -> Dict[int, str]:
        """
        Extract text from PDF using specified method
        
        Args:
            file_path: Path to PDF file
            method: "pypdf2", "pdfplumber", or "auto" (tries both)
        
        Returns:
            Dictionary mapping page numbers to text content
        """
        logger.info(f"Extracting text from {file_path.name} using {method} method")
        
        # Validate first
        is_valid, error = self.validate_pdf(file_path)
        if not is_valid:
            raise ValueError(f"Invalid PDF: {error}")
        
        if method == "pypdf2":
            return self.extract_text_pypdf2(file_path)
        
        elif method == "pdfplumber":
            return self.extract_text_pdfplumber(file_path)
        
        elif method == "auto":
            # Try PyPDF2 first (faster)
            try:
                page_texts = self.extract_text_pypdf2(file_path)
                
                # Check if extraction was successful (has actual content)
                total_chars = sum(len(text) for text in page_texts.values())
                
                if total_chars < 100:  # Very little text extracted
                    logger.warning("PyPDF2 extracted minimal text, trying pdfplumber")
                    page_texts = self.extract_text_pdfplumber(file_path)
                
                return page_texts
                
            except Exception as e:
                logger.warning(f"PyPDF2 failed, falling back to pdfplumber: {e}")
                return self.extract_text_pdfplumber(file_path)
        
        else:
            raise ValueError(f"Invalid method: {method}. Use 'pypdf2', 'pdfplumber', or 'auto'")
    
    def get_pdf_metadata(self, file_path: Path) -> Dict:
        """
        Extract PDF metadata
        
        Returns:
            Dictionary with metadata
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                metadata = {
                    'page_count': len(pdf_reader.pages),
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                    'is_encrypted': pdf_reader.is_encrypted,
                }
                
                # Extract PDF info if available
                if pdf_reader.metadata:
                    metadata.update({
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                    })
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {
                'page_count': 0,
                'file_size_mb': 0,
                'is_encrypted': False,
            }