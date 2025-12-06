import re
from typing import List
from loguru import logger


class TextCleaner:
    """Clean and normalize extracted text"""
    
    def __init__(self):
        # Patterns for cleaning
        self.multiple_spaces = re.compile(r'\s+')
        self.multiple_newlines = re.compile(r'\n\s*\n\s*\n+')
        self.page_numbers = re.compile(r'^\s*\d+\s*$', re.MULTILINE)
        self.header_footer = re.compile(r'^[\s\-_=]{3,}$', re.MULTILINE)
    
    def remove_excessive_whitespace(self, text: str) -> str:
        """Remove excessive spaces and newlines"""
        # Replace multiple spaces with single space
        text = self.multiple_spaces.sub(' ', text)
        
        # Replace multiple newlines with double newline
        text = self.multiple_newlines.sub('\n\n', text)
        
        return text.strip()
    
    def remove_page_numbers(self, text: str) -> str:
        """Remove standalone page numbers"""
        return self.page_numbers.sub('', text)
    
    def remove_headers_footers(self, text: str) -> str:
        """Remove common header/footer patterns"""
        return self.header_footer.sub('', text)
    
    def fix_hyphenation(self, text: str) -> str:
        """Fix words split by hyphens across lines"""
        # Match word-\n and join them
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Replace common unicode characters
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Ellipsis
            '\xa0': ' ',  # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)
    
    def clean_text(
        self, 
        text: str,
        remove_urls: bool = True,
        remove_emails: bool = False,
        remove_page_nums: bool = True
    ) -> str:
        """
        Apply all cleaning steps
        
        Args:
            text: Raw text to clean
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_page_nums: Whether to remove standalone page numbers
        
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        logger.debug(f"Cleaning text of length {len(text)}")
        
        # Apply cleaning steps in order
        text = self.normalize_unicode(text)
        text = self.fix_hyphenation(text)
        
        if remove_urls:
            text = self.remove_urls(text)
        
        if remove_emails:
            text = self.remove_emails(text)
        
        if remove_page_nums:
            text = self.remove_page_numbers(text)
        
        text = self.remove_headers_footers(text)
        text = self.remove_excessive_whitespace(text)
        
        logger.debug(f"Cleaned text length: {len(text)}")
        
        return text
    
    def clean_pages(self, page_texts: dict) -> dict:
        """
        Clean text from all pages
        
        Args:
            page_texts: Dictionary mapping page numbers to text
        
        Returns:
            Dictionary with cleaned text
        """
        cleaned_pages = {}
        
        for page_num, text in page_texts.items():
            cleaned_text = self.clean_text(text)
            if cleaned_text:  # Only keep non-empty pages
                cleaned_pages[page_num] = cleaned_text
        
        logger.info(f"Cleaned {len(cleaned_pages)} pages")
        return cleaned_pages