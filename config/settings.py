from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = Field(default="Smart Internship Assistant")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=True)
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    
    # Paths
    upload_folder: str = Field(default="./uploads")
    chroma_db_path: str = Field(default="./chroma_db")
    log_path: str = Field(default="./logs")
    
    # File Upload
    max_file_size_mb: int = Field(default=10)
    allowed_extensions: str = Field(default="pdf")
    
    # Document Processing
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    
    # RAG Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    top_k_results: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)
    
    # LLM Configuration
    llm_model: str = Field(default="gpt-3.5-turbo")
    llm_temperature: float = Field(default=0.3)
    llm_max_tokens: int = Field(default=500)
    
    # Database
    chroma_collection_name: str = Field(default="company_knowledge")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings()