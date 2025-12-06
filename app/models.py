from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DocumentMetadata(BaseModel):
    """Metadata for uploaded documents"""
    document_id: str
    document_name: str
    upload_date: datetime
    total_chunks: int
    file_size_mb: float
    page_count: Optional[int] = None


class UploadResponse(BaseModel):
    """Response after document upload"""
    success: bool
    message: str
    document_id: str
    metadata: DocumentMetadata


class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., min_length=1, max_length=500)
    document_ids: Optional[List[str]] = Field(
        default=None, 
        description="Filter by specific documents"
    )
    top_k: Optional[int] = Field(
        default=5, 
        ge=1, 
        le=10,
        description="Number of chunks to retrieve"
    )


class SourceCitation(BaseModel):
    """Source citation for answers"""
    document_name: str
    page_number: Optional[int] = None
    chunk_text: str
    similarity_score: float


class AnswerResponse(BaseModel):
    """Response model for answers"""
    question: str
    answer: str
    sources: List[SourceCitation]
    confidence_score: float
    processing_time_seconds: float


class DocumentListResponse(BaseModel):
    """Response for listing documents"""
    total_documents: int
    documents: List[DocumentMetadata]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    database_status: str
    total_documents: int
    total_chunks: int


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)