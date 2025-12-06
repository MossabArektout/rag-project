from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List
from pathlib import Path
import shutil
from loguru import logger
from datetime import datetime
import os

from app.models import (
    UploadResponse,
    DocumentMetadata,
    DocumentListResponse,
    ErrorResponse
)
from app.storage_manager import DocumentStorageManager
from config.settings import settings

router = APIRouter(
    prefix="/api/documents",
    tags=["documents"]
)

# Initialize storage manager
storage_manager = DocumentStorageManager()


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document
    
    - **file**: PDF file to upload
    
    Returns document metadata after processing
    """
    try:
        logger.info(f"Received upload request: {file.filename}")
        
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()  # Get position (size)
        file.file.seek(0)  # Reset to beginning
        
        max_size = settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Ensure upload directory exists
        upload_dir = Path(settings.upload_folder)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file temporarily
        file_path = upload_dir / file.filename
        
        # Handle duplicate filenames
        counter = 1
        original_path = file_path
        while file_path.exists():
            stem = original_path.stem
            suffix = original_path.suffix
            file_path = upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to: {file_path}")
        
        # Process and store document
        success, metadata = storage_manager.process_and_store_document(file_path)
        
        # Clean up uploaded file after processing
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not remove temporary file: {e}")
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=metadata.get("error", "Failed to process document")
            )
        
        # Create response
        doc_metadata = DocumentMetadata(
            document_id=metadata["document_id"],
            document_name=metadata["document_name"],
            upload_date=datetime.fromisoformat(metadata["upload_date"]),
            total_chunks=metadata["total_chunks"],
            file_size_mb=metadata["file_size_mb"],
            page_count=metadata.get("page_count")
        )
        
        response = UploadResponse(
            success=True,
            message="Document uploaded and processed successfully",
            document_id=metadata["document_id"],
            metadata=doc_metadata
        )
        
        logger.success(f"Document uploaded successfully: {metadata['document_id']}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents
    
    Returns list of all documents with metadata
    """
    try:
        logger.info("Listing all documents")
        
        documents = storage_manager.list_all_documents()
        
        # Convert to response format
        doc_list = []
        for doc in documents:
            doc_metadata = DocumentMetadata(
                document_id=doc["document_id"],
                document_name=doc["document_name"],
                upload_date=datetime.fromisoformat(doc["upload_date"]),
                total_chunks=doc["total_chunks"],
                file_size_mb=0.0,  # Not stored in this method
                page_count=None
            )
            doc_list.append(doc_metadata)
        
        response = DocumentListResponse(
            total_documents=len(doc_list),
            documents=doc_list
        )
        
        logger.info(f"Retrieved {len(doc_list)} documents")
        
        return response
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id: str):
    """
    Get information about a specific document
    
    - **document_id**: Unique document identifier
    
    Returns document metadata
    """
    try:
        logger.info(f"Getting document info: {document_id}")
        
        doc_info = storage_manager.get_document_info(document_id)
        
        if "error" in doc_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=doc_info["error"]
            )
        
        metadata = DocumentMetadata(
            document_id=doc_info["document_id"],
            document_name=doc_info["document_name"],
            upload_date=datetime.fromisoformat(doc_info["upload_date"]),
            total_chunks=doc_info["total_chunks"],
            file_size_mb=0.0,
            page_count=None
        )
        
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks
    
    - **document_id**: Unique document identifier
    
    Returns success confirmation
    """
    try:
        logger.info(f"Deleting document: {document_id}")
        
        # Check if document exists first
        doc_info = storage_manager.get_document_info(document_id)
        if "error" in doc_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Delete document
        success = storage_manager.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document"
            )
        
        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )