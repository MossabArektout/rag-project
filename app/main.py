from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config.settings import settings
from config.logger import setup_logging
from app.routers import documents, qa
from app.storage_manager import DocumentStorageManager

# Initialize logging
logger = setup_logging()

# Initialize storage manager for health checks
storage_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown)"""
    global storage_manager
    
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Embedding model: {settings.embedding_model}")
    logger.info(f"LLM model: {settings.llm_model}")
    
    # Initialize storage manager
    try:
        storage_manager = DocumentStorageManager()
        logger.success("✓ Storage manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize storage manager: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG-based Q&A system for company knowledge bases",
    debug=settings.debug,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "endpoints": {
            "upload_document": "/api/documents/upload",
            "list_documents": "/api/documents",
            "ask_question": "/api/qa/ask",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint with database statistics"""
    try:
        if storage_manager:
            stats = storage_manager.get_database_stats()
            
            return {
                "status": "healthy",
                "version": settings.app_version,
                "database_status": "connected",
                "total_documents": stats["total_documents"],
                "total_chunks": stats["total_chunks"],
                "embedding_model": stats["embedding_model"]["model_name"],
                "embedding_dimension": stats["embedding_model"]["embedding_dimension"]
            }
        else:
            return {
                "status": "degraded",
                "version": settings.app_version,
                "database_status": "not_initialized",
                "message": "Storage manager not initialized"
            }
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "version": settings.app_version,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )