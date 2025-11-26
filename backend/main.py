"""
FastAPI Main Application - Smart Evidence Pipeline

Simplified backend focused on the smart evidence pipeline workflow.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

# Import routers for Smart Pipeline
from backend.api.endpoints import video_upload
from backend.api.endpoints.smart_pipeline_router import router as smart_pipeline_router
from backend.api.endpoints.analytics import router as analytics_router

# Database initialization
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.schema import init_db
from database.operations import db_manager

# Configuration
from config.settings import settings

# Setup master logger for unified error tracking
from master_logger import init_master_logger, get_master_logger

# Initialize master logger
master_logger_instance = init_master_logger(log_dir="logs", level="DEBUG")
logger = get_master_logger()

# Initialize FastAPI app
app = FastAPI(
    title="Smart Evidence Pipeline API",
    description="Audio-first, genre-aware video analysis with multi-model AI validation",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CORS Configuration
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # Alternative port
        "http://localhost:3002",  # Alternative port
        "http://localhost:3003",  # Current frontend port
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Alternative frontend port
        # Add your production domains here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Startup & Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    master_logger_instance.log_startup("Smart Evidence Pipeline API", "3.0.0")

    try:
        # Initialize database tables
        logger.info("Initializing database...")
        db_manager.create_tables()
        logger.info("✅ Database initialized")

        # Add any other startup tasks here
        # - Load ML models
        # - Initialize cache
        # - Start background tasks

        logger.info("✅ API startup complete")
        logger.info("="*80)

    except Exception as e:
        master_logger_instance.log_error_section(
            "API Startup Failed",
            e,
            context={
                "service": "Video Q&A Generation API",
                "database_url": settings.database_url,
            }
        )
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")
    
    # Add cleanup tasks here
    # - Close database connections
    # - Stop background tasks
    # - Save state
    
    logger.info("✅ API shutdown complete")

# ============================================================================
# Include Routers - Smart Pipeline Only
# ============================================================================

# Video Upload Router (for uploading videos to process)
app.include_router(
    video_upload.router,
    prefix="/api/upload",
    tags=["Video Upload"]
)

# Smart Pipeline Router (main processing pipeline)
app.include_router(
    smart_pipeline_router,
    tags=["Smart Pipeline"]
)

# Analytics Router (for analytics page)
app.include_router(
    analytics_router,
    prefix="/api/analytics",
    tags=["Analytics"]
)

# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Smart Evidence Pipeline API",
        "version": "3.0.0",
        "status": "running",
        "description": "Audio-first, genre-aware video analysis with 91.6% cost savings",
        "features": [
            "Audio transcription with Whisper",
            "GPT-4 powered genre detection",
            "Smart frame extraction (key moments + bulk)",
            "Multi-model question generation",
            "Cross-model answer validation",
            "Real-time progress tracking",
            "Gemini adversarial testing",
            "Excel export"
        ],
        "docs": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        from database.operations import VideoOperations
        
        # Try a simple query
        videos = VideoOperations.get_videos_by_pipeline_stage('completed')
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/api/info")
async def api_info():
    """API information and available endpoints"""
    return {
        "api_version": "3.0.0",
        "endpoints": {
            "upload": [
                "POST /api/upload/video - Upload a new video for processing"
            ],
            "smart_pipeline": [
                "POST /api/smart-pipeline/run - Run complete smart pipeline on a video",
                "GET /api/smart-pipeline/status/{video_id} - Get pipeline processing status",
                "GET /api/smart-pipeline/audio/{video_id} - Get audio analysis results",
                "GET /api/smart-pipeline/genre/{video_id} - Get genre detection results",
                "GET /api/smart-pipeline/frames/{video_id} - Get frame extraction results",
                "GET /api/smart-pipeline/questions/{video_id} - Get generated questions",
                "GET /api/smart-pipeline/validation/{video_id} - Get validation results",
                "GET /api/smart-pipeline/transcript/{video_id} - Get full transcript with timestamps",
                "DELETE /api/smart-pipeline/{video_id} - Delete all pipeline results for a video"
            ]
        },
        "pipeline_phases": [
            "1. Audio Analysis - Whisper transcription with speaker diarization",
            "2. Genre Detection - GPT-4 powered genre classification and key moment identification",
            "3. Frame Planning - Smart frame extraction strategy (key moments + bulk)",
            "4. Frame Extraction - Extract frames based on strategy",
            "5. Question Generation - Multi-model question generation (templates + AI)",
            "6. Validation - Cross-model answer validation and consensus"
        ],
        "features": [
            "91.6% cost savings vs traditional approach",
            "Audio-first analysis",
            "Genre-aware processing",
            "Smart frame extraction",
            "Multi-model AI validation"
        ]
    }

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============================================================================
# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )