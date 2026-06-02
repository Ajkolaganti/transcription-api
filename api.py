"""
FastAPI REST API for transcription service
Designed for Android app integration
"""

import os
import shutil
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transcribe import transcribe_with_speakers, TranscriptionService
import tempfile
import asyncio

# Create FastAPI app
app = FastAPI(
    title="Transcription API",
    description="Audio/Video transcription with speaker identification",
    version="1.0.0"
)

# Global service instance (lazy loaded)
_service = None
_service_error = None

def get_service():
    """Get or initialize the transcription service"""
    global _service, _service_error
    
    if _service_error:
        raise HTTPException(status_code=500, detail=f"Service initialization failed: {_service_error}")
    
    if _service is None:
        try:
            print("Initializing transcription service...")
            _service = TranscriptionService(whisper_model="base")
            print("Transcription service ready!")
        except Exception as e:
            _service_error = str(e)
            raise HTTPException(status_code=500, detail=f"Failed to initialize service: {str(e)}")
    
    return _service

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class TranscriptionResponse(BaseModel):
    language: str
    duration: float
    segments: list[TranscriptionSegment]


@app.get("/")
async def root():
    """Health check endpoint"""
    hf_token_set = bool(os.environ.get("HF_TOKEN"))
    service_ready = _service is not None
    
    return {
        "status": "online",
        "service": "Transcription API",
        "version": "1.0.0",
        "hf_token_configured": hf_token_set,
        "models_loaded": service_ready,
        "error": _service_error,
        "endpoints": {
            "POST /transcribe": "Upload file for transcription",
            "GET /ready": "Check if models are loaded"
        }
    }


@app.get("/ready")
async def ready_check():
    """Check if service is fully initialized and ready"""
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": "HF_TOKEN environment variable not set. Add it in Railway dashboard."
            }
        )
    
    try:
        service = get_service()
        return {
            "ready": True,
            "models_loaded": True,
            "whisper_model": "base"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "reason": str(e)
            }
        )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model: str = Form("base")
):
    """
    Transcribe audio or video file with speaker identification
    
    Args:
        file: Audio/video file to transcribe
        language: Language code (en, es, fr, etc.) - optional, auto-detects
        model: Whisper model size (tiny, base, small, medium, large-v3)
    
    Returns:
        Transcription with speaker-labeled segments
    """
    # Check HF token first
    if not os.environ.get("HF_TOKEN"):
        raise HTTPException(
            status_code=503,
            detail="Service not configured: HF_TOKEN environment variable missing. Set it in Railway dashboard."
        )
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=Path(file.filename).suffix
    )
    
    try:
        # Write uploaded content
        with temp_file as f:
            shutil.copyfileobj(file.file, f)
        
        # Initialize service if needed (lazy load on first request)
        get_service()
        
        # Process transcription
        result = transcribe_with_speakers(
            audio_path=temp_file.name,
            language=language,
            whisper_model=model
        )
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    
    finally:
        # Cleanup
        file.file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Starting Transcription API Server")
    print("="*60)
    print("\nAPI Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nExample curl command:")
    print("curl -X POST http://localhost:8000/transcribe \\")
    print("  -F 'file=@your_audio.mp3' \\")
    print("  -F 'language=en'\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
