"""
FastAPI REST API for transcription service
Designed for Android app integration
"""

import os
import time
import logging
import tempfile
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from transcribe import transcribe_with_speakers, TranscriptionService
from security import (
    limiter,
    verify_api_key,
    SecurityHeadersMiddleware,
    validate_upload,
    validate_language,
    validate_model,
    MAX_FILE_SIZE_BYTES,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Transcription API",
    description="Audio/Video transcription with speaker identification",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SecurityHeadersMiddleware)

_allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "https://voicecast-topaz.vercel.app").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

_service = None
_service_error = None


def get_service():
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
@limiter.limit("30/minute")
async def root(request: Request):
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Transcription API",
        "version": "1.0.0",
        "hf_token_configured": bool(os.environ.get("HF_TOKEN")),
        "models_loaded": _service is not None,
        "endpoints": {
            "POST /transcribe": "Upload file for transcription",
            "GET /ready": "Check if models are loaded",
        },
    }


@app.get("/ready")
@limiter.limit("20/minute")
async def ready_check(request: Request):
    """Check if service is fully initialized and ready"""
    if not os.environ.get("HF_TOKEN"):
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": "HF_TOKEN not configured"},
        )
    try:
        get_service()
        return {"ready": True, "models_loaded": True, "whisper_model": "base"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"ready": False, "reason": str(e)})


@app.post("/transcribe", response_model=TranscriptionResponse)
@limiter.limit("10/minute")
async def transcribe_endpoint(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model: str = Form("base"),
    _: None = Depends(verify_api_key),
):
    """
    Transcribe audio or video file with speaker identification.

    - **file**: Audio/video file (max 100 MB by default)
    - **language**: ISO 639-1 language code (optional; auto-detected if omitted)
    - **model**: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
    """
    if not os.environ.get("HF_TOKEN"):
        raise HTTPException(status_code=503, detail="Service not configured: HF_TOKEN missing")

    safe_filename = validate_upload(file.filename or "")
    validate_language(language)
    validate_model(model)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(safe_filename).suffix)
    temp_path = temp_file.name
    temp_file.close()

    try:
        upload_start = time.time()
        bytes_written = 0
        with open(temp_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
                    )
                f.write(chunk)
        upload_time = time.time() - upload_start

        logger.info("Uploaded %r (%d bytes) in %.2fs", safe_filename, bytes_written, upload_time)

        get_service()

        process_start = time.time()
        result = transcribe_with_speakers(audio_path=temp_path, language=language, whisper_model=model)
        process_time = time.time() - process_start

        result["timing"] = {
            "upload_seconds": round(upload_time, 2),
            "processing_seconds": round(process_time, 2),
            "total_seconds": round(upload_time + process_time, 2),
        }

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Transcription error for %r", safe_filename)
        raise HTTPException(status_code=500, detail="Transcription failed. See server logs.")
    finally:
        file.file.close()
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check for monitoring"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("Starting Transcription API Server")
    print("=" * 60)
    print("\nAPI Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nExample curl command:")
    print("curl -X POST http://localhost:8000/transcribe \\")
    print("  -F 'file=@your_audio.mp3' \\")
    print("  -F 'language=en'\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
