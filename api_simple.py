"""
Simple transcription API without speaker identification
Use this version if you don't have pyannote model access yet
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
from faster_whisper import WhisperModel
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

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
    title="Simple Transcription API",
    description="Audio/Video transcription (no speaker identification)",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SecurityHeadersMiddleware)

_allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (base)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Whisper model ready!")
    return _whisper_model


class TranscriptionSegment(BaseModel):
    start: float
    end: float
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
        "service": "Simple Transcription API",
        "version": "1.0.0",
        "note": "No speaker identification - just transcription",
        "endpoints": {"POST /transcribe": "Upload file for transcription"},
    }


@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Health check for monitoring"""
    return {"status": "healthy"}


@app.get("/ready")
@limiter.limit("20/minute")
async def ready_check(request: Request):
    """Check if service is ready"""
    try:
        get_whisper_model()
        return {"ready": True, "whisper_model": "base"}
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
    Transcribe audio or video file (no speaker identification).

    - **file**: Audio/video file (max 100 MB by default)
    - **language**: ISO 639-1 language code (optional; auto-detected if omitted)
    - **model**: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
    """
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

        whisper = get_whisper_model()

        process_start = time.time()
        segments_iter, info = whisper.transcribe(
            temp_path, language=language, vad_filter=True, word_timestamps=False
        )

        result = {
            "language": info.language,
            "duration": info.duration,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in segments_iter
            ],
        }
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


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("Starting Simple Transcription API (No Speaker ID)")
    print("=" * 60)
    print("\nAPI Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
