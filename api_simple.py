"""
Simple transcription API without speaker identification
Use this version if you don't have pyannote model access yet
"""

import os
import shutil
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
import tempfile
import time

# Create FastAPI app
app = FastAPI(
    title="Simple Transcription API",
    description="Audio/Video transcription (no speaker identification)",
    version="1.0.0"
)

# Global whisper model
_whisper_model = None

def get_whisper_model():
    """Get or initialize Whisper model"""
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
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Simple Transcription API",
        "version": "1.0.0",
        "note": "No speaker identification - just transcription",
        "endpoints": {
            "POST /transcribe": "Upload file for transcription"
        }
    }


@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {"status": "healthy"}


@app.get("/ready")
async def ready_check():
    """Check if service is ready"""
    try:
        get_whisper_model()
        return {"ready": True, "whisper_model": "base"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": str(e)}
        )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model: str = Form("base")
):
    """
    Transcribe audio or video file (no speaker identification)
    
    Args:
        file: Audio/video file to transcribe
        language: Language code (en, es, fr, etc.) - optional, auto-detects
        model: Whisper model size (tiny, base, small) - ignored for now, uses base
    
    Returns:
        Transcription segments with timestamps
    """
    start_time = time.time()
    
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
        upload_start = time.time()
        with temp_file as f:
            shutil.copyfileobj(file.file, f)
        upload_time = time.time() - upload_start
        
        print(f"Upload took {upload_time:.2f}s for {file.filename}")
        
        # Get Whisper model
        whisper = get_whisper_model()
        
        # Transcribe
        process_start = time.time()
        segments, info = whisper.transcribe(
            temp_file.name,
            language=language,
            vad_filter=True,
            word_timestamps=False
        )
        
        result = {
            "language": info.language,
            "duration": info.duration,
            "segments": []
        }
        
        for segment in segments:
            result["segments"].append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        process_time = time.time() - process_start
        total_time = time.time() - start_time
        
        print(f"Processing took {process_time:.2f}s, total: {total_time:.2f}s")
        
        # Add timing info
        result["timing"] = {
            "upload_seconds": round(upload_time, 2),
            "processing_seconds": round(process_time, 2),
            "total_seconds": round(total_time, 2)
        }
        
        return JSONResponse(content=result)
    
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


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Starting Simple Transcription API (No Speaker ID)")
    print("="*60)
    print("\nAPI Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
