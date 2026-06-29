"""
Fast transcription API with speaker identification using SpeechBrain
No HuggingFace token required!
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
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

from security import (
    limiter,
    verify_api_key,
    SecurityHeadersMiddleware,
    validate_upload,
    validate_language,
    MAX_FILE_SIZE_BYTES,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Transcription API with Speaker ID",
    description="Audio/Video transcription with speaker diarization (SpeechBrain)",
    version="2.0.0",
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

_whisper_model = None
_speaker_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (base)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Whisper model ready!")
    return _whisper_model


def get_speaker_model():
    global _speaker_model
    if _speaker_model is None:
        print("Loading SpeechBrain speaker model...")
        try:
            _speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )
            print("SpeechBrain model ready!")
        except Exception as e:
            print(f"Warning: Could not load speaker model: {e}")
            _speaker_model = None
    return _speaker_model


def cluster_speaker_embeddings(embeddings, threshold=0.7):
    """Simple cosine-similarity speaker clustering."""
    if not embeddings:
        return []

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    emb_array = np.array(embeddings)
    speaker_labels = []
    speaker_centroids = []

    for emb in emb_array:
        assigned = False
        for idx, centroid in enumerate(speaker_centroids):
            if cosine_similarity([emb], [centroid])[0][0] > threshold:
                speaker_labels.append(f"SPEAKER_{idx}")
                assigned = True
                break
        if not assigned:
            speaker_centroids.append(emb)
            speaker_labels.append(f"SPEAKER_{len(speaker_centroids) - 1}")

    return speaker_labels


def extract_speaker_embeddings(audio_path, segments):
    speaker_model = get_speaker_model()
    if speaker_model is None:
        return None
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        embeddings = []
        for segment in segments:
            start_sample = int(segment["start"] * sample_rate)
            end_sample = int(segment["end"] * sample_rate)
            seg_audio = waveform[:, start_sample:end_sample]
            if seg_audio.shape[1] < sample_rate * 0.5:
                embeddings.append(None)
                continue
            with torch.no_grad():
                emb = speaker_model.encode_batch(seg_audio)
                embeddings.append(emb.squeeze().cpu().numpy())
        return embeddings
    except Exception as e:
        logger.warning("Speaker embedding extraction failed: %s", e)
        return None


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


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
        "service": "Transcription API with Speaker Identification",
        "version": "2.0.0",
        "features": {
            "transcription": "faster-whisper (base model)",
            "speaker_diarization": "SpeechBrain ECAPA-TDNN" if _speaker_model else "not yet loaded",
        },
        "endpoints": {
            "GET /": "Service info",
            "GET /health": "Health check",
            "GET /ready": "Readiness check",
            "POST /transcribe": "Upload file for transcription + speaker ID",
        },
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
        get_speaker_model()
        return {
            "ready": True,
            "whisper_model": "base",
            "speaker_model": "speechbrain-ecapa" if _speaker_model else "disabled",
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"ready": False, "reason": str(e)})


@app.post("/transcribe", response_model=TranscriptionResponse)
@limiter.limit("10/minute")
async def transcribe_endpoint(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    enable_speakers: bool = Form(True),
    translate_to_english: bool = Form(False),
    _: None = Depends(verify_api_key),
):
    """
    Transcribe audio or video file with speaker identification.

    - **file**: Audio/video file (max 100 MB by default)
    - **language**: ISO 639-1 language code (optional; auto-detected if omitted)
    - **enable_speakers**: Include speaker labels (default: true)
    - **translate_to_english**: Translate output to English (default: false)
    """
    safe_filename = validate_upload(file.filename or "")
    validate_language(language)

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

        transcribe_start = time.time()
        task = "translate" if translate_to_english else "transcribe"
        segments_iter, info = whisper.transcribe(
            temp_path, language=language, task=task, vad_filter=True, word_timestamps=False
        )

        segments = [
            {"start": s.start, "end": s.end, "text": s.text.strip(), "speaker": None}
            for s in segments_iter
        ]
        transcribe_time = time.time() - transcribe_start

        speaker_time = 0.0
        if enable_speakers and segments:
            speaker_start = time.time()
            embeddings = extract_speaker_embeddings(temp_path, segments)
            if embeddings:
                valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
                valid_embs = [e for e in embeddings if e is not None]
                if valid_embs:
                    labels = cluster_speaker_embeddings(valid_embs)
                    for pos, idx in enumerate(valid_indices):
                        segments[idx]["speaker"] = labels[pos]
            speaker_time = time.time() - speaker_start

        result = {
            "language": info.language,
            "duration": info.duration,
            "segments": segments,
            "timing": {
                "upload_seconds": round(upload_time, 2),
                "transcription_seconds": round(transcribe_time, 2),
                "speaker_seconds": round(speaker_time, 2),
                "total_seconds": round(upload_time + transcribe_time + speaker_time, 2),
            },
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

    print("\n" + "=" * 70)
    print("Starting Transcription API with Speaker Identification (SpeechBrain)")
    print("=" * 70)
    print("\nAPI Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
