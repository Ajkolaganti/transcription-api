"""
Fast transcription API with speaker identification using SpeechBrain
No HuggingFace token required!
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
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# Create FastAPI app
app = FastAPI(
    title="Transcription API with Speaker ID",
    description="Audio/Video transcription with speaker diarization (SpeechBrain)",
    version="2.0.0"
)

# Global models
_whisper_model = None
_speaker_model = None

def get_whisper_model():
    """Get or initialize Whisper model"""
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (base)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("✓ Whisper model ready!")
    return _whisper_model


def get_speaker_model():
    """Get or initialize SpeechBrain speaker recognition model"""
    global _speaker_model
    if _speaker_model is None:
        print("Loading SpeechBrain speaker model...")
        try:
            _speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"}
            )
            print("✓ SpeechBrain model ready!")
        except Exception as e:
            print(f"Warning: Could not load speaker model: {e}")
            _speaker_model = None
    return _speaker_model


def cluster_speaker_embeddings(embeddings, threshold=0.7):
    """
    Simple speaker clustering using cosine similarity
    Returns speaker labels for each segment
    """
    if not embeddings:
        return []
    
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Convert to numpy array
    emb_array = np.array(embeddings)
    
    # Compute similarity matrix
    similarity = cosine_similarity(emb_array)
    
    # Simple clustering: assign to existing speaker if similarity > threshold
    speaker_labels = []
    speakers = []  # List of speaker embeddings
    
    for i, emb in enumerate(emb_array):
        assigned = False
        
        # Check similarity with existing speakers
        for speaker_idx, speaker_emb in enumerate(speakers):
            sim = cosine_similarity([emb], [speaker_emb])[0][0]
            if sim > threshold:
                speaker_labels.append(f"SPEAKER_{speaker_idx}")
                assigned = True
                break
        
        # New speaker
        if not assigned:
            speaker_idx = len(speakers)
            speakers.append(emb)
            speaker_labels.append(f"SPEAKER_{speaker_idx}")
    
    return speaker_labels


def extract_speaker_embeddings(audio_path, segments):
    """
    Extract speaker embeddings for each segment
    """
    speaker_model = get_speaker_model()
    if speaker_model is None:
        return None
    
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        embeddings = []
        
        for segment in segments:
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            
            # Extract segment audio
            segment_audio = waveform[:, start_sample:end_sample]
            
            # Skip very short segments (< 0.5 seconds)
            if segment_audio.shape[1] < sample_rate * 0.5:
                embeddings.append(None)
                continue
            
            # Get embedding
            with torch.no_grad():
                embedding = speaker_model.encode_batch(segment_audio)
                embeddings.append(embedding.squeeze().cpu().numpy())
        
        return embeddings
    
    except Exception as e:
        print(f"Warning: Speaker embedding extraction failed: {e}")
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
async def root():
    """Health check endpoint"""
    speaker_available = get_speaker_model() is not None
    
    return {
        "status": "online",
        "service": "Transcription API with Speaker Identification",
        "version": "2.0.0",
        "features": {
            "transcription": "faster-whisper (base model)",
            "speaker_diarization": "SpeechBrain ECAPA-TDNN" if speaker_available else "unavailable",
            "authentication": "none required"
        },
        "endpoints": {
            "GET /": "Service info",
            "GET /health": "Health check",
            "GET /ready": "Readiness check",
            "POST /transcribe": "Upload file for transcription + speaker ID"
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
        whisper = get_whisper_model()
        speaker = get_speaker_model()
        
        return {
            "ready": True,
            "whisper_model": "base",
            "speaker_model": "speechbrain-ecapa" if speaker else "disabled"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "reason": str(e)}
        )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    enable_speakers: bool = Form(True),
    translate_to_english: bool = Form(False)
):
    """
    Transcribe audio or video file with speaker identification
    
    Args:
        file: Audio/video file to transcribe
        language: Language code (en, es, fr, etc.) - optional, auto-detects
        enable_speakers: Enable speaker diarization (default: true)
        translate_to_english: If true, translate to English (default: false)
    
    Returns:
        Transcription segments with timestamps and speaker labels
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
        transcribe_start = time.time()
        
        # Choose task: transcribe (original language) or translate (to English)
        task = "translate" if translate_to_english else "transcribe"
        
        segments_iter, info = whisper.transcribe(
            temp_file.name,
            language=language,
            task=task,
            vad_filter=True,
            word_timestamps=False
        )
        
        # Collect segments
        segments = []
        for segment in segments_iter:
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "speaker": None
            })
        
        transcribe_time = time.time() - transcribe_start
        print(f"Transcription took {transcribe_time:.2f}s")
        
        # Speaker diarization
        speaker_time = 0
        if enable_speakers and segments:
            speaker_start = time.time()
            print("Extracting speaker embeddings...")
            
            embeddings = extract_speaker_embeddings(temp_file.name, segments)
            
            if embeddings:
                # Filter out None embeddings
                valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
                valid_embeddings = [emb for emb in embeddings if emb is not None]
                
                if valid_embeddings:
                    speaker_labels = cluster_speaker_embeddings(valid_embeddings)
                    
                    # Assign labels back to segments
                    label_idx = 0
                    for i in range(len(segments)):
                        if i in valid_indices:
                            segments[i]["speaker"] = speaker_labels[label_idx]
                            label_idx += 1
            
            speaker_time = time.time() - speaker_start
            print(f"Speaker diarization took {speaker_time:.2f}s")
        
        result = {
            "language": info.language,
            "duration": info.duration,
            "segments": segments,
            "timing": {
                "upload_seconds": round(upload_time, 2),
                "transcription_seconds": round(transcribe_time, 2),
                "speaker_seconds": round(speaker_time, 2),
                "total_seconds": round(time.time() - start_time, 2)
            }
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    
    print("\n" + "="*70)
    print("Starting Transcription API with Speaker Identification (SpeechBrain)")
    print("="*70)
    print("\nAPI Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
