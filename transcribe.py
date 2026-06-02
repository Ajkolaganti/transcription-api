"""
Audio/Video Transcription with Speaker Identification
Uses faster-whisper for transcription and pyannote.audio for diarization
"""

import os
from typing import Optional, List, Dict
from pathlib import Path
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
import tempfile


class TranscriptionService:
    """Service for transcribing audio/video with speaker identification"""
    
    def __init__(
        self, 
        whisper_model: str = "base",
        device: str = "auto",
        compute_type: str = "auto"
    ):
        """
        Initialize transcription service
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to run on (cuda, cpu, auto)
            compute_type: Compute type (int8, float16, float32, auto)
        """
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Auto-detect compute type based on device
        if compute_type == "auto":
            if device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"
        
        print(f"Initializing Whisper model: {whisper_model} on {device} ({compute_type})")
        self.whisper = WhisperModel(whisper_model, device=device, compute_type=compute_type)
        
        # Load diarization pipeline
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable not set. "
                "Get token from https://huggingface.co/settings/tokens"
            )
        
        print("Loading speaker diarization model...")
        self.diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        
        if device == "cuda":
            self.diarization.to(torch.device("cuda"))
    
    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file (wav)
        """
        video_path = Path(video_path)
        
        # If already audio, return as-is
        if video_path.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
            return str(video_path)
        
        # Extract audio using pydub
        print(f"Extracting audio from {video_path.name}...")
        audio = AudioSegment.from_file(str(video_path))
        
        # Save as WAV for processing
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio.export(temp_wav.name, format='wav')
        
        return temp_wav.name
    
    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio without speaker identification
        
        Args:
            audio_path: Path to audio/video file
            language: Language code (en, es, fr, etc.) or None for auto-detect
            
        Returns:
            Dict with transcription segments
        """
        segments, info = self.whisper.transcribe(
            audio_path,
            language=language,
            vad_filter=True,  # Voice activity detection
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
        
        return result
    
    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Identify speakers in audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of speaker segments with timestamps
        """
        print("Running speaker diarization...")
        diarization = self.diarization(audio_path)
        
        speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        return speakers
    
    def merge_transcription_and_speakers(
        self, 
        transcription: Dict, 
        speakers: List[Dict]
    ) -> Dict:
        """
        Merge transcription segments with speaker labels
        
        Args:
            transcription: Output from transcribe()
            speakers: Output from diarize()
            
        Returns:
            Combined result with speaker labels
        """
        result = {
            "language": transcription["language"],
            "duration": transcription["duration"],
            "segments": []
        }
        
        for segment in transcription["segments"]:
            # Find overlapping speaker
            segment_mid = (segment["start"] + segment["end"]) / 2
            speaker = "UNKNOWN"
            
            for spk in speakers:
                if spk["start"] <= segment_mid <= spk["end"]:
                    speaker = spk["speaker"]
                    break
            
            result["segments"].append({
                "start": segment["start"],
                "end": segment["end"],
                "speaker": speaker,
                "text": segment["text"]
            })
        
        return result


def transcribe_with_speakers(
    audio_path: str,
    language: Optional[str] = None,
    whisper_model: str = "base"
) -> Dict:
    """
    Convenience function: transcribe audio/video with speaker identification
    
    Args:
        audio_path: Path to audio or video file
        language: Language code (optional, auto-detects)
        whisper_model: Whisper model size (tiny, base, small, medium, large-v3)
        
    Returns:
        Dict with segments containing start, end, speaker, and text
    """
    service = TranscriptionService(whisper_model=whisper_model)
    
    # Extract audio if video
    audio_file = service.extract_audio(audio_path)
    
    # Transcribe
    print("Transcribing audio...")
    transcription = service.transcribe(audio_file, language)
    
    # Identify speakers
    speakers = service.diarize(audio_file)
    
    # Merge results
    result = service.merge_transcription_and_speakers(transcription, speakers)
    
    # Cleanup temp file if created
    if audio_file != audio_path:
        os.unlink(audio_file)
    
    return result


if __name__ == "__main__":
    # Test with a sample file
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_or_video_file> [language]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"\n{'='*60}")
    print(f"Transcribing: {file_path}")
    print(f"{'='*60}\n")
    
    result = transcribe_with_speakers(file_path, language)
    
    print(f"\n{'='*60}")
    print(f"RESULTS (Language: {result['language']}, Duration: {result['duration']:.1f}s)")
    print(f"{'='*60}\n")
    
    for segment in result["segments"]:
        timestamp = f"[{segment['start']:.1f}s - {segment['end']:.1f}s]"
        print(f"{timestamp} {segment['speaker']}: {segment['text']}")
