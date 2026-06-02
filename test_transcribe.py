"""
Test script for transcription service
"""

import os
from transcribe import transcribe_with_speakers


def test_with_sample():
    """Test transcription with a sample audio file"""
    
    # You can generate a test audio file or download one
    # For now, this is a placeholder
    
    print("To test the transcription service:")
    print("\n1. Get a HuggingFace token:")
    print("   - Visit https://huggingface.co/settings/tokens")
    print("   - Create a new token")
    print("   - Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("\n2. Set the token:")
    print("   export HF_TOKEN='your_token_here'")
    print("\n3. Run with a test file:")
    print("   python transcribe.py your_audio.mp3")
    print("\nOr test the API:")
    print("   python api.py")
    print("   # In another terminal:")
    print("   curl -X POST http://localhost:8000/transcribe \\")
    print("     -F 'file=@your_audio.mp3' \\")
    print("     -F 'language=en'")


if __name__ == "__main__":
    test_with_sample()
