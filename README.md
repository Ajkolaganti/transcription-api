# Audio/Video Transcription API

Fast transcription service with speaker diarization (speaker identification).

## Features
- 🎤 Audio & video file support (mp3, wav, mp4, etc.)
- 🗣️ Speaker identification/diarization
- ⚡ Fast transcription using Whisper
- 📝 Sentence-segmented output
- 🔌 REST API ready for Android integration

## Tech Stack
- **Whisper** (faster-whisper): Fast, accurate transcription
- **Pyannote.audio**: Speaker diarization
- **FastAPI**: REST API framework
- **FFmpeg**: Audio/video processing

## Installation

### 1. System Dependencies
```bash
sudo apt update
sudo apt install -y ffmpeg python3-pip python3-venv
```

### 2. Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Hugging Face Token (for pyannote models)
You'll need a HuggingFace token with access to pyannote models:
1. Create account at https://huggingface.co
2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Create token at https://huggingface.co/settings/tokens
4. Set environment variable:
```bash
export HF_TOKEN="your_token_here"
```

## Usage

### As Python Module
```python
from transcribe import transcribe_with_speakers

result = transcribe_with_speakers(
    audio_path="meeting.mp4",
    language="en"  # optional, auto-detects if not provided
)

for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}")
```

### As REST API
```bash
# Start server
python api.py

# Upload file for transcription
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@meeting.mp4" \
  -F "language=en"
```

## API Endpoints

### POST /transcribe
Upload audio/video file for transcription.

**Parameters:**
- `file`: Audio/video file (multipart/form-data)
- `language` (optional): Language code (en, es, fr, etc.)

**Response:**
```json
{
  "duration": 125.5,
  "language": "en",
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker": "SPEAKER_00",
      "text": "Hello everyone, welcome to the meeting."
    }
  ]
}
```

## Android Integration Example

```kotlin
// Retrofit API interface
interface TranscriptionAPI {
    @Multipart
    @POST("/transcribe")
    suspend fun transcribe(
        @Part file: MultipartBody.Part,
        @Part("language") language: RequestBody?
    ): TranscriptionResponse
}

// Usage
val file = File(audioPath)
val requestFile = file.asRequestBody("audio/*".toMediaType())
val body = MultipartBody.Part.createFormData("file", file.name, requestFile)
val response = api.transcribe(body, null)
```

## Performance

- **Whisper Model Sizes:**
  - `tiny`: ~1GB RAM, 10x faster than realtime
  - `base`: ~1GB RAM, 7x faster than realtime  
  - `small`: ~2GB RAM, 4x faster than realtime (recommended)
  - `medium`: ~5GB RAM, 2x faster than realtime
  - `large-v3`: ~10GB RAM, realtime speed (most accurate)

Current configuration uses `base` model for balance of speed/accuracy.

## Project Structure
```
transcription-api/
├── transcribe.py       # Core transcription logic
├── api.py             # FastAPI REST server
├── requirements.txt   # Python dependencies
├── test_transcribe.py # Tests
└── uploads/           # Temporary file storage
```

## Deployment to Railway

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Transcription API"
gh repo create transcription-api --public --source=. --push
```

### 2. Deploy to Railway
1. Visit https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `transcription-api` repository
4. Railway will auto-detect Python and deploy

### 3. Set Environment Variable
In Railway dashboard:
- Go to your project → Variables
- Add: `HF_TOKEN` = your HuggingFace token

**IMPORTANT:** Without HF_TOKEN, the API will return 503 errors!

### 4. Wait for Model Download (First Deploy Only)
The first deployment downloads ~2GB of models:
- Whisper model (~500MB)
- Pyannote diarization model (~1.5GB)

This happens on first request, taking 2-3 minutes. Subsequent requests are instant.

**Check readiness:**
```bash
curl https://your-app.up.railway.app/ready
```

### 5. Get Your API URL
Railway provides a URL like: `https://transcription-api-production.up.railway.app`

### Notes
- Railway provides 512MB RAM on free tier (use `tiny` or `base` Whisper model)
- For production with larger models, upgrade to paid plan with more RAM
- Railway auto-installs ffmpeg via nixpacks.toml

## License
MIT
