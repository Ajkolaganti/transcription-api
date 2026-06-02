#!/bin/bash
# Performance test for transcription API

API_URL="https://web-production-0f5d1.up.railway.app"

echo "Testing transcription API performance..."
echo ""

# Download a small test audio file (10 seconds)
if [ ! -f "preamble10.wav" ]; then
    echo "Downloading test audio (10 seconds)..."
    wget -q https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav
fi

FILE_SIZE=$(ls -lh preamble10.wav | awk '{print $5}')
echo "File size: $FILE_SIZE"
echo ""

echo "Uploading and transcribing..."
time curl -X POST "$API_URL/transcribe" \
  -F "file=@preamble10.wav" \
  -F "language=en" \
  -F "model=tiny" \
  2>/dev/null | python3 -m json.tool

echo ""
echo "If this took more than 30 seconds for a 10-second audio file,"
echo "the server is underpowered or using a model that's too large."
