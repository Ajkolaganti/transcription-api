#!/bin/bash
# Test script for transcription API

API_URL="https://web-production-0f5d1.up.railway.app"

echo "================================"
echo "Testing Transcription API"
echo "================================"
echo ""

# Test 1: Root endpoint
echo "1. Testing root endpoint..."
curl -s "$API_URL/" | python3 -m json.tool
echo ""

# Test 2: Ready check
echo "2. Checking if service is ready..."
curl -s "$API_URL/ready" | python3 -m json.tool
echo ""

# Test 3: Health check
echo "3. Testing health endpoint..."
curl -s "$API_URL/health" | python3 -m json.tool
echo ""

echo "================================"
echo "API Base URL: $API_URL"
echo "================================"
echo ""
echo "To test transcription with an audio file:"
echo "  curl -X POST \"$API_URL/transcribe\" \\"
echo "    -F 'file=@your_audio.mp3' \\"
echo "    -F 'language=en'"
echo ""
