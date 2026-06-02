# HuggingFace Token Setup

The transcription API requires access to multiple gated models from pyannote.audio.

## Steps to Configure Your HF Token:

### 1. Create HuggingFace Account
Visit: https://huggingface.co/join

### 2. Accept Terms for ALL Required Models

You must accept the terms for each of these models:

1. **Speaker Diarization (main model)**
   - Visit: https://huggingface.co/pyannote/speaker-diarization-3.1
   - Click "Agree and access repository"

2. **Segmentation Model**
   - Visit: https://huggingface.co/pyannote/segmentation-3.0
   - Click "Agree and access repository"

3. **Speaker Embedding Model**
   - Visit: https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
   - Click "Agree and access repository"

### 3. Create Access Token

1. Visit: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `transcription-api` (or any name)
4. Type: **Read** (default is fine)
5. Click "Generate"
6. **Copy the token** (starts with `hf_...`)

### 4. Add Token to Railway

1. Go to your Railway project dashboard
2. Click on your service → **Variables** tab
3. Click "New Variable"
4. Variable name: `HF_TOKEN`
5. Value: Paste your token (e.g., `hf_xxxxxxxxxxxxxxxxxxxxx`)
6. Click "Add"

Railway will automatically redeploy with the new token.

### 5. Verify It Works

After Railway redeploys, test:

```bash
curl https://web-production-0f5d1.up.railway.app/ready
```

Should return:
```json
{
  "ready": true,
  "models_loaded": true,
  "whisper_model": "base"
}
```

## Troubleshooting

**If you still get errors:**
- Make sure you clicked "Agree" on ALL three model pages above
- Wait 5 minutes after accepting terms (HuggingFace needs to propagate permissions)
- Regenerate your token if it's old
- Make sure the token has "Read" permission

**Common error messages:**
- `Could not download 'pyannote/segmentation-3.0'` → Accept terms at link #2 above
- `Could not download 'pyannote/wespeaker-voxceleb-resnet34-LM'` → Accept terms at link #3 above
- `401 Unauthorized` → Token is invalid or missing permissions
