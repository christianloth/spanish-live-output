# Spanish Live Transcription

Real-time Spanish speech-to-text using OpenAI Whisper (local)

## Installation

```bash
uv sync
```

## Usage

```bash
uv run python spanish_live_transcription.py
```

## How It Works

Captures 3-second audio chunks from your microphone and transcribes them using OpenAI's Whisper model running locally on your CPU. No internet required after initial model download.

**Models:**
- `turbo` - 1.5GB, fastest, best quality (default)
- `base` - 74MB, very fast
- `small` - 244MB, fast
- `medium` - 769MB, slow
- `large-v3` - 1.5GB, slowest

**Chunk Duration:**
- Lower (2s) = faster response
- Higher (5s) = better context

