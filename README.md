# Spanish Live Transcription

Real-time Spanish speech-to-text using Faster-Whisper

## Features

- Live audio capture from microphone
- Real-time Spanish transcription
- Low latency (1-2 seconds)
- Timestamped output with confidence scores
- VAD filtering for silence removal

## Installation

```bash
uv pip install faster-whisper sounddevice numpy
```

## Usage

```bash
python spanish_live_transcription.py
```

Press Ctrl+C to stop

## How It Works

**Audio Capture**: Captures 3-second audio chunks at 16kHz
**VAD Processing**: Filters silence (>500ms) → reduces noise
**Transcription**: Faster-Whisper base model transcribes Spanish
**Output**: Timestamped text with confidence scores

## Configuration

Edit `spanish_live_transcription.py`:

```python
transcriber = SpanishLiveTranscriber(
    model_size="base",        # Options: tiny, base, small, medium, large
    device="cpu",             # Options: cpu, cuda
    compute_type="int8",      # Options: int8, float16, float32
    chunk_duration=3.0        # Seconds per chunk (lower = faster, less context)
)
```

**Model Size vs Performance**:
- `tiny`: Fastest, lowest accuracy
- `base`: Balanced (recommended)
- `small`: Better accuracy, 2x slower
- `medium/large`: Highest accuracy, significantly slower

**Chunk Duration**:
- Smaller (1-2s): Lower latency, may miss context
- Larger (4-6s): Better context, higher latency

## Requirements

- Python ≥3.13
- Microphone access
- ~200MB disk (base model)
- ~1-2GB RAM during operation
