#!/usr/bin/env python3
"""
Real-time Spanish Speech-to-Text using Faster-Whisper
Captures live audio and transcribes with low latency
"""

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from datetime import datetime
import sys
import queue

class SpanishLiveTranscriber:
    def __init__(
        self,
        model_size="base",
        device="cpu",
        compute_type="int8",
        sample_rate=16000,
        chunk_duration=3.0,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)

        print(f"Loading Whisper model: {model_size}")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        print("Model loaded successfully\n")

        self.audio_queue = queue.Queue()
        self.is_running = False

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def transcribe_chunk(self, audio_chunk):
        audio_float32 = audio_chunk.flatten().astype(np.float32)

        segments, info = self.model.transcribe(
            audio_float32,
            language="es",
            beam_size=1,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500
            },
            condition_on_previous_text=False
        )

        segments_list = list(segments)

        if segments_list:
            timestamp = datetime.now().strftime("%H:%M:%S")
            for segment in segments_list:
                text = segment.text.strip()
                if text:
                    confidence = 1.0 - segment.no_speech_prob
                    print(f"[{timestamp}] ({confidence:.2f}) {text}")
                    sys.stdout.flush()

    def start(self):
        self.is_running = True

        print("🎤 Real-time Spanish Transcription Active")
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Chunk Duration: {self.chunk_duration}s")
        print("Press Ctrl+C to stop\n")
        print("-" * 60)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_samples,
                callback=self.audio_callback
            ):
                audio_buffer = []

                while self.is_running:
                    try:
                        chunk = self.audio_queue.get(timeout=0.5)
                        audio_buffer.append(chunk)

                        current_samples = sum(len(c) for c in audio_buffer)

                        if current_samples >= self.chunk_samples:
                            audio_data = np.concatenate(audio_buffer)
                            audio_buffer = []

                            self.transcribe_chunk(audio_data)

                    except queue.Empty:
                        continue

        except KeyboardInterrupt:
            print("\n" + "-" * 60)
            print("Transcription stopped by user")
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
        finally:
            self.is_running = False

def main():
    transcriber = SpanishLiveTranscriber(
        model_size="base",
        device="cpu",
        compute_type="int8",
        chunk_duration=3.0
    )

    transcriber.start()

if __name__ == "__main__":
    main()
