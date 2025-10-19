#!/usr/bin/env python3
"""
Real-time Spanish Speech-to-Text using OpenAI Whisper (Local)
Captures live audio and transcribes with low latency
"""

import sounddevice as sd
import numpy as np
import whisper
from datetime import datetime
import sys
import queue
import threading

class SpanishLiveTranscriber:
    def __init__(
        self,
        model_size="turbo",
        sample_rate=16000,
        chunk_duration=3.0,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)

        print(f"Loading OpenAI Whisper model: {model_size}")
        print("This may take a moment on first run (downloading model)...")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully\n")

        self.audio_queue = queue.Queue()
        self.is_running = False

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def transcribe_chunk(self, audio_chunk):
        audio_float32 = audio_chunk.flatten().astype(np.float32)

        try:
            result = self.model.transcribe(
                audio_float32,
                language="es",
                temperature=0.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                fp16=False
            )

            if result["text"].strip():
                timestamp = datetime.now().strftime("%H:%M:%S")
                text = result["text"].strip()
                print(f"[{timestamp}] {text}")
                sys.stdout.flush()

        except Exception as e:
            print(f"Transcription error: {e}", file=sys.stderr)

    def start(self):
        self.is_running = True

        print("Real-time Spanish Transcription Active")
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

                            transcribe_thread = threading.Thread(
                                target=self.transcribe_chunk,
                                args=(audio_data,),
                                daemon=True
                            )
                            transcribe_thread.start()

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
        model_size="turbo",
        chunk_duration=3.0
    )

    transcriber.start()

if __name__ == "__main__":
    main()
