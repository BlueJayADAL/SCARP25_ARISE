import sounddevice as sd
from kokoro_onnx import Kokoro
import soundfile as sf
import torch
import re
import time
import numpy as np

import threading
import queue



kokoro = Kokoro(model_path="models/kokoro-v1.0.fp16.onnx", voices_path="models/voices-v1.0.bin")

text = "Older adults recovering from surgery often require guided exercise "\
        "and rehabilitation at home to regain strength and mobility. However," \
        "ensuring they perform exercises correctly, safely, and consistently is " \
        "challenging without professional supervision. AI-powered at-home " \
        "exercise systems aim to fill this gap by using multiple modalities " \
        "computer vision, voice interaction, and wearable sensors to monitor performance and coach the user in real time. Recent advances "\
        "in edge computing (for example, NVIDIA Jetson Nano) make it feasible to " \
        "run sophisticated pose-tracking and voice assistants locally, pre-serving privacy and reducing latency. This report reviews the state " \
        "of the art in these areas and outlines a development roadmap. We " \
        "focus on: (1) real-time pose estimation methods (2) Voice-based interface"

audio_queue = queue.Queue()

def split_text(text):
    """
    Splits input text into sentence-like chunks using punctuation boundaries.
    Handles '.', '?', '!', and newlines.
    """
    # Remove any excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Split on punctuation followed by a space or end of string
    # Lookbehind ensures punctuation is part of the split
    chunks = re.split(r'(?<=[.?!:;])\s+', text)

    # Filter out empty results
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def producer(text, voice="af_heart", speed=1.0, lang="en-us"):
    for chunk in split_text(text):
        try:
            samples, sr = kokoro.create(chunk, voice=voice, speed=speed, lang=lang)
            audio_queue.put((samples, sr))
        except Exception as e:
            print(f"❌ Producer error: {e}")
    audio_queue.put((None, None))  # signal end

def consumer():
    """Play audio as it becomes available, with tight gapless playback"""
    try:
        while True:
            samples, sr = audio_queue.get()
            if samples is None:
                break
            sd.play(samples, sr)
            sd.wait()
    except Exception as e:
        print(f"❌ Consumer error: {e}")

def speak_streamed(text, voice="af_heart", speed=1.0, lang="en-us"):
    start = time.time()
    producer_thread = threading.Thread(target=producer, args=(text, voice, speed, lang))
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    print(f"Total TTS time: {time.time() - start:.2f}s")

speak_streamed(text)
