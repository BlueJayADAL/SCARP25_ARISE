import wave
import time
import json
import numpy as np
from vosk import Model, KaldiRecognizer
import contextlib



AUDIO_PATH = "models/test_2.wav"
REPS = 5  # Number of repetitions

with contextlib.closing(wave.open("models/test_2.wav", 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    audio_duration = frames / float(rate)

# Load model
print("🔄 Loading Vosk model...")
vosk_model = Model("models/vosk-small")

# Prepare audio file
wf = wave.open(AUDIO_PATH, "rb")
audio_data = wf.readframes(wf.getnframes())
wf.close()

chunk_size = 4000  # ≈ 0.25 seconds

# Run benchmark
latencies = []
print(f"🎙️ Running STT test on {AUDIO_PATH} for {REPS} repetitions...\n")

for i in range(REPS):
    recognizer = KaldiRecognizer(vosk_model, 16000)
    recognizer.SetWords(True)

    start = time.time()
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        recognizer.AcceptWaveform(chunk)
    recognizer.FinalResult()
    end = time.time()

    total_latency = end - start
    processing_speed_ratio = audio_duration / total_latency
    overhead = total_latency - audio_duration

    print(f"Run {i+1}:")
    print(f"  🕒 Total Latency: {total_latency:.4f} sec")
    print(f"  🎧 Audio Length: {audio_duration:.4f} sec")
    print(f"  ⚡ Speed Ratio: {processing_speed_ratio:.2f}x")
    print(f"  📉 Overhead: {overhead:.4f} sec\n")

    latencies.append(total_latency)


# Report results
mean = np.mean(latencies)
std = np.std(latencies)
print(f"\n📊 Vosk STT Latency over {REPS} runs: avg = {mean:.4f}s, std = {std:.4f}s")
