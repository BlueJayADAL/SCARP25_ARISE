import sounddevice as sd
import numpy as np
import time
import json
from vosk import Model, KaldiRecognizer

# ---------- CONFIG ----------
MODEL_PATH = "models/vosk-small"
SAMPLE_RATE = 16000
CHUNK_SIZE = 216  # in bytes (4000 bytes = 2000 samples @ 16-bit audio)
DURATION = 10  # seconds to listen
# ----------------------------

# Load Vosk model
print("🔄 Loading model...")
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)
recognizer.SetWords(True)

# Timing variables
start_time = None
response_ready_time = None

def callback(indata, frames, time_info, status):
    global start_time, response_ready_time

    if status:
        print("⚠️", status)

    audio_bytes = bytes(indata)

    # Check if audio contains voice (i.e., not silent)
    volume_norm = np.linalg.norm(np.frombuffer(audio_bytes, dtype=np.int16)) / len(audio_bytes)

    text = ""
    partial = ""

    if start_time is None and volume_norm > 0.01:  # You can tune the silence threshold
        start_time = time.time()

    if recognizer.AcceptWaveform(audio_bytes):
        result = json.loads(recognizer.Result())
        text = result.get("text", "")
        if text and response_ready_time is None:
            response_ready_time = time.time()
    else:
        partial = json.loads(recognizer.PartialResult()).get("partial", "")
        if partial and response_ready_time is None:
            response_ready_time = time.time()

    # Once we have both timestamps, compute latency
    if start_time and response_ready_time:
        latency = response_ready_time - start_time
        print(f"\n🧍 First response: {partial or text}")
        print(f"⏱️ Response latency: {latency:.3f} seconds\n")
        # Reset for next utterance
        start_time = None
        response_ready_time = None



print("🎙️ Listening... Speak into the mic.")

# Start audio stream
with sd.RawInputStream(samplerate=SAMPLE_RATE,
                       blocksize=CHUNK_SIZE,
                       dtype='int16',
                       channels=1,
                       callback=callback):
    sd.sleep(DURATION * 1000)

print("🛑 Done.")
