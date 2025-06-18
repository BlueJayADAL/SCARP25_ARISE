import wave
import time
import json
import numpy as np
import contextlib
from vosk import Model, KaldiRecognizer

import subprocess
import os

AUDIO_PATH = "models/test_STT.wav"
CHUNK_SIZE = 4000  # 0.25s at 16kHz
REPS = 5

# Load model
print("🔄 Loading Vosk model...")
vosk_model = Model("models/vosk-small")



def convert_to_vosk_compatible_audio(input_path, output_path=None):
    """
    Converts an audio file to Vosk-compatible format:
    - 16kHz sample rate
    - Mono (1 channel)
    - 16-bit PCM WAV

    Parameters:
        input_path (str): Path to the original audio file (e.g., .wav, .mp3).
        output_path (str, optional): Path to save the converted file.
                                     If None, overwrites input file with "_converted.wav".

    Returns:
        str: Path to the converted file.
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + "_converted.wav"

    command = [
        "ffmpeg",
        "-y",  # Overwrite output file without asking
        "-i", input_path,
        "-ar", "16000",  # Set sample rate to 16kHz
        "-ac", "1",      # Set to mono
        "-sample_fmt", "s16",  # Set 16-bit PCM
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Converted to Vosk-compatible audio: {output_path}")
        return output_path
    except subprocess.CalledProcessError:
        raise RuntimeError("❌ Failed to convert audio using ffmpeg. Make sure ffmpeg is installed.")
    
AUDIO_PATH = convert_to_vosk_compatible_audio(AUDIO_PATH)


# Load audio and check format
with contextlib.closing(wave.open(AUDIO_PATH, 'rb')) as wf:
    assert wf.getnchannels() == 1, "Audio must be mono"
    assert wf.getsampwidth() == 2, "Audio must be 16-bit PCM"
    assert wf.getframerate() == 16000, "Audio must be 16kHz"
    audio_data = wf.readframes(wf.getnframes())
    audio_duration = wf.getnframes() / float(wf.getframerate())

# Test runs
print(f"\n🎧 Testing latency using '{AUDIO_PATH}' ({audio_duration:.2f}s long)\n")

for run in range(REPS):
    recognizer = KaldiRecognizer(vosk_model, 16000)
    recognizer.SetWords(True)

    print(f"▶️ Run {run+1}")
    sentence_buffer = ""
    text_ready_time = None
    start_time = time.time()

    for i in range(0, len(audio_data), CHUNK_SIZE):
        chunk = audio_data[i:i+CHUNK_SIZE]

         #Simulate real-time streaming (optional sleep)
          # simulate wall time

        if recognizer.AcceptWaveform(chunk):
            result = json.loads(recognizer.Result())
            sentence_buffer += result.get("text", "") + " "
            # Capture time when meaningful text first appears
            if sentence_buffer.strip() and text_ready_time is None:
                text_ready_time = time.time()
        time.sleep(CHUNK_SIZE / (2 * 16000))
    # Final flush
    final_result = json.loads(recognizer.FinalResult())
    sentence_buffer += final_result.get("text", "")

    if text_ready_time is None and sentence_buffer.strip():
        text_ready_time = time.time()
    
    

    end_time = time.time()  # When STT processing fully completes
    simulated_speech_end = start_time + audio_duration

    # Metrics
    total_latency = end_time - start_time
    processing_overhead = total_latency - audio_duration
    real_time_factor = total_latency / audio_duration

    print("\n🧍 Final Recognized Text:\n", (sentence_buffer or "").strip())

    print(f"\n🎧 Audio Duration         : {audio_duration:.3f} sec")
    print(f"🕒 Total STT Latency      : {total_latency:.3f} sec")
    print(f"📉 Processing Overhead    : {processing_overhead:.3f} sec")
    print(f"⚡ Real-Time Factor (RTF) : {real_time_factor:.3f}")
    