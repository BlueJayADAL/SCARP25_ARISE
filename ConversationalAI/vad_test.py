import sounddevice as sd
import numpy as np
import time
import json
from vosk import Model, KaldiRecognizer

# === Config ===
VOSK_MODEL_PATH = "models/vosk-small"
BLOCKSIZE = 1024
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.025
SILENCE_DURATION = 0.8  # Balanced: not too short, not too long
MIN_SPEECH_DURATION = 0.2  # Very short minimum

# === Load Vosk Model ===
print("🔄 Loading Vosk model...")
model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)
recognizer.SetWords(True)

# === VAD State ===
speaking = False
speech_start_time = 0.0
last_speech_time = 0.0
audio_buffer = bytearray()

def callback(indata, frames, time_info, status):
    global speaking, speech_start_time, last_speech_time, audio_buffer

    # Convert to float for RMS calculation
    samples = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(samples**2))
    
    current_time = time.time()
    
    print(f"\r🎚️ RMS: {rms:.4f} | Speaking: {speaking} | Buffer: {len(audio_buffer)} bytes", end="")

    if rms > VAD_THRESHOLD:
        # Voice detected
        last_speech_time = current_time
        
        if not speaking:
            speaking = True
            speech_start_time = current_time
            audio_buffer.clear()
            recognizer.Reset()  # Fresh start for new speech
            print("\n🎙️ Speech started")
        
        # Convert indata to bytes for Vosk
        audio_bytes = bytes(indata)
        audio_buffer.extend(audio_bytes)
        
        # CONTINUOUS PROCESSING: Feed audio to Vosk in real-time
        if recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(recognizer.Result())
            partial_text = result.get("text", "").strip()
            if partial_text:
                print(f"\n🔄 Partial: '{partial_text}'")
        
    else:
        # No voice detected
        if speaking:
            silence_duration = current_time - last_speech_time
            speech_duration = current_time - speech_start_time
            
            # Check if we should end the speech detection
            if silence_duration > SILENCE_DURATION and speech_duration > MIN_SPEECH_DURATION:
                speaking = False
                print(f"\n🛑 Speech ended (duration: {speech_duration:.1f}s)")
                
                # Get final result from any remaining audio
                final_result = json.loads(recognizer.FinalResult())
                final_text = final_result.get("text", "").strip()
                
                if final_text:
                    print(f"\n✅ FINAL RESULT: '{final_text}'\n")
                else:
                    print("\n⚠️ No final transcription\n")
                
                audio_buffer.clear()
                
            elif silence_duration > SILENCE_DURATION and speech_duration <= MIN_SPEECH_DURATION:
                # Speech too short, ignore
                speaking = False
                audio_buffer.clear()
                recognizer.Reset()
                print(f"\n⚠️ Speech too short ({speech_duration:.1f}s), ignored\n")

# Enhanced version with smoothing
def callback_with_smoothing(indata, frames, time_info, status):
    global speaking, speech_start_time, last_speech_time, audio_buffer
    
    # Convert to float for RMS calculation
    samples = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(samples**2))
    
    # Simple smoothing to reduce false triggers
    if not hasattr(callback_with_smoothing, 'rms_history'):
        callback_with_smoothing.rms_history = []
    
    callback_with_smoothing.rms_history.append(rms)
    if len(callback_with_smoothing.rms_history) > 3:
        callback_with_smoothing.rms_history.pop(0)
    
    # Use average of recent RMS values
    avg_rms = np.mean(callback_with_smoothing.rms_history)
    
    current_time = time.time()
    print(f"\r🎚️ RMS: {rms:.4f} (avg: {avg_rms:.4f}) | Speaking: {speaking}", end="")

    if avg_rms > VAD_THRESHOLD:
        last_speech_time = current_time
        
        if not speaking:
            speaking = True
            speech_start_time = current_time
            audio_buffer.clear()
            recognizer.Reset()
            print("\n🎙️ Speech started")
        
        audio_bytes = bytes(indata)
        audio_buffer.extend(audio_bytes)
        
        # Feed to Vosk immediately for best accuracy
        if recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                print(f"\n🔄 '{text}'")
        
    else:
        if speaking:
            silence_duration = current_time - last_speech_time
            speech_duration = current_time - speech_start_time
            
            if silence_duration > SILENCE_DURATION and speech_duration > MIN_SPEECH_DURATION:
                speaking = False
                print(f"\n🛑 Speech ended")
                
                # Get final result
                final_result = json.loads(recognizer.FinalResult())
                final_text = final_result.get("text", "").strip()
                
                if final_text:
                    print(f"\n✅ FINAL: '{final_text}'\n")
                else:
                    print("\n⚠️ No final result\n")
                
                audio_buffer.clear()

print("🎤 Voice Activity Detection with Real-time Processing")
print(f"📋 Threshold: {VAD_THRESHOLD}, Silence timeout: {SILENCE_DURATION}s")
print("🗣️ Choose your callback version:")
print("   1. Standard (immediate processing)")
print("   2. Smoothed (reduces false triggers)")

choice = input("Enter 1 or 2: ").strip()
selected_callback = callback_with_smoothing if choice == "2" else callback

print(f"\n🚀 Starting with {'smoothed' if choice == '2' else 'standard'} callback...")

try:
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, dtype='int16',
                           channels=1, callback=selected_callback):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n👋 Stopping...")
except Exception as e:
    print(f"\n❌ Error: {e}")