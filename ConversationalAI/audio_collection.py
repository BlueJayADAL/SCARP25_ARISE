import os
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

# ==========================
# Config
# ==========================
kokoro = Kokoro(model_path="models/kokoro-v1.0.fp16.onnx", voices_path="models/voices-v1.0.bin")
output_dir = "tts_cache"
os.makedirs(output_dir, exist_ok=True)

# ==========================
# TTS Audio Caching Function
# ==========================
def cache_tts_audio(text, filename=None, voice="af_heart", speed=1.15, lang="en-us", samplerate=24000):
    if not filename:
        # Create filename based on text hash
        safe_name = text.lower().strip().replace(" ", "_")[:40]
        filename = f"{safe_name}.wav"
    
    out_path = os.path.join(output_dir, filename)

    if os.path.exists(out_path):
        print(f"✅ Cached file already exists: {out_path}")
        return out_path

    try:
        samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        samples = samples.astype(np.float32)
        sf.write(out_path, samples, samplerate)
        print(f"💾 Saved TTS audio to: {out_path}")
        return out_path
    except Exception as e:
        print(f"❌ TTS generation failed: {e}")
        return None

# ==========================
# Example Usage
# ==========================
if __name__ == "__main__":
    phrases = [
        "Hello this is the ARISE system, how may I help you today?",
        "I noticed your range of motion may need adjusting. Would you like me to change it?",
        "Ok, starting back up",
        "what exercise would you like to perform please provide repetitions and name of exercise",
        "Finishing exercise",
        "Great work, finishing exercise",
        "Okay, adjusting your range of motion.",
        "Understood, keeping current range.",
        "Okay, pausing. Let me know if you want to continue, start something new, or end it.",
        "Okay, stopping the program now.",
        "Okay, restarting exercise.",
        "Keep your back straight and avoid rounding. Engage your core and maintain a neutral spine throughout the movement.",
     "Tuck your elbows in close to your sides to protect your shoulders and maintain better control.",
     "Fully extend your arms without locking your elbows. This helps control the movement and targets the right muscles.",
     "Lift your head and look slightly ahead. This keeps your neck aligned and helps balance your posture.",
     "Push your hips back as if you're sitting in a chair. Don’t let your knees go too far forward.",
     "Make sure your knees stay aligned over your toes. Avoid letting them cave inward or drift too far forward.",
     "Position your elbows directly under your shoulders to maintain joint alignment and control.",
     "Raise or lower your arms to match each other. Keeping them level helps maintain symmetry and proper form.",
     "Set your feet shoulder-width apart to create a stable base and prevent imbalance.",
     "Keep both shoulders at the same height. This improves balance and prevents overuse on one side.",
     "Lift your upper body so your back stays above your hips. Don’t lean too far forward.",
     "Angle your knees slightly outward, in line with your toes. This protects your joints and keeps your stance strong.",
     "You are out of the camera frame, for accurate critiques please enter the frame of the camera" ,
     "You are too close to the camera for this exercise, I will not be able to provide any meaningful corrections" ,
     "Please be front facing to the camera for this exercise",

    ]

    for phrase in phrases:
        cache_tts_audio(phrase, speed=1.15)
