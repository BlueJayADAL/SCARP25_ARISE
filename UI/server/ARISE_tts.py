import re
import time
import threading
import queue
import numpy as np
import soundfile as sf
import tempfile
from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro

router = APIRouter()


kokoro = Kokoro(
    model_path="../../models/kokoro-v1.0.fp16.onnx",
    voices_path="../../models/voices-v1.0.bin"
)

audio_queue = queue.Queue()

def split_text(text):
    # Only splits at sentence boundaries
    import re
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s for s in sentences if s]

def producer(text, voice="af_heart", speed=1.15, lang="en-us"):
    for chunk in split_text(text):
        try:
            samples, sr = kokoro.create(chunk, voice=voice, speed=speed, lang=lang)
            audio_queue.put((samples, sr))
        except Exception as e:
            print(f"❌ Producer error: {e}")
    audio_queue.put((None, None))  # signal end

def collect_audio_chunks(text, voice="af_heart", speed=1.15, lang="en-us"):
    """Runs the producer in a thread, collects all audio chunks into a numpy array."""
    # Clear the queue
    while not audio_queue.empty():
        audio_queue.get()
    # Start the producer
    prod_thread = threading.Thread(target=producer, args=(text, voice, speed, lang))
    prod_thread.start()
    # Collect all audio
    audio_segments = []
    sample_rate = 24000
    while True:
        samples, sr = audio_queue.get()
        if samples is None:
            break
        audio_segments.append(samples.astype(np.float32))
        sample_rate = sr  # Update in case it's dynamic
    prod_thread.join()
    if not audio_segments:
        return None, sample_rate
    full_audio = np.concatenate(audio_segments)
    return full_audio, sample_rate

# FastAPI setup

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.15
    lang: str = "en-us"

@router.post("/synthesize")
async def synthesize_tts(req: TTSRequest):
    audio, sr = collect_audio_chunks(req.text, req.voice, req.speed, req.lang)
    if audio is None:
        return {"error": "Audio synthesis failed."}
    # Save to temp WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, audio, int(sr))
        tmp_path = tmp.name
    return FileResponse(tmp_path, media_type="audio/wav", filename="tts_output.wav")