from fastapi import APIRouter, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer
import wave
import json
import os
import shutil


router = APIRouter()


# Load VOSK model once
model = Model("../../models/vosk-small")

@router.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    temp_path = f"temp_{audio.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    wf = wave.open(temp_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            results.append(res.get("text", ""))

    final_result = json.loads(rec.FinalResult())
    results.append(final_result.get("text", ""))

    transcript = " ".join(results)

    wf.close()
    os.remove(temp_path)

    return {"transcript": transcript}
