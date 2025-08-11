import sounddevice as sd
from kokoro_onnx import Kokoro
import re
import time
import threading
import queue
import numpy as np

kokoro = Kokoro(model_path="models/kokoro-v1.0.fp16.onnx", voices_path="models/voices-v1.0.bin")

import pandas as pd
import uuid
from datetime import datetime
import os

SESSION_ID = uuid.uuid4().hex[:8]
log_buffer = []
LOG_PATH = "latency_logs/TTS_latency-test.csv"
os.makedirs("latency_logs", exist_ok=True)

#-----------------------------------
#config variables
text = "Older adults recovering from surgery often require guided exercise " \
       "and rehabilitation at home to regain strength and mobility. However," \
       "ensuring they perform exercises correctly, safely, and consistently is " \
       "challenging without professional supervision. AI-powered at-home " \
       "exercise systems aim to fill this gap by using multiple modalities " \
       "computer vision, voice interaction, and wearable sensors to monitor performance and coach the user in real time. Recent advances " \
       "in edge computing (for example, NVIDIA Jetson Nano) make it feasible to " \
       "run sophisticated pose-tracking and voice assistants locally, pre-serving privacy and reducing latency. This report reviews the state " \
       "of the art in these areas and outlines a development roadmap. We " \
       "focus on: (1) real-time pose estimation methods (2) Voice-based interface"
speed = 1.15
out_samp = 24000
split_length = 20
#-----------------------------------
audio_queue = queue.Queue()

def split_text(text, max_words=22):
    text = re.sub(r'\s+', ' ', text.strip())
    sentence_chunks = re.split(r'(?<=[.!?])\s+', text)

    refined_chunks = []
    clause_split_regex = r'\b(and|or|but|so|because|although|while|if|though|unless|whereas|however|moreover|therefore|meanwhile|since|then|yet)\b'

    for sentence in sentence_chunks:
        clause_chunks = re.split(clause_split_regex, sentence, flags=re.IGNORECASE)

        # Rejoin split conjunctions into complete clauses
        clauses = []
        i = 0
        while i < len(clause_chunks):
            if i + 1 < len(clause_chunks):
                clause = clause_chunks[i].strip() + ' ' + clause_chunks[i + 1].strip()
                clauses.append(clause)
                i += 2
            else:
                clauses.append(clause_chunks[i].strip())
                i += 1

        for clause in clauses:
            words = clause.split()
            if len(words) <= max_words:
                refined_chunks.append(clause.strip())
            else:
                for j in range(0, len(words), max_words):
                    chunk = " ".join(words[j:j + max_words])
                    refined_chunks.append(chunk.strip())

    return refined_chunks



# ⏱️ Shared variable to store the initial call time
stream_start_time = None
first_chunk_played = threading.Event()

def producer(text, voice="af_heart", speed=speed, lang="en-us"):
    for chunk in split_text(text, max_words=split_length):
        try:
            samples, sr = kokoro.create(chunk, voice=voice, speed=speed, lang=lang)
            audio_queue.put((samples, sr))
        except Exception as e:
            print(f"❌ Producer error: {e}")
    audio_queue.put((None, None))  # signal end

def consumer():
    try:
        with sd.OutputStream(samplerate=out_samp, channels=1, dtype='float32') as stream:
            while True:
                samples, sr = audio_queue.get()
                if samples is None:
                    break

                if not first_chunk_played.is_set():
                    latency = time.time() - stream_start_time
                    times_lst.append(latency)
                    print(f"🕒 First chunk playback latency: {latency:.4f} seconds")

                    log_buffer.append({
                        "run_id": SESSION_ID,
                        "component": "LLM",
                        "timestamp": datetime.now().isoformat(),
                        "latency": round(latency, 4),
                        "text": text,
                        "speed":speed,
                        "split length":split_length,
                        "output sample rate":out_samp,
                        "model":"kokoro-v1.0.fp16.onnx"
                    })
                    first_chunk_played.set()

                # Ensure samples are float32 and 2D (required by stream.write)
                samples_np = samples.astype(np.float32).reshape(-1, 1)
                stream.write(samples_np)
                time.sleep(len(samples) / sr * 0.1)
    except Exception as e:
        print(f"❌ Consumer error: {e}")


def speak_streamed(text, voice="af_heart", speed=speed, lang="en-us"):
    global stream_start_time
    stream_start_time = time.time()
    first_chunk_played.clear()

    producer_thread = threading.Thread(target=producer, args=(text, voice, speed, lang))
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

times_lst = []
repetitions = 5
for i in range(repetitions):
    speak_streamed(text)
print(f"Average delay from function call to audio output: {sum(times_lst)/repetitions}")

df = pd.DataFrame(log_buffer)
if not os.path.isfile(LOG_PATH):
    df.to_csv(LOG_PATH, index=False)
else:
    df.to_csv(LOG_PATH, mode="a", header=False, index=False)

print(f"\n📁 Logged {len(df)} entries to {LOG_PATH} under run_id {SESSION_ID}")
