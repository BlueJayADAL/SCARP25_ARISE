import pyaudio
import soundfile as sf
import re
import time
import wave
import json
import os
import tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue




from vosk import Model, KaldiRecognizer
from llama_cpp import Llama

#from TTS.api import TTS
from kokoro import KPipeline

from pydub import AudioSegment
import simpleaudio as sa

# ------------------
# Configurable Paths
# ------------------
VOSK_MODEL_PATH = "vosk-small"

# ------------------
# Initialize Models
# ------------------
print("\nLoading Vosk Model \n")
vosk_model = Model(VOSK_MODEL_PATH)

print("\nLoading Llama Model \n")
llm = Llama(model_path="llm-models/tinyllama-1.1b-chat-v1.0.Q6_K.gguf", n_ctx=256, verbose=False, n_threads = 4, n_batch=64)

print("\nLoading TTS Model \n")
#tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True, gpu=False)
tts_pipeline = KPipeline(lang_code='a')

# ------------------
# Audio Recording (pyaudio)
# ------------------
def record_audio(filename="temp.wav", duration=8, rate=16000):
    chunk = 1024
    fmt = pyaudio.paInt16
    channels = 1

    pa = pyaudio.PyAudio()
    stream = pa.open(format=fmt, channels=channels, rate=rate,
                     input=True, frames_per_buffer=chunk)

    print("🎙️ Recording...")
    frames = [stream.read(chunk) for _ in range(int(rate / chunk * duration))]
    print("✅ Done recording.")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pa.get_sample_size(fmt))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

# ------------------
# Speech to Text (VOSK)
# ------------------
def transcribe_audio(file_path):
    vosk_start_time = time.time()
    wf = wave.open(file_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())

    result = ''
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())["text"]

    wf.close()
    vosk_end_time = time.time()
    print(f"Vosk processing time: {vosk_end_time - vosk_start_time:.4f} seconds")
    return result

# ------------------
# Generate Reply (LLaMA)
# ------------------
def generate_reply(prompt):
    llm_start_time = time.time()
    output = llm(f"PROMPT: {prompt}. Respond clearly, naturally, and concisely as if spoken aloud. No markdown or formatting.",max_tokens=250, temperature=0.4,  )
    llm_end_time = time.time()
    print(f"LLM processing time: {llm_end_time - llm_start_time:.4f} seconds")
    return output['choices'][0]['text'].strip()

# ------------------
# Cleaning Output of LLM 
# ------------------
def clean_text_for_tts(text):
    # Remove markdown: bold, italic, code
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)      # **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)          # *italic*
    text = re.sub(r'__(.*?)__', r'\1', text)          # __bold__
    text = re.sub(r'_(.*?)_', r'\1', text)            # _italic_
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text)       # inline code or code blocks
    text = re.sub(r'<[^>]+>', '', text)              # HTML-like tags

    # Remove special prefixes like /think /do /note
    text = re.sub(r'/\w+\b', '', text)

    # Remove stray slashes, brackets, braces, pipes, etc.
    text = re.sub(r'[\[\]\{\}\(\)\|\\\/]', '', text)

    # Replace multiple spaces with one
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


# ------------------
# Speak Response (TTS)
# ------------------
def speak(text):
    audio_queue = queue.Queue()
    first_chunk_time = None

    def producer():
        nonlocal first_chunk_time
        try:
            generator = tts_pipeline(text, voice='af_heart', speed=1.0)
            for idx, (_, _, audio) in enumerate(generator):
                if idx == 0:
                    first_chunk_time = time.time()
                audio_queue.put((idx, audio))
            audio_queue.put((None, None))  # Signal done
        except Exception as e:
            print(f"❌ Error in Kokoro TTS: {e}")
            audio_queue.put((None, None))

    def consumer():
        try:
            while True:
                idx, audio = audio_queue.get()
                if idx is None:
                    break
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    sf.write(f.name, audio, 24000)
                    f.close()
                    sound = AudioSegment.from_wav(f.name)
                    raw_data = sound.raw_data
                    play_obj = sa.play_buffer(raw_data, num_channels=sound.channels,
                                              bytes_per_sample=sound.sample_width,
                                              sample_rate=sound.frame_rate)
                    play_obj.wait_done()
                    os.remove(f.name)
        except Exception as e:
            print(f"❌ Error playing audio chunk: {e}")

    start_time = time.time()

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()

    if first_chunk_time:
        print(f"⏱️ TTS first chunk latency: {first_chunk_time - start_time:.4f} seconds")

    consumer_thread.join()

# ------------------
# Main Chat Loop
# ------------------
def chatbot_loop():
    while True:
        wav_file = record_audio()
        user_text = transcribe_audio(wav_file)

        if not user_text:
            print("❌ No input detected.")
            continue

        print(f"🧍 You: {user_text}")
        if user_text.lower() in ["exit", "quit", "stop"]:
            break

        reply = generate_reply(user_text)
        clean_reply = clean_text_for_tts(reply)
        print(f"🤖 Bot: {clean_reply}")
        speak(clean_reply)

# ------------------
# Start Chat
# ------------------
if __name__ == "__main__":
    chatbot_loop()
