import re
import time
import threading
import queue
import numpy as np
import sounddevice as sd
from llama_cpp import Llama
from kokoro_onnx import Kokoro

# ========== Model Initialization ==========
llm = Llama(model_path="models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf", n_ctx=512, verbose=False, n_threads=4, n_batch=64)

kokoro = Kokoro(
    model_path="models/kokoro-v1.0.fp16.onnx",
    voices_path="models/voices-v1.0.bin"
)

# ========== Global State ==========
SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
audio_queue = queue.Queue(maxsize=100)
stream_lock = threading.Lock()
audio_stream = None
first_chunk_played = threading.Event()
is_playing_audio = False
stream_start_time = None
consumer_thread_running = threading.Event()

# ========== Utility Functions ==========

def sanitize_text_for_tts(text):
    """Clean and normalize input for TTS"""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace(',', '')  # Remove commas
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'[`~]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\[\]\{\}\(\)\|\\/]', '', text)
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1 to \2', text)
    text = text.replace('-', ' ')
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def initialize_audio_stream():
    global audio_stream
    with stream_lock:
        if audio_stream is None:
            try:
                audio_stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype='float32',
                    blocksize=CHUNK_SIZE,
                    latency='low'
                )
                audio_stream.start()
            except Exception as e:
                print(f"❌ Audio stream init error: {e}")

def consumer(prebuffer_count=1):
    global is_playing_audio
    initialize_audio_stream()
    if audio_stream is None:
        print("❌ Could not initialize audio stream")
        return

    is_playing_audio = True
    consumer_thread_running.set()

    try:
        while True:
            while audio_queue.qsize() < prebuffer_count:
                time.sleep(0.01)

            while not audio_queue.empty():
                samples, sr = audio_queue.get()
                if samples is None:
                    continue
                if not first_chunk_played.is_set():
                    latency = time.time() - stream_start_time
                    print(f"First chunk latency: {latency:.4f}s")
                    first_chunk_played.set()
                samples_np = samples.astype(np.float32).reshape(-1, 1)
                with stream_lock:
                    if audio_stream and audio_stream.active:
                        audio_stream.write(samples_np)
    except Exception as e:
        print(f"❌ Consumer error: {e}")
    finally:
        is_playing_audio = False
        consumer_thread_running.clear()

def speak(text, voice="af_heart", speed=1.15, lang="en-us"):
    global stream_start_time
    if not consumer_thread_running.is_set():
        stream_start_time = time.time()
        threading.Thread(target=consumer, daemon=True).start()
        first_chunk_played.clear()

    clean = sanitize_text_for_tts(text)
    try:
        samples, sr = kokoro.create(clean, voice=voice, speed=speed, lang=lang)
        audio_queue.put((samples, sr))
    except Exception as e:
        print(f"❌ Producer error: {e}")

# ========== LLM Streaming + Sentence TTS ==========

def stream_reply(prompt):
    sys_prompt = "<|system|>\nYou are a friendly voice assistant. Respond clearly and naturally as if spoken aloud.</s>\n"
    user_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>"
    return llm.create_completion(sys_prompt + user_prompt, stream=True, max_tokens=300, temperature=0.4)

def streaming_response_worker(prompt):
    buffer = ""

    for token in stream_reply(prompt):
        token_text = token['choices'][0].get('text', '')
        if not token_text.strip():
            continue
        buffer += token_text
        print(token_text, end="", flush=True)

        if re.search(r'[.?!]["\']?\s*$', buffer):
            sentence = sanitize_text_for_tts(buffer)
            speak(sentence)
            buffer = ""

    if buffer.strip():
        speak(sanitize_text_for_tts(buffer))

    # Wait for all audio to finish
    while is_playing_audio or not audio_queue.empty():
        time.sleep(0.05)

# ========== Main Loop ==========
def main():
    print("🧠 ARISE Real-Time Streaming + TTS")

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except EOFError:
            break

        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye.")
            break

        print("🤖 AI:", end=" ", flush=True)
        streaming_response_worker(user_input)

if __name__ == "__main__":
    main()
