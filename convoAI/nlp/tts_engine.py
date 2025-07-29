import threading
import queue
import time
import re
import numpy as np
import sounddevice as sd

from kokoro_onnx import Kokoro

#---------------------------------------------------------------------------------------------------------
#   Text-to-speech handeling through the user of producer and consumer threads. Uses ONNX version of Kokoro 82M specifically the floating point 16
#
#   Text response from system broken up by ending punctionation and then processed into audio chunks for lower latency
#
#   audio is added to a queue where a prebuffr is used to ensure that there is no odd delays between audio chunks being played
#---------------------------------------------------------------------------------------------------------



class TTSEngine:
    def __init__(self,
                 model_path="models/kokoro-v1.0.fp16.onnx",
                 voices_path="models/voices-v1.0.bin",
                 sample_rate=24000,
                 chunk_size=1024):

        print("\n🔊 Loading TTS Model (Kokoro)\n")
        self.kokoro = Kokoro(model_path=model_path, voices_path=voices_path)

        self.SAMPLE_RATE = sample_rate
        self.CHUNK_SIZE = chunk_size

        self.audio_queue = queue.Queue(maxsize=20)
        self.stream_lock = threading.Lock()
        self.audio_stream = None

        self._consumer_thread = None
        self._is_playing_audio = False
        self.stream_start_time = None
        self.first_chunk_played = threading.Event()

        self._initialize_audio_stream()

    def _initialize_audio_stream(self):
        with self.stream_lock:
            if self.audio_stream is None:
                try:
                    self.audio_stream = sd.OutputStream(
                        samplerate=self.SAMPLE_RATE,
                        channels=1,
                        dtype='float32',
                        blocksize=self.CHUNK_SIZE,
                        latency='low'
                    )
                    self.audio_stream.start()
                    print("✅ Audio stream initialized")
                except Exception as e:
                    print(f"❌ Audio stream init error: {e}")

    def _split_text(self, text, max_words=10):
        text = re.sub(r'\s+', ' ', text.strip())
        raw_sentences = re.split(r'(?<=[.?!])\s+', text)
        chunks = []

        for sentence in raw_sentences:
            words = sentence.strip().split()
            if len(words) <= max_words:
                chunks.append(sentence.strip())
            else:
                parts = re.split(r'(,\s+|\s+and\s+|\s+but\s+)', sentence)
                current_chunk = ""
                for part in parts:
                    test_chunk = (current_chunk + " " + part).strip()
                    if len(test_chunk.split()) <= max_words:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = part.strip()
                if current_chunk:
                    chunks.append(current_chunk)
        return [chunk for chunk in chunks if chunk]

    def speak(self, text, voice="af_heart", speed=1.0, lang="en-us"):
        print(f"🗣️ [TTS] Speaking: {text}")
        self.stream_start_time = time.time()
        self.first_chunk_played.clear()
        self._is_playing_audio = True

        # Clear previous audio
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Start producer
        threading.Thread(target=self._producer, args=(text, voice, speed, lang), daemon=True).start()

        # Start consumer thread if not running
        if not self._consumer_thread or not self._consumer_thread.is_alive():
            print("🧵 [TTS] Starting new consumer thread...")
            self._consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
            self._consumer_thread.start()
        else:
            print("🔁 [TTS] Consumer already running.")

    def _producer(self, text, voice, speed, lang):
        print(f"🧪 [TTS PRODUCER] Starting with text: {text}")
        chunks = self._split_text(text)
        print(f"🧪 [TTS PRODUCER] Chunks created: {chunks}")

        for chunk in chunks:
            try:
                samples, sr = self.kokoro.create(chunk, voice=voice, speed=speed, lang=lang)
                print(f"🧪 [TTS PRODUCER] Audio created for chunk: '{chunk}'")
                self.audio_queue.put((samples, sr))
            except Exception as e:
                print(f"❌ TTS producer error: {e}")

        print("🧪 [TTS PRODUCER] Finished queuing. Sending end signal.")
        self.audio_queue.put((None, None))

    def _consumer_loop(self, prebuffer_count=1):
        print("🔁 [TTS CONSUMER] Entered consumer loop")
        try:
            self._initialize_audio_stream()
            if self.audio_stream is None:
                print("❌ Could not initialize audio stream")
                return

            print("🔁 [TTS CONSUMER] Waiting for prebuffer...")
            while True:
                if self.audio_queue.qsize() >= prebuffer_count:
                    break
                with self.audio_queue.mutex:
                    if any(x[0] is None for x in self.audio_queue.queue):
                        break
                time.sleep(0.01)

            while True:
                print("🔁 [TTS CONSUMER] Waiting for next item...")
                samples, sr = self.audio_queue.get()
                if samples is None:
                    print("🛑 [TTS CONSUMER] Reached end of queue")
                    break

                if not self.first_chunk_played.is_set():
                    latency = time.time() - self.stream_start_time
                    print(f"⏱️ First chunk latency: {latency:.4f} sec")
                    self.first_chunk_played.set()

                samples_np = samples.astype(np.float32).reshape(-1, 1)
                with self.stream_lock:
                    if self.audio_stream and self.audio_stream.active:
                        print("🔊 [TTS CONSUMER] Playing audio chunk...")
                        self.audio_stream.write(samples_np)
                        print("✅ [TTS CONSUMER] Chunk written")

        except Exception as e:
            print(f"❌ TTS consumer error: {e}")
        finally:
            print("🔕 [TTS CONSUMER] Playback complete. Resetting flag.")
            self._is_playing_audio = False

    def is_playing(self):
        return self._is_playing_audio
