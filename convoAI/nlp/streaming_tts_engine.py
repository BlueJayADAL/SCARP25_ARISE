import threading
import queue
import time
import re
import numpy as np
import sounddevice as sd

from kokoro_onnx import Kokoro

#---------------------------------------------------------------------------------------------------------
#   Enhanced TTS Engine optimized for streaming sentence-by-sentence input
#
#   Maintains audio queue continuity when receiving multiple speak() calls
#   Optimized for low-latency streaming from LLM sentence generation
#---------------------------------------------------------------------------------------------------------

class StreamingTTSEngine:
    def __init__(self,
                 model_path="models/kokoro-v1.0.fp16.onnx",
                 voices_path="models/voices-v1.0.bin",
                 sample_rate=24000,
                 chunk_size=1024):

        print("\n🔊 Loading Streaming TTS Model (Kokoro)\n")
        self.kokoro = Kokoro(model_path=model_path, voices_path=voices_path)

        self.SAMPLE_RATE = sample_rate
        self.CHUNK_SIZE = chunk_size

        self.audio_queue = queue.Queue(maxsize=50)  # Increased for streaming
        self.stream_lock = threading.Lock()
        self.audio_stream = None

        self._consumer_thread = None
        self._is_playing_audio = False
        self._consumer_running = False
        self.stream_start_time = None
        self.first_chunk_played = threading.Event()
        
        # Streaming-specific attributes
        self._sentence_count = 0
        self._current_stream_id = None

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

    def start_streaming_session(self):
        """Start a new streaming session - call before first sentence"""
        import uuid
        self._current_stream_id = str(uuid.uuid4())[:8]
        self._sentence_count = 0
        self.stream_start_time = time.time()
        self.first_chunk_played.clear()
        
        print(f"🎬 Starting streaming session: {self._current_stream_id}")
        
        # Ensure consumer is running
        self._ensure_consumer_running()

    def speak_streaming(self, text, voice="af_heart", speed=1.0, lang="en-us"):
        """
        Speak a sentence as part of a streaming session.
        Optimized for rapid successive calls.
        """
        if not self._current_stream_id:
            self.start_streaming_session()
            
        self._sentence_count += 1
        sentence_id = f"{self._current_stream_id}-{self._sentence_count}"
        
        print(f"🗣️ [TTS-STREAM] Speaking sentence {self._sentence_count}: {text}")
        
        if not self._is_playing_audio:
            self._is_playing_audio = True

        # Generate audio in separate thread to not block LLM generation
        threading.Thread(
            target=self._produce_audio,
            args=(text, voice, speed, lang, sentence_id),
            daemon=True
        ).start()

    def end_streaming_session(self):
        """Signal end of streaming session"""
        if self._current_stream_id:
            print(f"🎬 Ending streaming session: {self._current_stream_id}")
            # Add end marker to queue
            self.audio_queue.put((None, None, "END"))
            self._current_stream_id = None

    def speak(self, text, voice="af_heart", speed=1.0, lang="en-us"):
        """
        Traditional speak method - processes entire text at once
        Maintains backward compatibility
        """
        self.start_streaming_session()
        
        # Clear previous audio
        self._clear_queue()
        
        chunks = self._split_text(text)
        print(f"🗣️ [TTS] Speaking {len(chunks)} chunks: {text}")

        for i, chunk in enumerate(chunks):
            self._produce_audio(chunk, voice, speed, lang, f"chunk-{i}")
            
        self.end_streaming_session()

    def _produce_audio(self, text, voice, speed, lang, audio_id):
        """Generate audio and add to queue"""
        try:
            print(f"🧪 [TTS PRODUCER] Generating audio for: '{text}' (ID: {audio_id})")
            samples, sr = self.kokoro.create(text, voice=voice, speed=speed, lang=lang)
            print(f"🧪 [TTS PRODUCER] Audio ready for: '{text}'")
            self.audio_queue.put((samples, sr, audio_id))
        except Exception as e:
            print(f"❌ TTS producer error for '{text}': {e}")

    def _ensure_consumer_running(self):
        """Ensure consumer thread is running"""
        if not self._consumer_thread or not self._consumer_thread.is_alive():
            print("🧵 [TTS] Starting consumer thread...")
            self._consumer_running = True
            self._consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
            self._consumer_thread.start()

    def _consumer_loop(self, prebuffer_count=1):
        """Enhanced consumer loop for streaming"""
        print("🔁 [TTS CONSUMER] Started streaming consumer loop")
        
        try:
            self._initialize_audio_stream()
            if self.audio_stream is None:
                print("❌ Could not initialize audio stream")
                return

            while self._consumer_running:
                try:
                    # Wait for audio with timeout to allow thread cleanup
                    samples, sr, audio_id = self.audio_queue.get(timeout=1.0)
                    
                    # Check for end marker
                    if samples is None and audio_id == "END":
                        print("🛑 [TTS CONSUMER] Received END signal")
                        # Continue running for next streaming session
                        continue
                    
                    if samples is None:
                        continue

                    # Track first chunk latency
                    if not self.first_chunk_played.is_set() and self.stream_start_time:
                        latency = time.time() - self.stream_start_time
                        print(f"⏱️ First chunk latency: {latency:.4f} sec")
                        self.first_chunk_played.set()

                    # Play audio
                    samples_np = samples.astype(np.float32).reshape(-1, 1)
                    with self.stream_lock:
                        if self.audio_stream and self.audio_stream.active:
                            print(f"🔊 [TTS CONSUMER] Playing audio: {audio_id}")
                            self.audio_stream.write(samples_np)
                            
                except queue.Empty:
                    # Timeout - check if we should continue running
                    if not self._is_playing_audio and self.audio_queue.empty():
                        # Reset playing flag when queue is empty
                        self._is_playing_audio = False
                    continue
                    
        except Exception as e:
            print(f"❌ TTS consumer error: {e}")
        finally:
            print("🔕 [TTS CONSUMER] Consumer loop ended")
            self._consumer_running = False
            self._is_playing_audio = False

    def _clear_queue(self):
        """Clear audio queue"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _split_text(self, text, max_words=10):
        """Split text into manageable chunks"""
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

    def is_playing(self):
        """Check if TTS is currently playing audio"""
        return self._is_playing_audio or not self.audio_queue.empty()
    
    def wait_for_completion(self, timeout=30):
        """Wait for all queued audio to finish playing"""
        start_time = time.time()
        while self.is_playing() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if self.is_playing():
            print(f"⚠️ TTS completion timeout after {timeout}s")
            return False
        return True

    def stop_playback(self):
        """Stop current playback and clear queue"""
        print("🛑 Stopping TTS playback")
        self._clear_queue()
        self._is_playing_audio = False