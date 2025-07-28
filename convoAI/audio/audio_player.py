import threading
import queue
import sounddevice as sd
import soundfile as sf
import time

class AudioPlayer:
    def __init__(self):
        self.audio_file_queue = queue.Queue()
        self.stop_audio_flag = threading.Event()
        self._is_playing = False
        self.thread = threading.Thread(target=self._audio_playback_loop, daemon=True)
        self.thread_started = False

    def start(self):
        """Start the audio playback thread (should be called once)."""
        if not self.thread_started:
            self.thread.start()
            self.thread_started = True

    def play_file(self, filepath: str):
        """Add an audio file to the playback queue."""
        self.audio_file_queue.put(filepath)

    def stop(self):
        """Stops playback thread cleanly (optional use on shutdown)."""
        self.audio_file_queue.put(None)

    def is_playing(self):
        return self._is_playing

    def clear_queue(self):
        """Clear queued audio files (if interrupted)."""
        with self.audio_file_queue.mutex:
            self.audio_file_queue.queue.clear()

    def _audio_playback_loop(self):
        print("Audio file playback loop\n")
        while True:
            if self.stop_audio_flag.is_set():
                print("🛑 Audio playback stopped via flag.")
                self.stop_audio_flag.clear()
                self.clear_queue()
                continue

            filepath = self.audio_file_queue.get()
            if filepath is None:  # shutdown signal
                break

            self._is_playing = True
            try:
                print(f"🔊 Playing: {filepath}")
                data, samplerate = sf.read(filepath, dtype='float32')
                sd.play(data, samplerate)
                sd.wait()
            except Exception as e:
                print(f"❌ Audio playback error: {e}")
            finally:
                self._is_playing = False
    

    def stop_audio_playback(self):
        """External trigger to stop current playback and clear queue."""
        print("🛑 External stop_audio_playback() called.")
        self.stop_audio_flag.set()
