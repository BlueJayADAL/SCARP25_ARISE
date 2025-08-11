import sounddevice as sd
import time
import json
import webrtcvad

from vosk import Model, KaldiRecognizer

#---------------------------------------------------------------------------------------------------------
#   Voice-activity-detection and loading as well as processing of VOSK Speech to text model
#
#   Makes sure system isn't producing audio through file playback or text to speech to prevent feedback loop
#
#   Once a finalized thought or scentence goes through the VOSK model/user stops talking, text is sent to be parsed and processed  by chat manager
#---------------------------------------------------------------------------------------------------------

class VADController:
    def __init__(self, model_path, on_text_callback, sample_rate=16000, audio_guard_funcs = None):
        """
        :param model_path: Path to the Vosk model
        :param on_text_callback: Function to call when a sentence is recognized
        :param sample_rate: Audio sample rate
        """
        print("\n🎤 Loading Vosk Model\n")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(True)

        self.vad = webrtcvad.Vad(3)  # Aggressiveness level: 0–3
        self.sample_rate = sample_rate
        self.blocksize = 1024
        self.frame_size = 960  # 480 samples * 2 bytes/sample

        self.silence_duration = 1.5
        self.vad_active = False
        self.last_speech_time = 0
        self.sentence_buffer = ""

        self.on_text_callback = on_text_callback
        self.audio_guard_funcs = audio_guard_funcs or []

    def start(self):
        """Begin listening on the microphone stream."""
        print("🎧 Starting VAD listener...")
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            dtype='int16',
            channels=1,
            callback=self._callback
        ):
            while True:
                time.sleep(0.1)

    def _callback(self, indata, frames, time_info, status):

        for guard in self.audio_guard_funcs:
            if guard():  # If TTS or audio is active
                #print(f"🔇 Skipping callback — audio output active:{guard}")
                return
    
        raw_bytes = bytes(indata)
        current_time = time.time()

        speech_detected = False
        for i in range(0, len(raw_bytes) - self.frame_size + 1, self.frame_size):
            frame = raw_bytes[i:i + self.frame_size]
            if self.vad.is_speech(frame, self.sample_rate):
                speech_detected = True
                break

        if speech_detected:
            if not self.vad_active:
                print("🎙️ Voice detected")
            self.vad_active = True
            self.last_speech_time = current_time
        else:
            if self.vad_active and (current_time - self.last_speech_time) > self.silence_duration:
                self.vad_active = False
                print(f"🔇 Voice activity ended (silence: {current_time - self.last_speech_time:.1f}s)")

        if self.recognizer.AcceptWaveform(raw_bytes):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                print(f"🗣️ You said: {text}")
                self.sentence_buffer += " " + text
                self.on_text_callback(text)
                self.sentence_buffer = " "

                if not self.vad_active:
                    self._flush_buffer()

    def _flush_buffer(self):
        cleaned = self.sentence_buffer.strip()
        if cleaned:
            self.on_text_callback(cleaned)
        self.sentence_buffer = ""
