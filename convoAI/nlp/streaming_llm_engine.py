from llama_cpp import Llama
import time
import re
import threading
import queue
from collections import deque

#---------------------------------------------------------------------------------------------------------
#   Streaming LLM text generation with sentence-level streaming to TTS
#
#   Generates text token by token, accumulates into sentences, and streams complete
#   sentences to TTS engine for immediate audio playback while generation continues
#---------------------------------------------------------------------------------------------------------

class StreamingLLMEngine:
    def __init__(self, model_path="models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"):
        print("\n🧠 Loading Streaming Llama Model\n")
        self.model = Llama(
            model_path=model_path,
            n_ctx=256,
            verbose=False,
            n_threads=4,
            n_batch=64
        )
        
        # Sentence detection patterns
        self.sentence_endings = re.compile(r'[.!?]+(?:\s|$)')
        self.buffer = ""
        self.sentence_queue = queue.Queue()
        
    def generate_reply_streaming(self, prompt: str, tts_engine=None) -> str:
        """
        Generate reply with streaming to TTS engine.
        Returns the complete response text.
        """
        print("🧠 Starting streaming LLM generation")
        start_time = time.time()
        
        # Clear any previous state
        self.buffer = ""
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except queue.Empty:
                break
        
        # Start TTS consumer thread if TTS engine provided
        tts_consumer_thread = None
        if tts_engine:
            tts_consumer_thread = threading.Thread(
                target=self._tts_consumer,
                args=(tts_engine,),
                daemon=True
            )
            tts_consumer_thread.start()
        
        # Generate streaming response
        full_response = ""
        try:
            stream = self.model(
                f"<|system|>\nYou are a helpful voice assistant named arise. Respond clearly, naturally, and concisely as if spoken aloud. No markdown or formatting.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>",
                max_tokens=250,
                temperature=0.4,
                stream=True
            )
            
            for output in stream:
                token = output['choices'][0]['text']
                full_response += token
                self._process_token(token)
                
        except Exception as e:
            print(f"❌ Streaming generation error: {e}")
            
        # Process any remaining text in buffer
        self._flush_buffer()
        
        # Signal end of generation to TTS consumer
        self.sentence_queue.put(None)
        
        # Wait for TTS to finish if consumer thread exists
        if tts_consumer_thread:
            tts_consumer_thread.join(timeout=10)
        
        generation_time = time.time() - start_time
        print(f"🧠 Total LLM generation time: {generation_time:.2f} sec")
        print(f"🧠 Complete response: {full_response.strip()}")
        
        return full_response.strip()
    
    def _process_token(self, token: str):
        """Process each token and detect sentence boundaries."""
        self.buffer += token
        
        # Check for sentence endings
        sentences = self._extract_complete_sentences(self.buffer)
        
        for sentence in sentences:
            cleaned_sentence = self._clean_sentence(sentence)
            if cleaned_sentence:
                print(f"📝 Sentence ready: '{cleaned_sentence}'")
                self.sentence_queue.put(cleaned_sentence)
        
        # Update buffer with remaining text
        self.buffer = self._get_remaining_text(self.buffer)
    
    def _extract_complete_sentences(self, text: str) -> list:
        """Extract complete sentences from text buffer."""
        sentences = []
        
        # Find all sentence endings
        matches = list(self.sentence_endings.finditer(text))
        
        if matches:
            last_end = 0
            for match in matches:
                sentence_end = match.end()
                sentence = text[last_end:sentence_end].strip()
                if sentence:
                    sentences.append(sentence)
                last_end = sentence_end
        
        return sentences
    
    def _get_remaining_text(self, text: str) -> str:
        """Get text remaining after extracting complete sentences."""
        matches = list(self.sentence_endings.finditer(text))
        if matches:
            last_match = matches[-1]
            return text[last_match.end():].strip()
        return text
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean sentence for TTS processing."""
        # Remove extra whitespace
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        # Skip very short or empty sentences
        if len(sentence) < 3:
            return ""
            
        # Remove markdown and special characters (similar to text_utils.clean_text_for_tts)
        sentence = re.sub(r'\*\*(.*?)\*\*', r'\1', sentence)
        sentence = re.sub(r'\*(.*?)\*', r'\1', sentence)
        sentence = re.sub(r'`{1,3}.*?`{1,3}', '', sentence)
        sentence = re.sub(r'<[^>]+>', '', sentence)
        sentence = re.sub(r'[\[\]\{\}\(\)\|\\/]', '', sentence)
        
        return sentence.strip()
    
    def _flush_buffer(self):
        """Process any remaining text in buffer as a final sentence."""
        if self.buffer.strip():
            cleaned = self._clean_sentence(self.buffer)
            if cleaned:
                print(f"📝 Final sentence: '{cleaned}'")
                self.sentence_queue.put(cleaned)
        self.buffer = ""
    
    def _tts_consumer(self, tts_engine):
        """Consumer thread that feeds sentences to TTS engine."""
        print("🔊 TTS consumer thread started")
        sentence_count = 0
        
        try:
            while True:
                sentence = self.sentence_queue.get()
                
                # None signals end of generation
                if sentence is None:
                    print("🔊 TTS consumer received end signal")
                    break
                
                sentence_count += 1
                print(f"🔊 TTS processing sentence {sentence_count}: '{sentence}'")
                
                # Send sentence to TTS engine
                # Note: We use a separate thread so TTS can play while we wait for next sentence
                tts_thread = threading.Thread(
                    target=tts_engine.speak,
                    args=(sentence,),
                    daemon=True
                )
                tts_thread.start()
                
                # Small delay to allow TTS to start before next sentence
                # This prevents audio queue overflow
                time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ TTS consumer error: {e}")
        
        print(f"🔊 TTS consumer finished. Processed {sentence_count} sentences")

    # Backward compatibility method
    def generate_reply(self, prompt: str) -> str:
        """
        Non-streaming version for backward compatibility.
        """
        return self.generate_reply_streaming(prompt, tts_engine=None)