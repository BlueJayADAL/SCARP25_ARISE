from llama_cpp import Llama
import time

class LLMEngine:
    def __init__(self, model_path="models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"):
        print("\n🧠 Loading Llama Model\n")
        self.model = Llama(
            model_path=model_path,
            n_ctx=256,
            verbose=False,
            n_threads=4,
            n_batch=64
        )

    def generate_reply(self, prompt: str) -> str:
        print("user prompted llm\n")
        start_time = time.time()
        output = self.model(
            f"<|system|>\nYou are a helpful voice assistant named arise. Respond clearly, naturally, and concisely as if spoken aloud. No markdown or formatting.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>",
            max_tokens=250,
            temperature=0.4
        )
        print(f"🧠 LLM response time: {time.time() - start_time:.2f} sec")
        print(f"response: {output['choices'][0]['text'].strip()}")
        return output['choices'][0]['text'].strip()
