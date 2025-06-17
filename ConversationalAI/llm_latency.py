import time
import numpy as np
from llama_cpp import Llama

REPS = 5
user_prompt = "What are safe exercises for older adults recovering from surgery?"
prompt = (
    "<|system|>\nYou are a helpful voice assistant named Arise. "
    "Respond clearly, naturally, and concisely as if spoken aloud. "
    "No markdown or formatting.</s>\n"
    f"<|user|>\n{user_prompt}</s>\n<|assistant|>"
)

# Load model
print("🔄 Loading LLaMA model...")
llm = Llama(
    model_path="models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    n_ctx=256,
    n_threads=4,
    verbose=False
)

# Run benchmark
latencies = []
print(f"\n🧠 Running LLM test prompt for {REPS} repetitions...\n")

for i in range(REPS):
    start = time.time()
    response = llm(prompt, max_tokens=250, temperature=0.4)
    end = time.time()
    latency = end - start
    latencies.append(latency)
    print(f"Run {i+1}: {latency:.4f} sec")

# Report results
mean = np.mean(latencies)
std = np.std(latencies)
print(f"\n📊 LLM Generation Latency over {REPS} runs: avg = {mean:.4f}s, std = {std:.4f}s")

# Print last response for reference
print("\n🤖 Sample LLM output:\n")
print(response['choices'][0]['text'].strip())
