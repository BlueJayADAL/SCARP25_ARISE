import time
import numpy as np
from llama_cpp import Llama

import pandas as pd
import uuid
from datetime import datetime
import os

SESSION_ID = uuid.uuid4().hex[:8]
log_buffer = []
LOG_PATH = "latency_logs/llm_latency-test.csv"
os.makedirs("latency_logs", exist_ok=True)

#------
#configuration
user_prompt = "What are safe exercises for older adults recovering from surgery?"
ctx = 256
tkns = 250
temp = 0.000
#------

REPS = 5
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
    n_ctx=ctx,
    n_threads=4,
    verbose=False
)

# Run benchmark
latencies = []
print(f"\n🧠 Running LLM test prompt for {REPS} repetitions...\n")

for i in range(REPS):
    start = time.time()
    response = llm(prompt, max_tokens=250, temperature=0.0000)
    end = time.time()
    latency = end - start
    latencies.append(latency)
    print(f"Run {i+1}: {latency:.4f} sec")
    log_buffer.append({
        "run_id": SESSION_ID,
        "component": "LLM",
        "timestamp": datetime.now().isoformat(),
        "latency": round(latency, 4),
        "temperature": temp,
        "max_tokens": tkns,
        "n_ctx":ctx,
        "prompt":user_prompt,
        "response":response['choices'][0]['text'].strip(),
        "model_path": "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
    })

# Report results
mean = np.mean(latencies)
std = np.std(latencies)
print(f"\n📊 LLM Generation Latency over {REPS} runs: avg = {mean:.4f}s, std = {std:.4f}s")

# Print last response for reference
print("\n🤖 Sample LLM output:\n")
print(response['choices'][0]['text'].strip())

df = pd.DataFrame(log_buffer)
if not os.path.isfile(LOG_PATH):
    df.to_csv(LOG_PATH, index=False)
else:
    df.to_csv(LOG_PATH, mode="a", header=False, index=False)

print(f"\n📁 Logged {len(df)} entries to {LOG_PATH} under run_id {SESSION_ID}")
