import subprocess
import time
import os

# Optional: Ensure log folder and CSV exist
os.makedirs("latency_logs", exist_ok=True)
log_file = "latency_logs/all_latency_data.csv"

# Create header if file doesn't exist
if not os.path.isfile(log_file):
    with open(log_file, "w") as f:
        f.write("component,timestamp,latency,extra\n")

# Scripts to run in sequence
scripts = [
    ("llm_latency.py", "🔠 LLM Latency Test"),
    ("onnx_threaded.py", "🔊 TTS Latency Test"),
    ("vosk_chunksize.py", "🎙️ Live STT Latency Test"),
    ("vosk_latency.py", "📼 Offline STT Latency Test")
]

# Run each script with logging
for script, label in scripts:
    print(f"\n▶️ {label}")
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script}: {e}")
    print("⏳ Waiting briefly before next test...\n")
    time.sleep(3)

print("\n✅ All tests completed. Results logged to latency_logs/all_latency_data.csv")
