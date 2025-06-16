import sounddevice as sd
from kokoro_onnx import Kokoro
import time 
from kokoro import KPipeline



from kokoro import KPipeline
import sounddevice as sd
import soundfile as sf
import torch
import re
import time
import numpy as np
    

# Initialize the pipeline
pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M',lang_code='a')


# Use this function to play audio in real time with latency tracking
def play_kokoro_stream(pipeline, text, voice='af_heart', speed=1.0, split_pattern=r'\n+'):
    Kokoro_base_start = time.time()
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)

    for i, (gs, ps, audio) in enumerate(generator):
        print(f"\n🔊 Chunk {i}")
        print(f"Text: {gs}")
        print(f"Phonemes: {ps}")

        # Convert PyTorch tensor to NumPy array
        audio_np = audio.detach().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)  # Optional: convert to int16

        # Play the audio
        kokoro_base_finish = time.time()
        print(f"Kokoro base-82m latency :{kokoro_base_finish-Kokoro_base_start:.4f} seconds")
        sd.play(audio_int16, samplerate=24000)
        sd.wait()
        

text = "Older adults recovering from surgery often require guided exercise "\
        "and rehabilitation at home to regain strength and mobility. However," \
        "ensuring they perform exercises correctly, safely, and consistently is" \
        "challenging without professional supervision. AI-powered at-home" \
        "exercise systems aim to fill this gap by using multiple modalities –" \
        "computer vision, voice interaction, and wearable sensors – to mon-" \
        "itor performance and coach the user in real time. Recent advances"\
        "in edge computing (for example, NVIDIA Jetson Nano) make it feasible to" \
        "run sophisticated pose-tracking and voice assistants locally, pre-serving privacy and reducing latency. This report reviews the state" \
        "of the art in these areas and outlines a development roadmap. We" \
        "focus on: (1) real-time pose estimation methods (2) Voice-based interface"
# Run it
print('starting')
for i in range(10):
    play_kokoro_stream(pipeline, text)





"""

kokoro = Kokoro(model_path="ConversationalAI/quantized_tts/kokoro-v1.0.fp16.onnx", voices_path="ConversationalAI/quantized_tts/voices-v1.0.bin")



for i in range(10):
    starttime = time.time()
    samples, sample_rate = kokoro.create(
        "Older adults recovering from surgery often require guided exercise "
        "and rehabilitation at home to regain strength and mobility. However," \
        "ensuring they perform exercises correctly, safely, and consistently is" \
        "challenging without professional supervision. AI-powered at-home" \
        "exercise systems aim to fill this gap by using multiple modalities –" \
        "computer vision, voice interaction, and wearable sensors – to mon-" \
        "itor performance and coach the user in real time. Recent advances"
        "in edge computing (for example, NVIDIA Jetson Nano) make it feasible to" \
        "run sophisticated pose-tracking and voice assistants locally, pre-serving privacy and reducing latency. This report reviews the state" \
        "of the art in these areas and outlines a development roadmap. We" \
        "focus on: (1) real-time pose estimation methods (2) Voice-based interface", voice = "af_heart", speed = 1.0
    )
    procendtime = time.time()

    sd.play(samples, sample_rate)
    playtime = time.time()
    print(f"ONNX latency : {procendtime - starttime:.4f} seconds")
    sd.wait()
"""