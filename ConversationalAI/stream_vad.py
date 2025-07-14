import sounddevice as sd
import json
import os
import re
import time
import tempfile
import threading
import queue
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import simpleaudio as sa
from word2number import w2n 

import random
import queue
import cv2

from vosk import Model, KaldiRecognizer
from llama_cpp import Llama
from kokoro import KPipeline

from YOLO_Pose.yolo_threaded import thread_main
from YOLO_Pose.shared_data import SharedState

from kokoro_onnx import Kokoro

#-------------
pose_thread = None
pose_running = threading.Event()

pose_shared_state = SharedState()

global awaiting_exercise_ready
awaiting_exercise_ready = False

def start_pose_detection():
    global pose_thread
    global my_queue
    if not pose_shared_state.running.is_set():
        pose_shared_state.running.set()
        pose_thread = threading.Thread(target=thread_main, args=(pose_shared_state,), kwargs={'thread_queue':my_queue}, daemon=True)
        pose_thread.start()

def stop_pose_detection():
    if pose_shared_state.running.is_set():
        pose_shared_state.running.clear()
        if pose_thread:
            pose_thread.join(timeout=1)
            print("🛑 Pose detection stopped.")


#--------------

# ------------------
# Configurable Paths
# ------------------
VOSK_MODEL_PATH = "models/vosk-small"

# ------------------
# Load Models
# ------------------
print("\nLoading Vosk Model\n")
vosk_model = Model(VOSK_MODEL_PATH)

print("\nLoading Llama Model\n")
llm = Llama(model_path="models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf", n_ctx=256, verbose=False, n_threads=4, n_batch=64)

print("\nLoading TTS Model\n")

"""
tts_pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M',lang_code='a')
"""
kokoro = Kokoro(
    model_path="models/kokoro-v1.0.fp16.onnx",
    voices_path="models/voices-v1.0.bin"
)

'''
play pre-made audio files
'''

def play_audio_file(filepath):
    data, samplerate = sf.read(filepath, dtype='float32')
    sd.play(data, samplerate)
    sd.wait()  # wait until playback finishes


# ------------------
# speak onnx threaded
# ------------------

audio_queue = queue.Queue()

def split_text(text, max_words=8):
    text = re.sub(r'\s+', ' ', text.strip())
    raw_sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []

    for sentence in raw_sentences:
        words = sentence.strip().split()
        if len(words) <= max_words:
            chunks.append(sentence.strip())
        else:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk)
    return chunks

# ⏱️ Shared variable to store the initial call time
stream_start_time = None
first_chunk_played = threading.Event()

def producer(text, voice="af_heart", speed=1.15, lang="en-us"):
    for chunk in split_text(text):
        try:
            samples, sr = kokoro.create(chunk, voice=voice, speed=speed, lang=lang)
            audio_queue.put((samples, sr))
        except Exception as e:
            print(f"❌ Producer error: {e}")
    audio_queue.put((None, None))  # signal end

def consumer(prebuffer_count=2):
    try:
        # Wait until we have enough buffered chunks or see end of producer
        while True:
            if audio_queue.qsize() >= prebuffer_count:
                break
            # Check if the end signal is already in the queue
            with audio_queue.mutex:
                if any(x[0] is None for x in audio_queue.queue):
                    break
            time.sleep(0.01)

        with sd.OutputStream(samplerate=24000, channels=1, dtype='float32') as stream:
            while True:
                samples, sr = audio_queue.get()
                if samples is None:
                    break

                if not first_chunk_played.is_set():
                    latency = time.time() - stream_start_time
                    print(f"First chunk playback latency: {latency:.4f} seconds")
                    first_chunk_played.set()

                samples_np = samples.astype(np.float32).reshape(-1, 1)
                stream.write(samples_np)
    except Exception as e:
        print(f"❌ Consumer error: {e}")



def speak(text, voice="af_heart", speed=1.0, lang="en-us"):
    global stream_start_time
    stream_start_time = time.time()
    first_chunk_played.clear()

    producer_thread = threading.Thread(target=producer, args=(text, voice, speed, lang))
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

# ------------------
# speak 82m threaded - old
# ------------------

""" 
def speak(text):
    audio_queue = queue.Queue()
    first_chunk_time = None

    def producer():
        nonlocal first_chunk_time
        try:
            generator = tts_pipeline(text, voice='af_heart', speed=1.0)
            for idx, (_, _, audio) in enumerate(generator):
                if idx == 0:
                    first_chunk_time = time.time()
                # Convert to numpy and enqueue immediately
                audio_np = audio.detach().cpu().numpy()
                audio_int16 = (audio_np * 32767).astype(np.int16)
                audio_queue.put(audio_int16)
            audio_queue.put(None)  # End signal
        except Exception as e:
            print(f"❌ Error in TTS generation: {e}")
            audio_queue.put(None)

    def consumer():
        try:
            while True:
                audio_int16 = audio_queue.get()
                if audio_int16 is None:
                    break
                # Play directly with sounddevice
                sd.play(audio_int16, samplerate=24000)
                sd.wait()
        except Exception as e:
            print(f"❌ Error in playback: {e}")

    start_time = time.time()

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()

    if first_chunk_time:
        print(f"TTS first chunk latency: {first_chunk_time - start_time:.4f} seconds")

    consumer_thread.join()
"""

def generate_reply(prompt):
    llm_start_time = time.time()
    output = llm(f"<|system|>\nYou are a helpful voice assistant named arise. Respond clearly, naturally, and concisely as if spoken aloud. No markdown or formatting.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>", max_tokens=250, temperature=0.4)
    llm_end_time = time.time()
    print(f"LLM processing time: {llm_end_time - llm_start_time:.4f} seconds")
    return output['choices'][0]['text'].strip()

def clean_text_for_tts(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'/\w+\b', '', text)
    text = re.sub(r'[\[\]\{\}\(\)\|\\/]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(\d)\s*-\s*(\d)',r'\1 to \2', text)
    text = re.sub(r'-',' ', text)
    return text.strip()

def contains_keyword(text, keywords):
    words = text.lower().split()
    for word in words:
        if word in keywords:
            return word
    return None
# New: Basic exercise parser
exercise_keywords = {
    "bicep": {
        "name": "bicep curl",
        "muscle": "short head of the bicep",
        "instruction": "Extend your arm straight, bend at the elbow to lift your hand toward your shoulder, then slowly return."
    },
    "curls": {
        "name": "bicep curl",
        "muscle": "short head of the bicep",
        "instruction": "Extend your arm straight, bend at the elbow to lift your hand toward your shoulder, then slowly return."
    },
    "hammer": {
        "name": "hammer curl",
        "muscle": "long head of the bicep",
        "instruction": "Extend your arm straight, bend at the elbow to lift your hand toward your shoulder, then slowly return."
    },
    "squat": {
        "name": "squat",
        "muscle": "quadriceps, glutes, hamstrings",
        "instruction": "Stand with feet shoulder-width apart, bend your knees and hips to lower your body, then return to standing."
    },
    #adding squad which is a common mistake from vosk when trying to say squat if no explicit emphasis on the t
    "squad": {
        "name": "squat",
        "muscle": "quadriceps, glutes, hamstrings",
        "instruction": "Stand with feet shoulder-width apart, bend your knees and hips to lower your body, then return to standing."
    },
    # add more exercises here as needed
}

def parse_exercise_intent(text):
    text = text.lower()
    found_exercise = None
    reps = None

    for key in exercise_keywords:
        if key in text:
            found_exercise = key
            break

    match = re.search(r'\b(\d+)\b', text)
    if match:
        reps = int(match.group(1))
    else:
        try:
            cleaned = re.sub(r'[^\w\s]', '', text)
            reps = w2n.word_to_num(cleaned)
        except:
            reps = None

    return found_exercise, reps

def process_user_input(user_text):
    global awaiting_exercise_ready
    print(f"\nprocessed from You: {user_text}")

    exercise, reps = parse_exercise_intent(user_text)
    if exercise:
        details = exercise_keywords[exercise]
        if reps:
            pose_shared_state.set_value("reset_exercise",True)
            pose_shared_state.set_value("current_exercise",details['name'])
            pose_shared_state.set_value("adjust_reps_threshold",reps)
            pose_shared_state.set_value("exercise_paused",False)
            awaiting_exercise_ready = True
            speak(f"Okay, starting {reps} reps of {details['name']}. Let me know when you're ready.")
        else:
            speak(f"a {details['name']} works the {details['muscle']}. Here's how: {details['instruction']}")
        return
    
    prebuilt_responses = ["let me think about that", "Give me a second for this one", "Just a moment", "Good question, one second", "working right on it"]
    speak(random.choice(prebuilt_responses))

    reply = generate_reply(user_text)
    clean_reply = clean_text_for_tts(reply)
    print(f"\LLM Response: {clean_reply}")
    speak(clean_reply)

bad_form_dict = {
    "KEEP_BACK_STRAIGHT": "tts_cache/keep_your_back_straight_and_avoid_roundi.wav",
    "KEEP_ELBOWS_CLOSE_TO_BODY": "tts_cache/tuck_your_elbows_in_close_to_your_sides_.wav",
    "KEEP_ARMS_STRAIGHT": "tts-cache/fully_extend_your_arms_without_locking_y.wav",
    "KEEP_HEAD_UP": "tts_cache/lift_your_head_and_look_slightly_ahead._.wav",
    "KEEP_HIPS_BACK_SQUAT": "tts_cache/push_your_hips_back_as_if_you're_sitting.wav",
    "KEEP_KNEES_OVER_TOES_SQUAT": "tts_cache/make_sure_your_knees_stay_aligned_over_y.wav",
    
    "KEEP_ELBOWS_UNDER_SHOULDERS": "tts_cahce/position_your_elbows_directly_under_your.wav",
    "KEEP_ARMS_LEVEL": "tts_cache/raise_or_lower_your_arms_to_match_each_o.wav",
    "KEEP_FEET_SHOULDER_WIDTH": "tts_cache/set_your_feet_shoulder-width_apart_to_cr.wav",
    "KEEP_SHOULDERS_LEVEL": "tts_cache/keep_both_shoulders_at_the_same_height._.wav",
    "KEEP_SHOULDERS_ABOVE_HIPS": "tts_cache/lift_your_upper_body_so_your_back_stays_.wav",
    "KEEP_KNEES_POINTED_OUT": "tts_cache/angle_your_knees_slightly_outward,_in_li.wav",
    "MOVE_INTO_CAMERA_FRAME" : "tts_cache/you_are_out_of_the_camera_frame,_for_acc.wav" ,
    "MOVE_AWAY_FROM_CAMERA" : "tts_cache/you_are_too_close_to_the_camera_for_this.wav" ,
    "FACE_CAMERA" : "tts_cache/please_be_front_facing_to_the_camera_for.wav",
}

#------------------
#better keyword parsing, looks for 'arise' anywhere 
# in buffer and returns what is in the buffer after the keyword
#------------------

def extract_after_keyword(text, keyword_list):
    """
    Returns the portion of text after the first keyword match in keyword_list.
    """
    text = text.lower()
    for keyword in keyword_list:
        pattern = re.compile(rf"\b{re.escape(keyword.lower())}\b\s+(.*)")
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None

# ------------------
# Streaming Chatbot Loop
# ------------------
def chatbot_loop():
    #default values for shared state data 
    pose_shared_state.set_value("exercise_completed",False)
    pose_shared_state.set_value("reps",-1)
    pose_shared_state.set_value("reps_threshold",-1)
    pose_shared_state.set_value("ask_adjust_rom",False)
    pose_shared_state.set_value("bad_form",[])
    pose_shared_state.set_value("rom",[])
    pose_shared_state.set_value("angles",{})

    pose_shared_state.set_value("adjust_reps_threshold",-1)
    pose_shared_state.set_value("exercise_paused",False)
    pose_shared_state.set_value("current_exercise",None)
    pose_shared_state.set_value("reset_exercise",False)
    pose_shared_state.set_value("adjust_rom",False)

    recognizer = KaldiRecognizer(vosk_model, 16000)
    recognizer.SetWords(True)
    sentence_buffer = ""
    stop_keywords = ["exit", "quit", "stop", "end"]
    pause_keywords = [
    "pause", "hold", "wait", "break", "timeout", "hang on",
    "hold on", "give me a moment", "need a second", "take a breather"
    ]
    prompt_keywords = ["arise","rise"]
    restart_keywords = ["restart", "redo","do over", "start over"]
    agree_keywords = ['yes','yeah','of course']
    continue_keywords = ["continue","unpause","start over"]
    new_exercise_keywords = ["new exercise","switch exercise","different exercise"]

    start_keywords = ['ready','start','begin','go']


    # VVVV testing placeholders VVVV
    #bad_form_list = []
    #global adj_rom
    #adj_rom = True
    global awaiting_rom_confirmation
    awaiting_rom_confirmation = False
    global awaiting_pause_confirmation
    awaiting_pause_confirmation = False
    global awaiting_new_exercise
    awaiting_new_exercise = False
    global form_inc
    form_inc = 0

    # Add these VAD variables at the top of your file, after your existing globals
    VAD_THRESHOLD = 0.025  # Adjust based on your microphone sensitivity
    SILENCE_DURATION = 1.5  # 1.5 seconds of silence before stopping processing
    global vad_active, last_speech_time
    vad_active = False
    last_speech_time = 0.0

    # Replace your existing callback function with this VAD-enhanced version
    def callback(indata, frames, time_info, status):
        global awaiting_rom_confirmation, adj_rom, awaiting_pause_confirmation, awaiting_new_exercise, form_inc
        global awaiting_exercise_ready
        global vad_active, last_speech_time  # Add VAD globals
        
        # === VAD LOGIC ===
        # Calculate RMS for voice activity detection
        samples = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(samples**2))
        current_time = time.time()
        
        # Add RMS smoothing to reduce false triggers
        if not hasattr(callback, 'rms_history'):
            callback.rms_history = []
        
        callback.rms_history.append(rms)
        if len(callback.rms_history) > 3:
            callback.rms_history.pop(0)
        
        # Use smoothed RMS for more stable VAD
        avg_rms = np.mean(callback.rms_history)
        
        # Check if there's voice activity (using smoothed RMS)
        if avg_rms > VAD_THRESHOLD:
            if not vad_active:
                vad_active = True
                print(f"\nVoice detected (RMS: {rms:.4f}, Avg: {avg_rms:.4f})")
            last_speech_time = current_time
        else:
            # Check if we should turn off VAD due to silence
            if vad_active and (current_time - last_speech_time) > SILENCE_DURATION:
                vad_active = False
                print(f"\nVoice activity ended (silence: {current_time - last_speech_time:.1f}s)")
        
        # Only process audio if VAD is active
        if not vad_active:
            return  # Skip all processing when no voice detected
        
        # === YOUR EXISTING LOGIC CONTINUES UNCHANGED ===
        current_exercise = pose_shared_state.get_value("current_exercise")

        finish_exercise = pose_shared_state.get_value("exercise_completed")

        if finish_exercise:
            stop_pose_detection()
            play_audio_file('tts_cache/great_work,_finishing_exercise.wav')
            finish_exercise = False
            pose_shared_state.set_value("exercise_completed",finish_exercise)

        bad_form_list = pose_shared_state.get_value('bad_form')




        while (len(bad_form_list)>0 ):
            play_audio_file(bad_form_dict[bad_form_list[form_inc]])
            bad_form_list = pose_shared_state.get_value('bad_form')
            form_inc = form_inc + 1
            length = len(bad_form_list)
            if form_inc >=length:
                form_inc = 0
            bad_form_list.clear()
            


        adj_rom = pose_shared_state.get_value('ask_adjust_rom')

        if adj_rom and not awaiting_rom_confirmation:
                play_audio_file('tts_cache/i_noticed_your_range_of_motion_may_need_.wav')
                awaiting_rom_confirmation = True
                return

        nonlocal sentence_buffer
        if recognizer.AcceptWaveform(bytes(indata)):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if not text:
                return

            print("You (live audio):", text)
            sentence_buffer += " " + text

            # ... rest of your existing logic remains exactly the same ...
            if awaiting_pause_confirmation:
                if any(kw in sentence_buffer for kw in continue_keywords):
                    pose_shared_state.set_value("exercise_paused",False)
                    play_audio_file('tts_cache/ok,_starting_back_up.wav')
                    sentence_buffer = ""
                    awaiting_pause_confirmation=False
                elif any(kw in sentence_buffer for kw in new_exercise_keywords):
                    awaiting_new_exercise = True
                    play_audio_file('tts_cache/ok,what_exercise_would_you_like_to_perform_.wav')
                    sentence_buffer = ""
                    awaiting_pause_confirmation=False
                    pose_shared_state.set_value("exercise_paused",False)
                elif any(kw in sentence_buffer for kw in stop_keywords):
                    pose_shared_state.set_value("exercise_paused",False)
                    pose_shared_state.set_value("exercise_completed",False)
                    pose_shared_state.set_value("current_exercise",None)
                    stop_pose_detection()
                    play_audio_file('tts_cache/finishing_exercise.wav')
                    awaiting_pause_confirmation=False
                sentence_buffer = ""
                return

            if awaiting_new_exercise:
                exercise, reps = parse_exercise_intent(sentence_buffer.strip())
                if exercise:
                    details = exercise_keywords[exercise]
                    if reps:
                        pose_shared_state.set_value("reset_exercise",True)
                        pose_shared_state.set_value("current_exercise",details['name'])
                        pose_shared_state.set_value("adjust_reps_threshold",reps)
                        pose_shared_state.set_value("exercise_paused",False)
                        awaiting_exercise_ready = True
                        speak(f"Okay, starting {reps} reps of {details['name']}. Let me know when you're ready.")
                    else:
                        speak(f"a {details['name']} works the {details['muscle']}. Here's how: {details['instruction']}")
                sentence_buffer = ""
                awaiting_new_exercise=False
                return

            if awaiting_rom_confirmation:
                if any(kw in sentence_buffer for kw in agree_keywords):
                    play_audio_file('tts_cache/okay,_adjusting_your_range_of_motion..wav')
                    pose_shared_state.set_value("adjust_rom", True)
                else:
                    
                    play_audio_file('tts_cache/understood,_keeping_current_range..wav')
                    pose_shared_state.set_value("adjust_rom", False)
                awaiting_rom_confirmation = False
                adj_rom=False
                sentence_buffer = ""
                return         

            if(awaiting_exercise_ready):
                print('waiting for go keyword')
                if any(kw in sentence_buffer for kw in start_keywords):
                    start_pose_detection()
                    awaiting_exercise_ready=False
                    sentence_buffer =""
                    return

            parsed_prompt = extract_after_keyword(sentence_buffer, prompt_keywords)

            if parsed_prompt:
                if any(kw in parsed_prompt for kw in (pause_keywords + stop_keywords)) and (current_exercise is not None):
                    pose_shared_state.set_value("exercise_paused", True)
                    play_audio_file('tts_cache/okay,_pausing._let_me_know_if_you_want_t.wav')
                    awaiting_pause_confirmation = True
                    sentence_buffer = ""
                    return

                if any(kw in parsed_prompt for kw in stop_keywords) and (current_exercise is None):
                    play_audio_file('tts_cache/okay,_stopping_the_program_now..wav')
                    os._exit(0)

                if any(kw in parsed_prompt for kw in restart_keywords) and (current_exercise is not None):
                    play_audio_file('tts_cache/okay, restarting exercise..wav')
                    pose_shared_state.set_value("reset_exercise", True)
                    sentence_buffer = ""
                    return

                if len(parsed_prompt.split()) >= 3:
                    process_user_input(parsed_prompt)
                    sentence_buffer = ""
            else:
                sentence_buffer = ""


    print("Listening... Speak naturally. Say 'stop' to end.")
    with sd.RawInputStream(samplerate=16000, blocksize=1024, dtype='int16',
                           channels=1, callback=callback):
        while True:
            time.sleep(0.1)
            #print('sleeping in converesational thread.')
                           


# ------------------
# Start Chat
# ------------------
if __name__ == "__main__":
    global my_queue
    play_audio_file('tts_cache/hello_this_is_the_arise_system,_how_may_.wav')
    my_queue = queue.Queue()
    conversational_thread = threading.Thread(target=chatbot_loop, daemon=True)
    conversational_thread.start()
    
    # Main thread loop - handle GUI
    window_open = False
    while True:
        exercise_name = pose_shared_state.get_value('current_exercise')
        #print('mainthread')
        if (exercise_name is not None):
            try:
                frame = my_queue.get(timeout=2)
            except queue.Empty:
                continue
            if frame is None:
                print("no frame beans")
            else:
                #print("got frame")
                cv2.imshow("example text", frame)
                window_open=True
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
        else:
            if window_open:
                print("destroying window")
                cv2.destroyAllWindows()
                print("window destroyed")
                window_open=False
            
                
            # Testing for setting exercise type
            """
            elif key & 0xFF == ord('b'):
                current_exercise = 'bicep curl'
                reps = 0
                shared_data.set_value('current_exercise', current_exercise)
                shared_data.set_value('reps', reps)
                start_time = time.perf_counter()
                reset_bad_form_times()
            elif key & 0xFF == ord('s'):
                current_exercise = 'squat'
                reps = 0
                shared_data.set_value('current_exercise', current_exercise)
                shared_data.set_value('reps', reps)
                start_time = time.perf_counter()
                reset_bad_form_times()
            elif key & 0xFF == ord('a'):
                current_exercise = 'arm raise'
                reps = 0
                shared_data.set_value('current_exercise', current_exercise)
                shared_data.set_value('reps', reps)
                start_time = time.perf_counter()
                reset_bad_form_times()
            elif key & 0xFF == ord('l'):
                current_exercise = 'lunge'
                reps = 0
                shared_data.set_value('current_exercise', current_exercise)
                shared_data.set_value('reps', reps)
                start_time = time.perf_counter()
                reset_bad_form_times()
            """
