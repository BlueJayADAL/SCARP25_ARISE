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
tts_pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M',lang_code='a')

# ------------------
# Utility Functions
# ------------------
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
    print(f"\n\U0001f9cd You: {user_text}")

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
    print(f"\U0001f916 Bot: {clean_reply}")
    speak(clean_reply)

bad_form_dict = {
    "KEEP_BACK_STRAIGHT": "Keep your back straight and avoid rounding. Engage your core and maintain a neutral spine throughout the movement.",
    "KEEP_ELBOWS_CLOSE_TO_BODY": "Tuck your elbows in close to your sides to protect your shoulders and maintain better control.",
    "KEEP_ARMS_STRAIGHT": "Fully extend your arms without locking your elbows. This helps control the movement and targets the right muscles.",
    "KEEP_HEAD_UP": "Lift your head and look slightly ahead. This keeps your neck aligned and helps balance your posture.",
    "KEEP_HIPS_BACK_SQUAT": "Push your hips back as if you're sitting in a chair. Don’t let your knees go too far forward.",
    "KEEP_KNEES_OVER_TOES_SQUAT": "Make sure your knees stay aligned over your toes. Avoid letting them cave inward or drift too far forward.",
    "KEEP_ELBOWS_UNDER_SHOULDERS": "Position your elbows directly under your shoulders to maintain joint alignment and control.",
    "KEEP_ARMS_LEVEL": "Raise or lower your arms to match each other. Keeping them level helps maintain symmetry and proper form.",
    "KEEP_FEET_SHOULDER_WIDTH": "Set your feet shoulder-width apart to create a stable base and prevent imbalance.",
    "KEEP_SHOULDERS_LEVEL": "Keep both shoulders at the same height. This improves balance and prevents overuse on one side.",
    "KEEP_SHOULDERS_ABOVE_HIPS": "Lift your upper body so your back stays above your hips. Don’t lean too far forward.",
    "KEEP_KNEES_POINTED_OUT": "Angle your knees slightly outward, in line with your toes. This protects your joints and keeps your stance strong.",
    "MOVE_INTO_CAMERA_FRAME" : "You are out of the camera frame, for accurate critiques please enter the frame of the camera" ,
    "MOVE_AWAY_FROM_CAMERA" : "You are too close to the camera for this exercise, I will not be able to provide any meaningful corrections" ,
    "FACE_CAMERA" : "Please be front facing to the camera for this exericse",
}


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

    def callback(indata, frames, time_info, status):


        global awaiting_rom_confirmation, adj_rom, awaiting_pause_confirmation, awaiting_new_exercise, form_inc
        global awaiting_exercise_ready
        current_exercise = pose_shared_state.get_value("current_exercise")

        finish_exercise = pose_shared_state.get_value("exercise_completed")

        if finish_exercise:
            stop_pose_detection()
            speak("Great work, finishing exercise")
            finish_exercise = False
            pose_shared_state.set_value("exercise_completed",finish_exercise)

        bad_form_list = pose_shared_state.get_value('bad_form')

        while (len(bad_form_list)>0 ):
            #correct form for each bad form value encountered
           # print(f'form increment: {form_inc}')
            speak(bad_form_dict[bad_form_list[form_inc]])
            #check if user form is corrected before trying to check other form
            bad_form_list = pose_shared_state.get_value('bad_form')
            #print(bad_form_list)
            form_inc = form_inc + 1
            length = len(bad_form_list)
            #print(f'bad form list length:{length}')
            if form_inc >=length:
                #print(f'resetting form increment: {form_inc} >= {length}')
                form_inc = 0

            bad_form_list.clear()

        adj_rom = pose_shared_state.get_value('ask_adjust_rom')

        if adj_rom and not awaiting_rom_confirmation:
                speak("I noticed your range of motion may need adjusting. Would you like me to change it?")
                awaiting_rom_confirmation = True
                return

        nonlocal sentence_buffer
        if recognizer.AcceptWaveform(bytes(indata)):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if not text:
                return

            print("🧍 You (live):", text)
            sentence_buffer += " " + text


            if awaiting_pause_confirmation:
                if any(kw in sentence_buffer for kw in continue_keywords):
                    pose_shared_state.set_value("exercise_paused",False)
                    speak("Ok, starting back up")
                    sentence_buffer = ""
                    awaiting_pause_confirmation=False
                elif any(kw in sentence_buffer for kw in new_exercise_keywords):
                    awaiting_new_exercise = True
                    speak("what exercise would you like to perform? please provide repetitions and name of exercise")
                    sentence_buffer = ""
                    awaiting_pause_confirmation=False
                    pose_shared_state.set_value("exercise_paused",False)
                elif any(kw in sentence_buffer for kw in stop_keywords):
                    pose_shared_state.set_value("exercise_paused",False)
                    stop_pose_detection()
                    speak("Finishing exercise")
                    pose_shared_state.set_value("exercise_completed",False)
                    awaiting_pause_confirmation=False
                    pose_shared_state.set_value("current_exercise",None)
                    

                    
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
                    speak("Okay, adjusting your range of motion.")
                    pose_shared_state.set_value("adjust_rom", True)
                else:
                    speak("Understood, keeping current range.")
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

            if(sentence_buffer.lower().split()[0] in prompt_keywords):
                if (any(kw in sentence_buffer.lower() for kw in (pause_keywords + stop_keywords)) and (current_exercise is not None)):
                    pose_shared_state.set_value("exercise_paused",True)
                    speak("Okay, pausing, let me know if you would like to continue with this exercise, start a new exercise, or end the current one")
                    awaiting_pause_confirmation = True
                    sentence_buffer = ""
                    return
                    

                if any(kw in sentence_buffer.lower() for kw in stop_keywords)  and (current_exercise is None) :
                    speak("Okay, stopping program now.")
                    os._exit(0)

                if (any(kw in sentence_buffer.lower() for kw in restart_keywords)and (current_exercise is not None)):
                    speak("Okay, restarting exercise.")
                    pose_shared_state.set_value("reset_exercise", True)
                    sentence_buffer = ""
                    return
                    

                if len(sentence_buffer.split()) >= 3:
                    process_user_input(sentence_buffer.strip())
                    sentence_buffer = ""
            else:
                sentence_buffer= ""


    print("🎤 Listening... Speak naturally. Say 'stop' to end.")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            time.sleep(0.1)
            #print('sleeping in converesational thread.')
                           


# ------------------
# Start Chat
# ------------------
if __name__ == "__main__":
    global my_queue
    speak("Hello this is the ARISE system, how may I help you today?")
    my_queue = queue.Queue()
    conversational_thread = threading.Thread(target=chatbot_loop)
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
                cv2.waitKey(1)
                cv2.destroyAllWindows()
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