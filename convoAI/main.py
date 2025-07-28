"""
ARISE Voice Assistant Main Entry Point

Coordinates all subsystems: VAD, LLM, TTS, Exercise Parser, Pose Detection
"""

import threading
import queue
import cv2

from convoAI.audio.audio_player import AudioPlayer
from convoAI.core.chat_manager import ChatManager
from convoAI.core.vad_controller import VADController
from convoAI.nlp.tts_engine import TTSEngine
from convoAI.nlp.llm_engine import LLMEngine
from convoAI.exercise.exercise_parser import ExerciseParser
from convoAI.exercise.form_monitor import start_bad_form_monitor


from YOLO_Pose.yolo_threaded import thread_main
from YOLO_Pose.shared_data import SharedState

bad_form_dict = {
    "KEEP_BACK_STRAIGHT": "tts_cache/keep_your_back_straight_and_avoid_roundi.wav",
    "KEEP_ELBOWS_CLOSE_TO_BODY": "tts_cache/tuck_your_elbows_in_close_to_your_sides_.wav",
    "KEEP_ARMS_STRAIGHT": "tts-cache/fully_extend_your_arms_without_locking_y.wav",
    "KEEP_HEAD_UP": "tts_cache/lift_your_head_and_look_slightly_ahead._.wav",
    "KEEP_HIPS_BACK_SQUAT": "tts_cache/push_your_hips_back_as_if_you're_sitting.wav",
    "KEEP_KNEES_OVER_TOES_SQUAT": "tts_cache/make_sure_your_knees_stay_aligned_over_y.wav",
   
    "KEEP_ELBOWS_UNDER_SHOULDERS": "tts_cache/position_your_elbows_directly_under_your.wav",
    "KEEP_ARMS_LEVEL": "tts_cache/raise_or_lower_your_arms_to_match_each_o.wav",
    "KEEP_FEET_SHOULDER_WIDTH": "tts_cache/set_your_feet_shoulder-width_apart_to_cr.wav",
    "KEEP_SHOULDERS_LEVEL": "tts_cache/keep_both_shoulders_at_the_same_height._.wav",
    "KEEP_SHOULDERS_ABOVE_HIPS": "tts_cache/lift_your_upper_body_so_your_back_stays_.wav",
    "KEEP_KNEES_POINTED_OUT": "tts_cache/angle_your_knees_slightly_outward,_in_li.wav",
    "MOVE_INTO_CAMERA_FRAME" : "tts_cache/you_are_out_of_the_camera_frame,_for_acc.wav" ,
    "MOVE_AWAY_FROM_CAMERA" : "tts_cache/you_are_too_close_to_the_camera_for_this.wav" ,
    "FACE_CAMERA" : "tts_cache/please_be_front_facing_to_the_camera_for.wav",
}


class ARISEVoiceAssistant:
    def __init__(self):
        # Shared pose state and queue
        self.pose_state = SharedState()
        self.pose_queue = queue.Queue(maxsize=5)
        self.pose_thread = None

        # Subsystems
        self.audio_player = AudioPlayer()
        self.tts_engine = TTSEngine()
        self.llm_engine = LLMEngine()
        self.exercise_parser = ExerciseParser()

        # Chat logic (inject self for pose control)
        self.chat_manager = ChatManager(
            state_manager=self.pose_state,
            tts_engine=self.tts_engine,
            llm_engine=self.llm_engine,
            exercise_parser=self.exercise_parser,
            audiofile_player= self.audio_player,
            pose_controller=self  # pass this instance to allow chat to call start/stop_pose_detection
        )

        # VAD and microphone handler
        self.vad_controller = VADController(
            model_path="models/vosk-small",
            on_text_callback=self.chat_manager.process_text,
            audio_guard_funcs=[self.audio_player.is_playing,self.tts_engine.is_playing]
        )

        start_bad_form_monitor(
            pose_shared_state=self.pose_state,
            audio_player=self.audio_player,
            bad_form_dict=bad_form_dict
        )

        # GUI
        self.window_open = False

    def start(self):
        print("🚀 Starting ARISE Voice Assistant")

        # Start audio player
        self.audio_player.start()
        self.audio_player.play_file("tts_cache/hello_this_is_the_arise_system,_how_may_.wav")

        # Start VAD recognizer in a thread
        threading.Thread(target=self.vad_controller.start, daemon=True).start()

        # Start main UI loop
        self._main_loop()

    def _main_loop(self):
        while True:
            exercise_name = self.pose_state.get_value("current_exercise")
            if exercise_name:
                self._handle_exercise_gui()
            else:
                self._handle_no_exercise_gui()

    def _handle_exercise_gui(self):
        try:
            frame = self.pose_queue.get(timeout=2)
            if frame is not None:
                cv2.imshow("ARISE Exercise Monitor", frame)
                self.window_open = True

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    self.shutdown()
        except queue.Empty:
            pass

    def _handle_no_exercise_gui(self):
        if self.window_open:
            cv2.destroyAllWindows()
            self.window_open = False

    def start_pose_detection(self):
        if not self.pose_state.running.is_set():
            print("🏃 Starting pose detection")
            self.pose_state.running.set()
            self.pose_thread = threading.Thread(
                target=thread_main,
                args=(self.pose_state,),
                kwargs={"thread_queue": self.pose_queue},
                daemon=True
            )
            self.pose_thread.start()

    def stop_pose_detection(self):
        if self.pose_state.running.is_set():
            print("🛑 Stopping pose detection")
            self.pose_state.running.clear()
            if self.pose_thread:
                self.pose_thread.join(timeout=1)

    def shutdown(self):
        print("👋 Shutting down")
        self.stop_pose_detection()
        self.audio_player.stop()
        cv2.destroyAllWindows()
        exit(0)


if __name__ == "__main__":
    assistant = ARISEVoiceAssistant()
    assistant.start()
