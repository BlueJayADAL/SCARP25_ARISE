"""
ARISE Voice Assistant Main Entry Point - Streaming Version

Coordinates all subsystems with streaming LLM-to-TTS integration for reduced latency
"""

import threading
import queue
import cv2

from convoAI.audio.audio_player import AudioPlayer
from convoAI.core.streaming_chat_manager import StreamingChatManager  # Updated import
from convoAI.core.vad_controller import VADController
from convoAI.nlp.streaming_tts_engine import StreamingTTSEngine  # Updated import
from convoAI.nlp.streaming_llm_engine import StreamingLLMEngine  # Updated import
from convoAI.exercise.exercise_parser import ExerciseParser
from convoAI.exercise.form_monitor import start_bad_form_monitor

from YOLO_Pose.yolo_threaded import thread_main
from YOLO_Pose.shared_data import SharedState

#---------------------------------------------------------------------------------------------------------
#   Enhanced main file with streaming LLM-to-TTS integration
#  
#   Significantly reduced latency for LLM responses through sentence-level streaming
#   Maintains all existing functionality while improving response times
#---------------------------------------------------------------------------------------------------------

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
    "MOVE_INTO_CAMERA_FRAME": "tts_cache/you_are_out_of_the_camera_frame,_for_acc.wav",
    "MOVE_AWAY_FROM_CAMERA": "tts_cache/you_are_too_close_to_the_camera_for_this.wav",
    "FACE_CAMERA": "tts_cache/please_be_front_facing_to_the_camera_for.wav",
}


class ARISEStreamingVoiceAssistant:
    def __init__(self):
        # Shared pose state and queue
        self.pose_state = SharedState()
        self.pose_queue = queue.Queue(maxsize=5)
        self.pose_thread = None

        # Enhanced subsystems with streaming capabilities
        self.audio_player = AudioPlayer()
        self.tts_engine = StreamingTTSEngine()  # Updated to streaming version
        self.llm_engine = StreamingLLMEngine()  # Updated to streaming version
        self.exercise_parser = ExerciseParser()

        # Enhanced chat logic with streaming support
        self.chat_manager = StreamingChatManager(  # Updated to streaming version
            state_manager=self.pose_state,
            tts_engine=self.tts_engine,
            llm_engine=self.llm_engine,
            exercise_parser=self.exercise_parser,
            audiofile_player=self.audio_player,
            pose_controller=self  # pass this instance to allow chat to call start/stop_pose_detection
        )

        # VAD and microphone handler with enhanced audio guards
        self.vad_controller = VADController(
            model_path="models/vosk-model-small-en-us-0.15",
            on_text_callback=self.chat_manager.process_text,
            audio_guard_funcs=[
                self.audio_player.is_playing,
                self.tts_engine.is_playing  # Enhanced TTS playing detection
            ]
        )

        # Start form monitoring
        start_bad_form_monitor(
            pose_shared_state=self.pose_state,
            audio_player=self.audio_player,
            bad_form_dict=bad_form_dict
        )

        # GUI
        self.window_open = False

    def start(self):
        print("🚀 Starting ARISE Streaming Voice Assistant")
        print("✨ Enhanced with streaming LLM-to-TTS for reduced latency")

        # Start audio player
        self.audio_player.start()
        self.audio_player.play_file("tts_cache/hello_this_is_the_arise_system,_how_may_.wav")

        # Start VAD recognizer in a thread
        threading.Thread(target=self.vad_controller.start, daemon=True).start()

        # Start main UI loop
        self._main_loop()

    def _main_loop(self):
        while not self.chat_manager.shutdown_event.is_set():
            exercise_name = self.pose_state.get_value("current_exercise")
            if exercise_name:
                self._handle_exercise_gui()
            else:
                self._handle_no_exercise_gui()

        self.shutdown()

    def _handle_exercise_gui(self):
        try:
            frame = self.pose_queue.get(timeout=2)
            if frame is not None:
                cv2.imshow("ARISE Exercise Monitor", frame)
                self.window_open = True

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    self.shutdown()
                elif key & 0xFF == ord('i'):  # 'i' for interrupt
                    print("🛑 Manual interrupt triggered")
                    self.chat_manager.interrupt_current_response()
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
        print("👋 Shutting down streaming assistant")
        self.stop_pose_detection()
        
        # Enhanced shutdown with streaming support
        if hasattr(self.tts_engine, 'stop_playback'):
            self.tts_engine.stop_playback()
        
        self.audio_player.stop()
        cv2.destroyAllWindows()
        exit(0)


# Backward compatibility class
class ARISEVoiceAssistant(ARISEStreamingVoiceAssistant):
    """Backward compatibility alias"""
    pass


if __name__ == "__main__":
    print("🎯 Starting ARISE with streaming LLM-to-TTS integration")
    print("📈 Expected improvements:")
    print("   • Reduced response latency")
    print("   • Smoother conversation flow") 
    print("   • Sentence-by-sentence audio playback")
    print()
    
    assistant = ARISEStreamingVoiceAssistant()
    assistant.start()