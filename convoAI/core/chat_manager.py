from convoAI.nlp.text_utils import clean_text_for_tts, extract_after_keyword
import threading

#---------------------------------------------------------------------------------------------------------
#   Handling text input from user, use of different keywords for different areas of conversation for easier intteruption and lower latency
#   
#   First checks for prompt keyword, how an alexa or google home devce would wait for "alexa do  ____"
#
#   after finding prompt keyword, depending on the area of conversation the text is handled and processed differently
#---------------------------------------------------------------------------------------------------------


class ChatManager:
    def __init__(self, state_manager, tts_engine, llm_engine, exercise_parser, pose_controller, audiofile_player):
        self.state = state_manager
        self.tts = tts_engine
        self.llm = llm_engine
        self.parser = exercise_parser
        self.pose = pose_controller
        self.audio = audiofile_player
        self.shutdown_event = threading.Event()


        self.awaiting_exercise_ready = False
        self.awaiting_rom_confirmation = False
        self.awaiting_pause_confirmation = False
        self.awaiting_new_exercise = False

        self.prompt_keywords = ["arise", "rise"]
        self.start_keywords = ["ready", "start", "go", "begin"]
        self.stop_keywords = ["stop", "exit", "quit", "end"]
        self.pause_keywords = ["pause", "hold", "wait", "timeout"]
        self.continue_keywords = ["continue", "resume", "start again"]
        self.agree_keywords = ["yes", "yeah", "of course"]
        self.new_exercise_keywords = ["new exercise", "switch exercise", "different exercise"]
        self.restart_keywords = ["restart", "redo", "start over"]
        self.all_keywords = self.prompt_keywords + self.start_keywords +self.stop_keywords +self.pause_keywords + self.continue_keywords+ self.agree_keywords + self.new_exercise_keywords+ self.restart_keywords 

    def process_text(self, text: str):
        print(f"🧠 Received: {text}")
        text = text.lower().strip()

        # no prompt keyword and has less than 3 words so ignore text could be just noise interpreted as a word
        if not any(word in self.all_keywords for word in text.split()) and len(text.split()) < 3:
            print("⚠️ No significance detected — skipping input.")
            return
        
        # Stop keyword detected, if not in exercise stop the program
        if any(kw in text.lower() for kw in self.stop_keywords):
            current_exercise = self.state.get_value("current_exercise")
            if not current_exercise:
                self.tts.speak("Goodbye!")
                print("👋 No active exercise. Exiting ARISE system.")
                self.shutdown_event.set()


        # Pause handling
        if self.awaiting_pause_confirmation:
            if any(k in text for k in self.continue_keywords):
                self.state.set_value("exercise_paused", False)
                self.tts.speak("Okay, resuming.")
                self.awaiting_pause_confirmation = False
                return
            elif any(k in text for k in self.new_exercise_keywords):
                self.awaiting_new_exercise = True
                self.tts.speak("Okay, what exercise would you like to perform?")
                self.awaiting_pause_confirmation = False
                self.state.set_value("exercise_paused", False)
                return
            elif any(k in text for k in self.stop_keywords):
                self.pose.stop_pose_detection()
                self.state.set_value("exercise_paused", False)
                self.state.set_value("exercise_completed", False)
                self.state.set_value("current_exercise", None)
                self.tts.speak("Finishing the exercise.")
                self.awaiting_pause_confirmation = False
                return

        # Awaiting new exercise
        if self.awaiting_new_exercise:
            details, reps = self.parser.parse_exercise_intent(text)
            if details:
                self.state.set_value("reset_exercise", True)
                self.state.set_value("current_exercise", details['name'])
                if reps:
                    self.state.set_value("adjust_reps_threshold", reps)
                    self.tts.speak(f"Okay, starting {reps} reps of {details['name']}. Let me know when you're ready.")
                    self.awaiting_exercise_ready = True
                else:
                    self.tts.speak(f"A {details['name']} works the {details['muscle']}. Here's how: {details['instruction']}")
            self.awaiting_new_exercise = False
            return

        # ROM confirmation if user struggles to hit range of motion
        if self.awaiting_rom_confirmation:
            if any(k in text for k in self.agree_keywords):
                self.tts.speak("Okay, adjusting your range of motion.")
                self.state.set_value("adjust_rom", True)
            else:
                self.tts.speak("Understood, keeping current range.")
                self.state.set_value("adjust_rom", False)
            self.awaiting_rom_confirmation = False
            return

        # Start pose detection when user says they are "ready" for exercise
        if self.awaiting_exercise_ready and any(k in text for k in self.start_keywords):
            self.pose.start_pose_detection()
            self.awaiting_exercise_ready = False
            return

        # Strip prompt keyword like "arise ..."
        parsed_prompt = extract_after_keyword(text, self.prompt_keywords)
        print(f"🔎 Extracted prompt after keyword: '{parsed_prompt}'")
        if not parsed_prompt:
            print("⚠️ No prompt extracted. Skipping downstream checks.")
        if parsed_prompt:
            text = parsed_prompt

        # Pause / stop request during exercise
        if self.state.get_value("current_exercise"):
            if any(k in text for k in (self.pause_keywords + self.stop_keywords)):
                self.state.set_value("exercise_paused", True)
                self.awaiting_pause_confirmation = True
                self.audio.play_file("tts_cache\okay,_pausing._let_me_know_if_you_want_t.wav")
                return
            if any(k in text for k in self.stop_keywords):
                self.tts.speak("Okay, stopping now.")
                self.pose.stop_pose_detection()
                self.state.set_value("current_exercise", None)
                return
            if any(k in text for k in self.restart_keywords):
                self.tts.speak("Okay, restarting exercise.")
                self.state.set_value("reset_exercise", True)
                return

        # Exercise detection, parsed from the users input to dictionary of exercises available and supported
        # having only the exercise leads to a description and explaination of the exercise but inclusion of a number
        # leads to the starting of an exercise for the user
        details, reps = self.parser.parse_exercise_intent(text)
        print(f"🏋️ Exercise parse: details={details}, reps={reps}")
        if details:
            if reps:
                self.state.set_value("reset_exercise", True)
                self.state.set_value("current_exercise", details['name'])
                self.state.set_value("adjust_reps_threshold", reps)
                self.state.set_value("exercise_paused", False)
                self.awaiting_exercise_ready = True
                self.tts.speak(f"Okay, starting {reps} reps of {details['name']}. Let me know when you're ready.")
            else:
                self.tts.speak(f"A {details['name']} works the {details['muscle']}. Here's how: {details['instruction']}")
            return

        # Fallback to LLM generation
        #self.tts.speak("Give me a moment to think.")
        response = self.llm.generate_reply(text)
        cleaned = clean_text_for_tts(response)
        self.tts.speak(cleaned)
