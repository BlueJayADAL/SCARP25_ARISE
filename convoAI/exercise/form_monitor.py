
import threading
import time

#---------------------------------------------------------------------------------------------------------
#  file used to iterate through the bad forms of user while they are performing an exercise 
#  
#   as a user fixes their form, the audio from the audio player will get cleared as if a user corrects their form there is no longer
#
#   a reason to iterate through the whole list of bad forms, becasue fixing one area of form could fix multiple others
#---------------------------------------------------------------------------------------------------------

def start_bad_form_monitor(pose_shared_state, audio_player, bad_form_dict):
    def monitor_loop():
        seen_issues = set()

        while True:
            current_exercise = pose_shared_state.get_value("current_exercise")
            if current_exercise:
                bad_form_list = pose_shared_state.get_value("bad_form")

                # If no bad forms, clear seen issues and stop audio
                if not bad_form_list:
                    seen_issues.clear()
                    audio_player.stop_audio_playback()  
                    time.sleep(0.5)
                    continue

                for key in bad_form_list:
                    if key in bad_form_dict and key not in seen_issues:
                        print(f"🔁 Playing correction for: {key}")
                        audio_player.play_file(bad_form_dict[key])
                        seen_issues.add(key)

                # Check for removed corrections
                current_set = set(bad_form_list)
                seen_issues.intersection_update(current_set)

            else:
                seen_issues.clear()

            time.sleep(0.5)

    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
