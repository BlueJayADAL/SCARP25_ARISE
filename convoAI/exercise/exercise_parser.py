import re
from word2number import w2n

#---------------------------------------------------------------------------------------------------------
#  exercise parsing, searches for exercise mentioned in text and if its in the dicitonary, as well as a number for repetitions
#  returns the dicitonary contents and the number of reps if found and is then handled from the chat manager
#---------------------------------------------------------------------------------------------------------

class ExerciseParser:
    def __init__(self):
        self.exercise_keywords = {
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
            "squad": {  # common Vosk error
                "name": "squat",
                "muscle": "quadriceps, glutes, hamstrings",
                "instruction": "Stand with feet shoulder-width apart, bend your knees and hips to lower your body, then return to standing."
            }
            # Add more exercises as needed
        }

    def parse_exercise_intent(self, text):
        print("parsing exercise intent\n")
        text = text.lower()
        found_exercise = None
        reps = None

        for key in self.exercise_keywords:
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

        if found_exercise:
            return self.exercise_keywords[found_exercise], reps
        else:
            return None, reps
