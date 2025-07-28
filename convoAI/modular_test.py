from convoAI.exercise.exercise_parser import ExerciseParser

parser = ExerciseParser()

ex1 = "Can I do 10 bicep curls?"
ex2 = "Let's try hammer curls"
ex3 = "I want to do fifteen squats"

print(parser.parse_exercise_intent(ex1))  # Should return bicep curl, 10
print(parser.parse_exercise_intent(ex2))  # Should return hammer curl, None
print(parser.parse_exercise_intent(ex3))  # Should return squat, 15
