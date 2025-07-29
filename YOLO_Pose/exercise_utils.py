# exercise_forms.py
# Utility functions for exercise tracking and analysis

BOTH = 0
LEFT = 1
RIGHT = 2
EITHER = 3

# Range of Motion (ROM) thresholds for each exercise
# Required angle movement to switch from concentric to eccentric phase
ROM_thresholds = {
    'bicep curl': 20,
    'squat': 20,
    'arm raise': 20,
    'lunge': 20
}

# Match exercises to their respective body parts
exercise_body_parts = {
    "bicep curl": {
        LEFT: "left_elbow",
        RIGHT: "right_elbow",
    },
    "squat": {
        LEFT: "left_knee",
        RIGHT: "right_knee",
    },
    "arm raise": {
        LEFT: "left_shoulder",
        RIGHT: "right_shoulder",
    },
    "lunge": {
        LEFT: "left_knee",
        RIGHT: "right_knee",
    }
}

# Default Target Range of Motion (ROM) for exercises
default_ROM = {
    'bicep curl': {
        LEFT: {"left_elbow" : (50, 150), "right_elbow" : (50, 150)},
        RIGHT: {"left_elbow" : (50, 150), "right_elbow" : (50, 150)},
        BOTH: {"left_elbow" : (50, 150), "right_elbow" : (50, 150)},
        EITHER: {"left_elbow" : (50, 150), "right_elbow" : (50, 150)}
    },
    'squat': {
        LEFT: {"left_knee" : (90, 150), "right_knee" : (90, 150)},
        RIGHT: {"left_knee" : (90, 150), "right_knee" : (90, 150)},
        BOTH: {"left_knee" : (90, 150), "right_knee" : (90, 150)},
        EITHER: {"left_knee" : (90, 150), "right_knee" : (90, 150)}
    },
    'arm raise': {
        LEFT: {"left_shoulder" : (20, 150), "right_shoulder" : (20, 150)},
        RIGHT: {"left_shoulder" : (20, 150), "right_shoulder" : (20, 150)},
        BOTH: {"left_shoulder" : (20, 150), "right_shoulder" : (20, 150)},
        EITHER: {"left_shoulder" : (20, 150), "right_shoulder" : (20, 150)}
    },
    'lunge': {
        LEFT: {"left_knee" : (90, 150), "right_knee" : (90, 150)},
        RIGHT: {"left_knee" : (90, 150), "right_knee" : (90, 150)},
        BOTH: {"left_knee" : (90, 150), "right_knee" : (90, 150)},
        EITHER: {"left_knee" : (90, 150), "right_knee" : (90, 150)}
    }
}

def get_exercise_body_parts(exercise, side):
    """
    Get the body part string associated with the exercise and side.
    """
    if (side == LEFT or side == RIGHT) and exercise in exercise_body_parts:
        return [exercise_body_parts[exercise].get(side, None)]
    elif (side == EITHER or side == BOTH) and exercise in exercise_body_parts:
        left = exercise_body_parts[exercise].get(LEFT, None)
        right = exercise_body_parts[exercise].get(RIGHT, None)
        return [left, right]
    return []

# Track Range of Motion (ROM) for each rep
def track_ROM(shared_data, angles, past_rep_min_angles, past_rep_max_angles, angles_decreasing, current_exercise, exercise_side, ROM, past_rep_list_size=4):
    body_part_keys = get_exercise_body_parts(current_exercise, exercise_side)
    while len(past_rep_min_angles) < len(body_part_keys):
        past_rep_min_angles.append([])
    while len(past_rep_max_angles) < len(body_part_keys):
        past_rep_max_angles.append([])
    while len(angles_decreasing) < len(body_part_keys):
        angles_decreasing.append(False)
    for i, (body_part, min_angles, max_angles) in enumerate(zip(body_part_keys, past_rep_min_angles, past_rep_max_angles)):
        if body_part is None:
            break
        angle = angles[body_part]
        if angle is not None:
            if len(min_angles) == 0:
                min_angles.append(angle)
            if len(max_angles) == 0:
                max_angles.append(angle)
            
            # Check if current angle has moved significantly from either end of known ROM:
            # Indicates change between decreasing/increasing angle movement (concentric vs eccentric phase)
            if angle < max_angles[-1] - ROM_thresholds[current_exercise]:
                # Check direction of angle movement
                if angles_decreasing[i]:
                    min_angles[-1] = min(angle, min_angles[-1])
                else: # base case, movement initialized
                    angles_decreasing[i] = True
                    if len(min_angles) == past_rep_list_size:
                        del(min_angles[0])
                    min_angles.append(angle)
            if angle > min_angles[-1] + ROM_thresholds[current_exercise]:
                # Check direction of angle movement
                if not angles_decreasing[i]:
                    max_angles[-1] = max(angle, max_angles[-1])
                else: # base case, movement initialized
                    angles_decreasing[i] = False
                    if len(max_angles) == past_rep_list_size:
                        del(max_angles[0])
                    max_angles.append(angle)

    # Perform check for if ROM should be adjusted
    # Does not check when exercise_size=EITHER - hard to track which side is being used
    suggested_rom = []
    ask_adjust_rom = False
    for body_part, min_angles, max_angles in zip(body_part_keys, past_rep_min_angles, past_rep_max_angles):
        if len(min_angles) == len(max_angles) == past_rep_list_size:
            # Check if most of last reps less than target ROM
            above_min_count = sum(1 for a in min_angles if a > ROM[current_exercise][exercise_side][body_part][0])
            below_max_count = sum(1 for a in max_angles if a < ROM[current_exercise][exercise_side][body_part][1])
            if exercise_side != EITHER and (above_min_count >= past_rep_list_size-1 or below_max_count >= past_rep_list_size-1):
                ask_adjust_rom = True
                suggested_rom.append((sum(min_angles[0:-1]) // past_rep_list_size-1, sum(max_angles[0:-1]) // past_rep_list_size-1))
            else:
                suggested_rom.append(None)
    shared_data.set_value('ask_adjust_rom', ask_adjust_rom)

    # Adjust current exercise ROM if confirmed by user
    if shared_data.get_value('adjust_rom'):
        shared_data.set_value('adjust_rom', False)
        shared_data.set_value('ask_adjust_rom', False)

        # Update the ROM for the current exercise and side
        for new_rom, body_part in zip(suggested_rom, body_part_keys):
            if new_rom is not None and body_part in ROM[current_exercise][exercise_side]:
                ROM[current_exercise][exercise_side][body_part] = new_rom
        shared_data.set_value('rom', ROM)