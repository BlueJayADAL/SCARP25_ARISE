import numpy as np
from ultralytics import YOLO
import time

# Conditional import for testing purposes, if running directly 
from shared_data import SharedState
from exercise_forms import check_bad_form

# Constants
WIDTH = None
HEIGHT = None
BOTH = 0
LEFT = 1
RIGHT = 2
EITHER = 3

# Utility to calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    # Check for duplicate or invalid points
    if np.array_equal(a, b) or np.array_equal(b, c) or np.linalg.norm(a - b) == 0 or np.linalg.norm(c - b) == 0:
        return None 

    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    if np.isnan(angle):
        return None
    return int(np.degrees(angle))
def A(a, b, c):
    return calculate_angle(coords[a], coords[b], coords[c])

# Side on view
# Returns True if rep is done, otherwise False
def check_bicep_curl_rep(coords, angles, side=RIGHT):
    global rep_done
    global reps

    # do left side
    if (side==LEFT or side==EITHER) and angles['left_elbow']!=None:
        if angles['left_elbow'] < 55 and not rep_done and good_form:
            rep_done = True
            reps += 1
            return True
        elif angles['left_elbow'] > 150:
            rep_done = False
    # do right side
    elif (side==RIGHT or side==EITHER) and angles['right_elbow']!=None:
        if angles['right_elbow'] < 55 and not rep_done and good_form:
            rep_done = True
            reps += 1
            return True
        elif angles['right_elbow'] > 150:
            rep_done = False
    elif side==BOTH and angles['left_elbow'] and angles['right_elbow']!=None:
        if angles['left_elbow'] < 55 and angles['right_elbow'] < 55 and not rep_done and good_form:
            rep_done = True
            reps += 1
            return True
        elif angles['left_elbow'] > 150 and angles['right_elbow'] > 150:
            rep_done = False
    return False

# Straight on view
# Returns True if rep is done, otherwise False
def check_arm_raise_rep(coords, angles, side=BOTH):
    global rep_done
    global reps

    # do left side
    if (side==LEFT or side==EITHER) and angles['left_shoulder']!=None:
        if angles['left_shoulder'] > 150 and not rep_done and good_form:
            rep_done = True
            reps += 1
            return True
        elif angles['left_shoulder'] > 20:
            rep_done = False
    # do right side
    elif (side==RIGHT or side==EITHER) and angles['right_shoulder']!=None:
        if angles['right_shoulder'] > 150 and not rep_done and good_form:
            rep_done = True
            reps += 1
            return True
        elif angles['right_shoulder'] < 20:
            rep_done = False
    elif side==BOTH and angles['left_shoulder']!=None and angles['right_shoulder']!=None:
        if angles['left_shoulder'] > 150 and angles['right_shoulder'] > 150 and not rep_done and good_form:
            rep_done = True
            reps += 1
            return True
        elif angles['left_shoulder'] < 20 and angles['right_shoulder'] < 20:
            rep_done = False
    return False

# Side on view
# Returns True if rep is done, otherwise False
def check_squat_rep(coords, angles, side=BOTH):
    global rep_done
    global reps

    # do left side
    if (side==LEFT or side==EITHER) and angles['left_knee']!=None:
        if angles['left_knee'] < 90 and not rep_done and good_form:
            rep_done = True
            reps += 1
            return True
        elif angles['left_knee'] > 150:
            rep_done = False
    # do right side
    elif (side==RIGHT or side==EITHER) and angles['right_knee']!=None:
        if angles['right_knee'] < 90 and not rep_done and good_form:
            rep_done = True
            reps += 1
            return True
        elif angles['right_knee'] > 150:
            rep_done = False
    elif side==BOTH and angles['left_knee']!=None and angles['right_knee']!=None:
        if angles['left_knee'] is not None and angles['right_knee'] is not None:
            if angles['left_knee'] < 90 and angles['right_knee'] < 90 and not rep_done and good_form:
                rep_done = True
                reps += 1
                return True
            elif angles['left_knee'] > 150 and angles['right_knee'] > 150:
                rep_done = False
    return False

# Side on view
# Returns True if rep is done, otherwise False
def check_lunge_rep(coords, angles, side=RIGHT):
    global rep_done
    global reps

    if side == BOTH:
        side = EITHER
    # do left side
    if (side==LEFT or side==EITHER) and angles['left_knee']!=None and angles['right_knee']!=None:
        if angles['left_knee'] < 90 and coords[11] <= coords[13] and angles['right_knee'] < 130 and not rep_done and good_form:
            rep_done = True
            reps += 1
            return True
        elif angles['left_knee'] > 150 and angles['right_knee'] > 150:
            rep_done = False
    # do right side
    elif (side==RIGHT or side==EITHER) and angles['left_knee']!=None and angles['right_knee']!=None:
        if angles['right_knee'] < 90 and coords[12] <= coords[14] and angles['left_knee'] < 130 and not rep_done and good_form:
            rep_done = True
            reps += 1
            return True
        elif angles['right_knee'] > 150 and angles['left_knee'] > 150:
            rep_done = False
    return False

def adjust_ROM():
    print("Adjusting Range of Motion (ROM) is not implemented yet.")

# Sets pixel threshold that forces keypoints to have changed significantly in order to be recognized as different position
def smooth_keypoints(keypoints):
    pass

def init_yolo(exercise=None):
    global shared_data
    global current_exercise
    global exercise_side
    global reps_threshold
    global start_grace_threshold
    global form_threshold
    global start_time
    global bad_form_times

    global model
    global reps_done
    global reps
    global good_form


    # Load the YOLO pose model
    model = YOLO("../../../models/yolo11n-pose_openvino_model_320") 
    rep_done = False
    reps = 0
    good_form = True

    current_exercise = None
    exercise_side = EITHER  # BOTH, LEFT, RIGHT, EITHER(exclusive)
    reps_threshold = 10

    # Cooldown and threshold settings for form checking
    start_grace_threshold = 2.5
    form_threshold = 1.5
    start_time = time.perf_counter()
    bad_form_times = {
        'KEEP_BACK_STRAIGHT': -1,
        'KEEP_ELBOWS_CLOSE_TO_BODY': -1,
        'KEEP_ARMS_STRAIGHT': -1,
        'KEEP_HEAD_UP': -1,
        'KEEP_HIPS_BACK_SQUAT': -1,
        'KEEP_KNEES_OVER_TOES_SQUAT': -1,
        'KEEP_ELBOWS_UNDER_SHOULDERS': -1,
        'KEEP_ARMS_LEVEL': -1,
        'KEEP_FEET_SHOULDER_WIDTH': -1,
        'KEEP_SHOULDERS_LEVEL': -1,
        'KEEP_SHOULDERS_ABOVE_HIPS': -1,
        'KEEP_KNEES_POINTED_OUT': -1,
        'MOVE_INTO_CAMERA_FRAME': -1,
        'MOVE_AWAY_FROM_CAMERA': -1,
        'FACE_CAMERA': -1
    }

    #default values for shared state data 
    shared_data = SharedState()
    shared_data.set_value("exercise_completed",False)
    shared_data.set_value("reps",reps)
    shared_data.set_value("reps_threshold",reps_threshold)
    shared_data.set_value("ask_adjust_rom",False)
    shared_data.set_value("bad_form",[])
    shared_data.set_value("rom",[])
    shared_data.set_value("angles",{})

    shared_data.set_value("adjust_reps_threshold",-1)
    shared_data.set_value("exercise_paused",False)
    shared_data.set_value("current_exercise",exercise)
    shared_data.set_value("reset_exercise",False)
    shared_data.set_value("adjust_rom",False)

def arise_vision(frame):
    global model
    global reps
    global rep_done
    global good_form
    global angles
    global coords
    global WIDTH
    global HEIGHT

    global start_grace_threshold
    global form_threshold
    global start_time
    global bad_form_times

    global current_exercise
    global exercise_side
    global reps_threshold
    global shared_data

    def reset_bad_form_times():
        for key in bad_form_times:
            bad_form_times[key] = -1

    # Handle stop signal
    if not shared_data.running.is_set():
        print("🛑 Stop signal received — exiting pose thread.")
        reps = 0
        shared_data.set_value('reps', reps)
        return
    # Handle exercise paused
    if shared_data.get_value('exercise_paused'):
        time.sleep(0.2)
        return
    # Check if new exercise is set
    if shared_data.get_value('reset_exercise'):
        reps = 0
        shared_data.set_value('reps', reps)
        # default 10 reps threshold
        reps_threshold = shared_data.get_value('adjust_reps_threshold') if shared_data.get_value('adjust_reps_threshold') >= 0 else 10
        shared_data.set_value('adjust_reps_threshold', -1)
        shared_data.set_value('reps_threshold', reps_threshold)
        rep_done = False
        good_form = True
        current_exercise = shared_data.get_value('current_exercise')
        shared_data.set_value('bad_form', [])
        shared_data.set_value('reset_exercise', False)
        shared_data.set_value('exercise_completed', False)
        current_exercise = shared_data.get_value('current_exercise')
        start_time = time.perf_counter()
        reset_bad_form_times()
    # Check for adjusted Range of Motion (ROM)
    if shared_data.get_value('adjust_rom'):
        ROM = adjust_ROM()
        shared_data.set_value('ROM', ROM)
        shared_data.set_value('adjust_rom', False) 
    # Check for adjusted reps threshold
    if shared_data.get_value('adjust_reps_threshold') >= 0:
        reps_threshold = shared_data.get_value('adjust_reps_threshold')
        shared_data.set_value('reps_threshold', reps_threshold)
        shared_data.set_value('adjust_reps_threshold', -1)

    results = model(frame, verbose=False, conf=0.2)
        
    pose = results[0].keypoints # only focuses on one person at a time
    keypoints = pose.data[0].cpu().numpy().reshape(-1, 3)
    coords = [(int(x), int(y)) for x, y, _ in keypoints]
            
    WIDTH = annotated_frame.shape[1]
    HEIGHT = annotated_frame.shape[0]

    # No detection on screen / not clear
    if len(coords) < 17:
        keypoints = np.zeros((17, 3))
        coords = [(0, 0) for i in range(17)]

    shared_data.set_value('coords', coords) 

    # COCO keypoints:
    # 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear
    # 5-left_shoulder, 6-right_shoulder
    # 7-left_elbow, 8-right_elbow
    # 9-left_wrist, 10-right_wrist
    # 11-left_hip, 12-right_hip
    # 13-left_knee, 14-right_knee
    # 15-left_ankle, 16-right_ankle

    # Joint angles:
    angles = {
        'neck': int(keypoints[3][1] - keypoints[4][1]),  
        'left_shoulder': (a-90 if coords[7][1] > coords[5][1] else 270-a) if (a := A(6, 5, 7)) != None else None,
        'right_shoulder': (a-90 if coords[8][1] > coords[6][1] else 270-a) if (a := A(5, 6, 8)) != None else None,
        'left_elbow': A(5, 7, 9),
        'right_elbow': A(6, 8, 10),
        'left_hip': A(5, 11, 13),
        'right_hip': A(6, 12, 14),
        'left_knee': A(11, 13, 15),
        'right_knee': A(12, 14, 16)
    }
    shared_data.set_value('angles', angles) 

    # Mapping angles to their display locations (near joints)
    locations = {
        'neck': coords[0],
        'left_shoulder': coords[5],
        'right_shoulder': coords[6],
        'left_elbow': coords[7],
        'right_elbow': coords[8],
        'left_hip': coords[11],
        'right_hip': coords[12],
        'left_knee': coords[13],
        'right_knee': coords[14]
    }
    
    # Update exercise reps, display exercise
    if current_exercise != None and current_exercise != 'complete':
        rep_inc = False
        if current_exercise == 'bicep curl':
            rep_inc = check_bicep_curl_rep(coords, angles, side=exercise_side)
        elif current_exercise == 'squat':
            rep_inc = check_squat_rep(coords, angles, side=exercise_side)
        elif current_exercise == 'arm raise':
            rep_inc = check_arm_raise_rep(coords, angles, side=exercise_side)
        elif current_exercise == 'lunge':
            rep_inc = check_lunge_rep(coords, angles, side=exercise_side)
        # Update shared state
        if rep_inc:
            shared_data.set_value('reps', reps)

        # Check form, warn user if improper form
        form_check_cooldown = time.perf_counter()
        bad_form_list = []
        if form_check_cooldown - start_time > start_grace_threshold:
            bad_form_list = check_bad_form(current_exercise, coords, angles, (WIDTH, HEIGHT), side=exercise_side)
            for bad_form, form_time in bad_form_times.items():
                if form_time == -1 and bad_form in bad_form_list:
                    bad_form_times[bad_form] = form_check_cooldown
                    bad_form_list.remove(bad_form)
                elif form_time != -1 and bad_form in bad_form_list:
                    if form_check_cooldown - form_time < form_threshold:
                        bad_form_list.remove(bad_form) # remove if not enough time has passed
                elif form_time != -1 and bad_form not in bad_form_list:
                    bad_form_times[bad_form] = -1 # reset time if no longer in bad form list            
        shared_data.set_value('bad_form', bad_form_list)
        good_form = len(bad_form_list) == 0

        # Check if exercise complete
        if reps >= reps_threshold:
            current_exercise = None
            shared_data.set_value('current_exercise', current_exercise)
            shared_data.set_value('exercise_completed', True)   

    return shared_data 
