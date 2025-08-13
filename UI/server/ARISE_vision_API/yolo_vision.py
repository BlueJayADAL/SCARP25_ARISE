import numpy as np
from ultralytics import YOLO
import time
import os

# Conditional import for testing purposes, if running directly 
from ARISE_vision_API.shared_data import SharedState
from ARISE_vision_API.exercise_forms import check_bad_form, check_rep
from YOLO_Pose.exercise_utils import track_ROM, default_ROM, load_user_data

# Constants for exercise side selection
WIDTH = None
HEIGHT = None
BOTH = 0
LEFT = 1
RIGHT = 2
EITHER = 3

def calculate_angle(a, b, c):
    '''
    Utility to calculate angle between three points
    '''
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
    '''
    Helper for angle calculation using global coords
    '''
    return calculate_angle(coords[a], coords[b], coords[c])

def init_yolo(exercise: str = None):
    '''
    Initialize all global state and model for pose detection and exercise tracking
    '''
    global shared_data
    global current_exercise
    global exercise_side
    global reps_threshold
    global start_grace_threshold
    global form_threshold
    global start_time
    global bad_form_times

    global model
    global rep_done
    global reps
    global good_form

    global past_rep_list_size, past_rep_max_angles, past_rep_min_angles, angles_decreasing, ROM

    # Load the YOLO pose model
    model_path = os.path.join(os.getcwd(), "../../models/yolo11n-pose_openvino_model")
    model = YOLO(model_path)
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

    # Range of Motion (ROM) adjustment
    past_rep_list_size = 4
    past_rep_max_angles = [[]] # previous reps: max angle detected on each rep, per angle tracked
    past_rep_min_angles = [[]] # previous reps: mix angle detected on each rep, per angle tracked
    angles_decreasing = [False]
    # Load user-specific ROM settings; or use default
    ROM = load_user_data().get('DEFAULT_USER', {}).get('rom', default_ROM.copy())

    # Default values for shared state data 
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
    '''
    Main function to process a frame, update shared state, and check exercise logic
    '''
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

    global past_rep_list_size, past_rep_max_angles, past_rep_min_angles, angles_decreasing, ROM

    def reset_bad_form_times():
        # Reset all bad form cooldown timers
        for key in bad_form_times:
            bad_form_times[key] = -1

    # Handle exercise paused
    if shared_data.get_value('exercise_paused'):
        time.sleep(0.2)
        return shared_data
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
        shared_data.set_value('ROM', ROM)
        shared_data.set_value('adjust_rom', False)
        past_rep_max_angles = []
        past_rep_min_angles = []
        angles_decreasing = []
    # Check for adjusted reps threshold
    if shared_data.get_value('adjust_reps_threshold') >= 0:
        reps_threshold = shared_data.get_value('adjust_reps_threshold')
        shared_data.set_value('reps_threshold', reps_threshold)
        shared_data.set_value('adjust_reps_threshold', -1)

    # Run YOLO pose estimation on the frame
    results = model(frame, verbose=False, conf=0.2)
        
    pose = results[0].keypoints # only focuses on one person at a time
    keypoints = pose.data[0].cpu().numpy().reshape(-1, 3)
    coords = [(int(x), int(y)) for x, y, _ in keypoints]
            
    WIDTH = frame.shape[1]
    HEIGHT = frame.shape[0]

    # No detection on screen / not clear
    if len(coords) < 17:
        keypoints = np.zeros((17, 3))
        coords = [(0, 0) for i in range(17)]

    shared_data.set_value('coords', coords) 
    shared_data.set_value('keypoints', keypoints)

    # COCO keypoints:
    # 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear
    # 5-left_shoulder, 6-right_shoulder
    # 7-left_elbow, 8-right_elbow
    # 9-left_wrist, 10-right_wrist
    # 11-left_hip, 12-right_hip
    # 13-left_knee, 14-right_knee
    # 15-left_ankle, 16-right_ankle

    # Joint angles: calculate for each relevant joint
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
    
    # Update exercise reps, display exercise
    if current_exercise != None and current_exercise != 'complete':
        # Update shared state rep count
        rep_done, reps, rep_increment = check_rep(current_exercise, rep_done, reps, good_form, coords, angles, side=exercise_side)
        if rep_increment:
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
        
        # Track/record current exercise ROM
        track_ROM(shared_data, angles, past_rep_min_angles, past_rep_max_angles, angles_decreasing, current_exercise, exercise_side, ROM, past_rep_list_size=past_rep_list_size)

        # Check if exercise complete
        if reps >= reps_threshold:
            current_exercise = None
            shared_data.set_value('current_exercise', current_exercise)
            shared_data.set_value('exercise_completed', True)   

    return shared_data
