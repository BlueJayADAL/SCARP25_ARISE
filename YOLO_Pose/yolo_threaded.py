import torch
# torch.manual_seed(10)
import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
import queue
import cv2
import time
import queue
import threading
import numpy as np
from pathlib import Path
from PIL import Image
from functools import partial
try:
    from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm
except ImportError as e:
    pass

# Conditional import for testing purposes, if running directly 
from YOLO_Pose.shared_data import SharedState
from YOLO_Pose.exercise_forms import check_bad_form

OPENCV = 0
PICAM = 1
SYNCHRONOUS = 0
QUEUED = 1

CAMERA_TYPE = OPENCV
HAILO = 0   #   0: False  |  1: True
HAILO_METHOD = QUEUED

if CAMERA_TYPE == PICAM:
    from picamera2 import Picamera2

if HAILO == 1:
    from YOLO_Pose.hailo.hailo_pose_util import hailo_init, get_postprocess, postprocess, hailo_sync_infer
    
cam = None
if CAMERA_TYPE == PICAM:
    cam = Picamera2()
    cam.preview_configuration.main.size = (1280, 1280)
    cam.preview_configuration.main.format = "RGB888"
    cam.preview_configuration.align()
    cam.configure("preview")
    cam.start()

# Load the YOLO pose model
if HAILO == 0:  
    model = YOLO("models/yolo11n-pose_openvino_model_320") 
rep_done = False
reps = 0
good_form = True
angles = {}
coords = []
logging_file_path = 'exercise_log.csv'
logging_file_readable_path = 'exercise_log_readable.txt'
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


# Display text on the frame
def display_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=2):
    x, y = (int(position[0]), int(position[1]))
    (text_width, text_height), baseline = cv2.getTextSize(str(text), font, font_scale, font_thickness)
    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), (255, 255, 255), thickness=cv2.FILLED)
    cv2.putText(frame, str(text), (x, y), font, font_scale, (255, 0, 255), 2)

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

def thread_main(shared_data=SharedState(), logging=False, save_log=False, thread_queue=None):
    global model
    global reps
    global rep_done
    global good_form
    global angles
    global coords
    global WIDTH
    global HEIGHT
    current_exercise = None
    exercise_side = EITHER  # BOTH, LEFT, RIGHT, EITHER(exclusive)
    reps_threshold = 10
    global cam

    if HAILO == 1:
        hailo_init(shared_data, CAMERA_TYPE, HAILO_METHOD)

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
    def reset_bad_form_times():
        for key in bad_form_times:
            bad_form_times[key] = -1

    # Initialize camera if using OpenCV
    if CAMERA_TYPE == OPENCV:
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        if not ret:
            print("Switching USB camera port to 8")
            cam = cv2.VideoCapture(8)

    if save_log:
        with open(logging_file_path, 'w') as log_file:
            log_file.write("time_unix_ms,current_exercise,reps,reps_threshold,neck,left_shoulder,right_shoulder,left_elbow,right_elbow,left_hip,right_hip,left_knee,right_knee\n")

    if __name__ == '__main__':
        #default values for shared state data 
        shared_data.set_value("exercise_completed",False)
        shared_data.set_value("reps",-1)
        shared_data.set_value("reps_threshold",-1)
        shared_data.set_value("ask_adjust_rom",False)
        shared_data.set_value("bad_form",[])
        shared_data.set_value("rom",[])
        shared_data.set_value("angles",{})

        shared_data.set_value("adjust_reps_threshold",-1)
        shared_data.set_value("exercise_paused",False)
        shared_data.set_value("current_exercise",None)
        shared_data.set_value("reset_exercise",False)
        shared_data.set_value("adjust_rom",False)
    paused = False
    fps_time = time.perf_counter()
    while True:
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord('p'):
            # paused = not paused
        # if paused:
            # time.sleep(0.2)
            # continue

        # Handle stop signal
        if not __name__ == '__main__':
            if not shared_data.running.is_set():
                print("🛑 Stop signal received — exiting pose thread.")
                reps = 0
                shared_data.set_value('reps', reps)
                break
        # Handle exercise paused
        if shared_data.get_value('exercise_paused'):
            time.sleep(0.2)
            continue
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

        frame = None
        # Check what hardware & inference method to use
        if HAILO == 0:
            if CAMERA_TYPE == PICAM:
                frame = cam.capture_array()
            elif CAMERA_TYPE == OPENCV:
                #print(cam.isOpened())
                ret, frame = cam.read()
                if not ret:
                    print("Failed to grab frame")
                    break

            results = model(frame, verbose=False, conf=0.2)
            annotated_frame = results[0].plot()
            
            #for pose in results[0].keypoints: # accounts for multiple people in frame
            pose = results[0].keypoints # only focuses on one person at a time
            keypoints = pose.data[0].cpu().numpy().reshape(-1, 3)
            coords = [(int(x), int(y)) for x, y, _ in keypoints]
        else: # HAILO == 1
            if HAILO_METHOD == SYNCHRONOUS:
                if CAMERA_TYPE == PICAM:
                    frame = cam.capture_array()
                elif CAMERA_TYPE == OPENCV:
                    #print(cam.isOpened())
                    ret, frame = cam.read()
                    if not ret:
                        print("Failed to grab frame")
                        break

                keypoints, annotated_frame = hailo_sync_infer(frame)
                keypoints = np.array(keypoints).reshape(-1,3)
                coords = [(int(x), int(y)) for x, y, _ in keypoints]
            else: # HAILO_METHOD == QUEUED
                try:
                    post_data = get_postprocess()
                except queue.Empty:
                    print('postprocess_queue empty')
                    continue

                keypoints, annotated_frame = postprocess(post_data)
                keypoints = np.array(keypoints).reshape(-1,3)
                coords = [(int(x), int(y)) for x, y, _ in keypoints]
                
        WIDTH = annotated_frame.shape[1]
        HEIGHT = annotated_frame.shape[0]

        i = 0
        for point in coords:
            display_text(annotated_frame, f'{i}: {point[0],point[1]}', (5,10+20*i))
            i+=1

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

        for joint, angle in angles.items():
            pos = locations[joint]
            display_text(annotated_frame, angle, (pos[0] + 10, pos[1] - 10))

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
            display_text(annotated_frame, f'Reps: {reps}/{reps_threshold}', (WIDTH-100, HEIGHT-50))
            display_text(annotated_frame, f'Current Exercise: {current_exercise}', (int(WIDTH/2-100), 30))
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
            # print([(_,__) for _, __ in bad_form_times.items() if __ != -1])  # DEBUG: Print bad form times

            # DEBUG: Display bad form warnings
            for i, warning in enumerate(bad_form_list):
                display_text(annotated_frame, warning, (200, 60 + 20 * i))
            # DEBUG: Display shared state data
            shared_data_data = shared_data.get_all_data()
            for i, (_key, _value) in enumerate(shared_data_data.items()):
                display_text(annotated_frame, f'{_key}: {_value}', (250, 300 + 20 * i))

            # Check if exercise complete
            if reps >= reps_threshold:
                current_exercise = None
                shared_data.set_value('current_exercise', current_exercise)
                shared_data.set_value('exercise_completed', True)
                
        elif current_exercise == 'complete':
            display_text(annotated_frame, 'Exercise complete!', (int(WIDTH/2-100), 30))
        elif current_exercise == None:
            display_text(annotated_frame, 'No exercise selected', (int(WIDTH/2-100), 30))

        if __name__=="__main__":
            # DEBUG: Display fps
            display_text(annotated_frame, f'FPS: {1/(time.perf_counter()-fps_time)}', (50, HEIGHT-50))
            fps_time = time.perf_counter()
            
            annotated_frame = cv2.resize(annotated_frame, (640,480))
            cv2.imshow("Pose with Angles", annotated_frame)
            # Handle quitting, key pressing
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
                
            # Testing for setting exercise type
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
        else:
            thread_queue.put(annotated_frame)

        # Logging
        if logging:
            print(f"Current Exercise: {current_exercise}, Reps: {reps}/{reps_threshold}, Angles: {angles}")
        if save_log:
            with open(logging_file_path, 'a') as log_file:
                log_file.write(f"{int(time.time()*1000)},{current_exercise},{reps},{reps_threshold},"
                               f"{angles['neck']},{angles['left_shoulder']},{angles['right_shoulder']},"
                               f"{angles['left_elbow']},{angles['right_elbow']},"
                               f"{angles['left_hip']},{angles['right_hip']},"
                               f"{angles['left_knee']},{angles['right_knee']}\n")
            with open(logging_file_readable_path, 'a') as log_file:
                log_file.write(f"Current Exercise: {current_exercise}, Reps: {reps}/{reps_threshold}, Angles: {angles}\n")

    # Cleanup
    if CAMERA_TYPE == PICAM:
        cam.stop()
    elif CAMERA_TYPE == OPENCV:
        cam.release()
    if(__name__ =='__main__'):
        cv2.destroyAllWindows()
    

if __name__=='__main__':
    thread_main()
    
