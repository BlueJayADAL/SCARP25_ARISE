import cv2
import numpy as np
from ultralytics import YOLO
import math

OPENCV = 0
PICAM = 1

CAMERA_TYPE = OPENCV  

if CAMERA_TYPE == PICAM:
    from picamera2 import Picamera2

# Set up the camera with Picam
# if CAMERA_TYPE == PICAM:
#     picam2 = Picamera2()
#     picam2.preview_configuration.main.size = (1280, 1280)
#     picam2.preview_configuration.main.format = "RGB888"
#     picam2.preview_configuration.align()
#     picam2.configure("preview")
# # Or set up the camera with OpenCV
# elif CAMERA_TYPE == OPENCV:
#     cap = cv2.VideoCapture(0)  # Uncomment this line if you want to use OpenCV instead of Picamera2

# Load the YOLOv8 pose model
model = YOLO("models/yolo11n-pose_openvino_model_320")  # You can use yolov8s-pose.pt or better for accuracy
rep_done = False
reps = 0
angles = {}
coords = []

# Utility to calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    # Check for duplicate or invalid points
    if np.array_equal(a, b) or np.array_equal(b, c) or np.linalg.norm(a - b) == 0 or np.linalg.norm(c - b) == 0:
        return None  # or None if you'd rather skip the annotation

    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    if np.isnan(angle):
        return None
    return int(np.degrees(angle))
def A(a, b, c):
    return calculate_angle(coords[a], coords[b], coords[c])


# Annotate angle at a specific keypoint
def display_text(frame, text, position):
    cv2.putText(frame, str(text), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


def check_bicep_rep(coords, angles, side=2):
    global rep_done
    global reps
    both=0
    left=1
    right=2

    # do left side
    if side!=right and angles['left_elbow']!=None:
        if angles['left_elbow'] < 40 and not rep_done:
            rep_done = True
            reps += 1
        elif angles['left_elbow'] > 150:
            rep_done = False
    # do right side
    if side!=left and angles['right_elbow']!=None:
        if angles['right_elbow'] < 40 and not rep_done:
            rep_done = True
            reps += 1
        elif angles['right_elbow'] > 150:
            rep_done = False

def check_squat_rep(coords, angles):
    global rep_done
    global reps

    if angles['left_knee'] is not None and angles['right_knee'] is not None:
        if angles['left_knee'] < 90 and angles['right_knee'] < 90 and not rep_done:
            rep_done = True
            reps += 1
        elif angles['left_knee'] > 150 and angles['right_knee'] > 150:
            rep_done = False

def get_angles():
    global angles
    return angles


def thread_main(shared_data=None):
    global model
    global reps
    global rep_done
    global angles
    global coords
    current_exercise = "bicep"
    reps_threshold = 10

    # Initialize camera with Picamera2 or OpenCV
    cam = None
    if CAMERA_TYPE == PICAM:
        cam = Picamera2()
        cam.preview_configuration.main.size = (1280, 1280)
        cam.preview_configuration.main.format = "RGB888"
        cam.preview_configuration.align()
        cam.configure("preview")
        cam.start()
    elif CAMERA_TYPE == OPENCV:
        cam = cv2.VideoCapture(0) 

    while True:
        frame = None
        # Capture frame from Picamera2 or OpenCV
        if CAMERA_TYPE == PICAM:
            frame = cam.capture_array()
        elif CAMERA_TYPE == OPENCV:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        WIDTH = annotated_frame.shape[1]
        HEIGHT = annotated_frame.shape[0]

        
        #for pose in results[0].keypoints: # accounts for multiple people in frame
        pose = results[0].keypoints # only focuses on one person at a time
        keypoints = pose.data[0].cpu().numpy().reshape(-1, 3)
        coords = [(int(x), int(y)) for x, y, _ in keypoints]

        i = 0
        for point in coords:
            display_text(annotated_frame, f'{i}: {point[0],point[1]}', (5,10+20*i))
            i+=1

        if len(coords) < 17:
            continue

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
            'neck': keypoints[3][1] - keypoints[4][1],  
            'left_shoulder': a-90 if (a := A(6, 5, 7)) != None else None,
            'right_shoulder': a-90 if (a := A(5, 6, 8)) != None else None,
            'left_elbow': A(5, 7, 9),
            'right_elbow': A(6, 8, 10),
            'left_hip': A(5, 11, 13),
            'right_hip': A(6, 12, 14),
            'left_knee': A(11, 13, 15),
            'right_knee': A(12, 14, 16)
        }

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
        if current_exercise == 'bicep' or current_exercise == 'squat':
            if current_exercise == 'bicep':
                check_bicep_rep(coords, angles)
            elif current_exercise == 'squat':
                check_squat_rep(coords, angles)
            display_text(annotated_frame, f'Reps: {reps}', (WIDTH-100, HEIGHT-50))
            display_text(annotated_frame, f'Current Exercise: {current_exercise}', (int(WIDTH/2-100), 30))

            # Check if exercise complete
            if reps >= reps_threshold:
                current_exercise = 'complete'
        elif current_exercise == 'complete':
            display_text(annotated_frame, 'Exercise complete!', (int(WIDTH/2-100), 30))
        elif current_exercise == 'none':
            display_text(annotated_frame, 'No exercise selected', (int(WIDTH/2-100), 30))

        cv2.imshow("Pose with Angles", annotated_frame)

        # Handle quitting, key pressing
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('b'):
            current_exercise = 'bicep'
            reps = 0
        elif key & 0xFF == ord('s'):
            current_exercise = 'squat'
            reps = 0

    # Cleanup
    if CAMERA_TYPE == PICAM:
        cam.stop()
    elif CAMERA_TYPE == OPENCV:
        cam.release()
    cv2.destroyAllWindows()
    

if __name__=='__main__':
    thread_main()
    
