import numpy as np
import matplotlib.pyplot as plt
from hailo_platform import HEF, VDevice, HailoSchedulingAlgorithm
import cv2
import time
import sys
import subprocess
from picamera2 import Picamera2

cam = Picamera2()
cam.preview_configuration.main.size = (1280, 1280)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.align()
cam.configure("preview")
cam.start()

labels = {0: '_BACKGROUND_',
 1: 'person',
 2: 'bicycle',
 3: 'car',
 4: 'motorcycle',
 5: 'airplane',
 6: 'bus',
 7: 'train',
 8: 'truck',
 9: 'boat',
 10: 'traffic light',
 11: 'fire hydrant',
 12: 'stop sign',
 13: 'parking meter',
 14: 'bench',
 15: 'bird',
 16: 'cat',
 17: 'dog',
 18: 'horse',
 19: 'sheep',
 20: 'cow',
 21: 'elephant',
 22: 'bear',
 23: 'zebra',
 24: 'giraffe',
 25: 'backpack',
 26: 'umbrella',
 27: 'handbag',
 28: 'tie',
 29: 'suitcase',
 30: 'frisbee',
 31: 'skis',
 32: 'snowboard',
 33: 'sports ball',
 34: 'kite',
 35: 'baseball bat',
 36: 'baseball glove',
 37: 'skateboard',
 38: 'surfboard',
 39: 'tennis racket',
 40: 'bottle',
 41: 'wine glass',
 42: 'cup',
 43: 'fork',
 44: 'knife',
 45: 'spoon',
 46: 'bowl',
 47: 'banana',
 48: 'apple',
 49: 'sandwich',
 50: 'orange',
 51: 'broccoli',
 52: 'carrot',
 53: 'hot dog',
 54: 'pizza',
 55: 'dont',
 56: 'cake',
 57: 'chair',
 58: 'couch',
 59: 'potted plant',
 60: 'bed',
 61: 'dining table',
 62: 'toilet',
 63: 'tv',
 64: 'laptop',
 65: 'mouse',
 66: 'remote',
 67: 'keyboard',
 68: 'cell phone',
 69: 'microwave',
 70: 'oven',
 71: 'toaster',
 72: 'sink',
 73: 'refrigerator',
 74: 'book',
 75: 'clock',
 76: 'vase',
 77: 'scissors',
 78: 'teddy bear',
 79: 'hair drier',
 80: 'toothbrush'}

# Add helper functions
def get_input_shape():
#     input_infos = hef.get_input_vstream_infos()
#     info = input_infos[0]  # Get the first vstream info object
#     print(dir(info))
    return hef.get_input_vstream_infos()[0].shape

def show_image(image: np.ndarray):
    # Show the image
    plt.imshow(image)
    plt.axis('off')  # Hide axis ticks
    plt.show()

def preprocess(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    """
    Resize image with unchanged aspect ratio using padding.

    Args:
        image (np.ndarray): Input image.
        model_w (int): Model input width.
        model_h (int): Model input height.

    Returns:
        np.ndarray: Preprocessed and padded image.
    """
    # Convert and backup the original image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    padding_color = (114, 114, 114)
    
    img_h, img_w, _ = image.shape[:3]
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
    image = cv2.resize(image, (new_img_w, new_img_h), interpolation=cv2.INTER_CUBIC)

    padded_image = np.full((model_h, model_w, 3), padding_color, dtype=np.uint8)
    x_offset = (model_w - new_img_w) // 2
    y_offset = (model_h - new_img_h) // 2
    padded_image[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = image
    return padded_image

def _get_output_type_str(output_info) -> str:
    return str(output_info.format.type).split(".")[1].lower()

def extract_detections(input_data: list, threshold: float = 0.5) -> dict:
    """
    Extract detections from the input data.

    Args:
        input_data (list): Raw detections from the model.
        threshold (float): Score threshold for filtering detections. Defaults to 0.5.

    Returns:
        dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
    """
    boxes, scores, classes = [], [], []
    num_detections = 0

    for i, detection in enumerate(input_data):
        if len(detection) == 0:
            continue

        for det in detection:
            print("LEN: "+str(len(det)))
            bbox, score = det[:4], det[4]

            if score >= threshold:
                boxes.append(bbox)
                scores.append(score)
                classes.append(i)
                num_detections += 1

    return {
        'detection_boxes': boxes, 
        'detection_classes': classes, 
        'detection_scores': scores,
        'num_detections': num_detections
    }
    
def generate_color(class_id: int) -> tuple:
    """
    Generate a unique color for a given class ID.

    Args:
        class_id (int): The class ID to generate a color for.

    Returns:
        tuple: A tuple representing an RGB color.
    """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())

def denormalize_and_rm_pad(box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
    """
    Denormalize bounding box coordinates and remove padding.

    Args:
        box (list): Normalized bounding box coordinates.
        size (int): Size to scale the coordinates.
        padding_length (int): Length of padding to remove.
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.

    Returns:
        list: Denormalized bounding box coordinates with padding removed.
    """
    for i, x in enumerate(box):
        box[i] = int(x * size)
        if (input_width != size) and (i % 2 != 0):
            box[i] -= padding_length
        if (input_height != size) and (i % 2 == 0):
            box[i] -= padding_length

    return box    

def draw_detection(image: np.ndarray, box: list, cls: int, score: float, color: tuple, scale_factor: float):
    """
    Draw box and label for one detection.

    Args:
        image (np.ndarray): Image to draw on.
        box (list): Bounding box coordinates.
        cls (int): Class index.
        score (float): Detection score.
        color (tuple): Color for the bounding box.
        scale_factor (float): Scale factor for coordinates.
    """
    #labels = ['red-ball', 'blue-ball']
    
    label = f"{labels[cls+1]}: {score:.2f}%"
    ymin, xmin, ymax, xmax = box
    ymin, xmin, ymax, xmax = int(ymin * scale_factor), int(xmin * scale_factor), int(ymax * scale_factor), int(xmax * scale_factor)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, label, (xmin + 4, ymin + 20), font, 0.5, color, 1, cv2.LINE_AA)

def draw_detections(detections: dict, image: np.ndarray, min_score: float = 0.45, scale_factor: float = 1):
    """
    Draw detections on the image.

    Args:
        detections (dict): Detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
        image (np.ndarray): Image to draw on.
        min_score (float): Minimum score threshold. Defaults to 0.45.
        scale_factor (float): Scale factor for coordinates. Defaults to 1.

    Returns:
        np.ndarray: Image with detections drawn.
    """
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']

    # Values used for scaling coords and removing padding
    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    for idx in range(detections['num_detections']):
        if scores[idx] >= min_score:
            color = generate_color(classes[idx])
            scaled_box = denormalize_and_rm_pad(boxes[idx], size, padding_length, img_height, img_width)
            draw_detection(image, scaled_box, classes[idx], scores[idx] * 100.0, color, scale_factor)

    return image

# Initialize HailoRT Start #

HEF_PATH = "./SCARP25_ARISE/models/yolov8m_pose.hef"

params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

vdevice = VDevice(params)
infer_model = vdevice.create_infer_model(HEF_PATH)
infer_model.set_batch_size(1) # batch size 1 for realtime application

hef = HEF(HEF_PATH)

# Get input shape of the model (could be optimized out later)
height, width, _ = get_input_shape()

# Configure the infered model
configured_infer_model = infer_model.configure()

print(hef.get_output_vstream_infos())

# Configure output buffers
output_buffers = {output_info.name : np.empty(infer_model.output(output_info.name).shape,
    dtype=(getattr(np, _get_output_type_str(output_info))))       
    for output_info in hef.get_output_vstream_infos()
}
bindings = configured_infer_model.create_bindings(output_buffers=output_buffers)

# Initialize HailoRT End #

# Initialize OpenCV Start #

## DOESNT WORK, OpenCV overrides V4L2 setting at start
# Use terminal command to set the camera settings
# opencv functions to set v4l2 configuration is not reliable
# subprocess.run("""
# v4l2-ctl -d /dev/video0 --set-fmt-video=pixelformat=MJPG
# v4l2-ctl -d /dev/video0 --set-fmt-video=width=1280,height=720
# v4l2-ctl -d /dev/video0 --set-parm=30
# v4l2-ctl -d /dev/video0 --set-ctrl exposure_dynamic_framerate=0 
# """, shell=True, check=True)

#source = cv2.VideoCapture(8)

# if not source.isOpened():
    # vdevice.release()
    # print("Error: Could not open video source.")
    # sys.exit(1)

# source.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# source.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
# source.set(cv2.CAP_PROP_FPS, 30)

# Check if the settings were applied successfully
# cam_width = source.get(cv2.CAP_PROP_FRAME_WIDTH)
# cam_height = source.get(cv2.CAP_PROP_FRAME_HEIGHT)
# fps = source.get(cv2.CAP_PROP_FPS)

cam_width = 1280
cam_height = 1280
fps = 30

print(f"Resolution: {cam_width}x{cam_height}")
print(f"Frame rate: {fps}")
# Initialize OpenCV End #

def parse_yolo_pose_output(output_tensor, num_classes, num_keypoints, confidence_threshold=0.4):
    detections = []
    for detection in output_tensor:
        # x, y, w, h = detection[:4]
        # obj_conf = detection[4]
        # class_probs = detection[5:5+num_classes]
        # class_id = np.argmax(class_probs)
        # class_conf = class_probs[class_id]

        # if obj_conf * class_conf < confidence_threshold:
            # continue

        # bbox = {
            # 'x': x,
            # 'y': y,
            # 'w': w,
            # 'h': h,
            # 'confidence': obj_conf * class_conf,
            # 'class_id': class_id
        # }

        # keypoints = []
        # offset = 5 + num_classes
        # for i in range(num_keypoints):
            # kx = detection[offset + i*3]
            # ky = detection[offset + i*3 + 1]
            # kconf = detection[offset + i*3 + 2]
            # keypoints.append({'x': kx, 'y': ky, 'confidence': kconf})

        # detections.append({'bbox': bbox, 'keypoints': keypoints})
        
        keypoints = []
        for i in range(num_keypoints):
            kx = detection[i*3]
            ky = detection[i*3 + 1]
            kconf = detection[i*3 + 2]
            keypoints.append({'x': kx, 'y': ky, 'confidence': kconf})

        detections.append(keypoints)
    return detections

try:
    while cv2.waitKey(1) != 27:
        print('start')
        start = time.perf_counter()
        # has_frame, frame = source.read()

        # if not has_frame:
            # break
        frame = cam.capture_array()
        print('mid')

        original_img = frame.copy()
        processed_img = preprocess(frame, width, height)

        # Place the image in the model input buffer
        bindings.input().set_buffer(np.array(processed_img))
        
        print(processed_img.shape)
        print(processed_img)
        
        # Run the inference
        configured_infer_model.run([bindings], 1000)
        buffer = bindings.output(name='yolov8m_pose/conv92').get_buffer()

        infer_results = buffer
            
        # detections = extract_detections(infer_results)
        # print(detections)

        # Extract the center of the bounding box for object class 0
        # for idx, cls in enumerate(detections['detection_classes']):
            # if cls == 0:  # Check if the class is 0
                # ymin, xmin, ymax, xmax = detections['detection_boxes'][idx]
                # center_x = (xmin + xmax) / 2
                # center_y = (ymin + ymax) / 2

                # # Scale normalized coordinates to actual screen coordinates
                # actual_center_x = int(center_x * cam_width)  # Multiply by frame width
                # actual_center_y = int(center_y * cam_height)  # Multiply by frame height

                # print(f"Center of bounding box for class 0: ({actual_center_x}, {actual_center_y})")
        
        # frame_with_detections = draw_detections(detections, original_img)
     #   input(infer_results)
     #   input(infer_results.shape)
        # output_tensor = infer_results['yolov8m_pose/conv92'].reshape(-1)
        output_tensor = infer_results.reshape(-1)
      #  input(output_tensor)
        # detection_size = num_keypoints * 3
        detection_size = 17 * 3
        num_detections = output_tensor.shape[0] // detection_size
        output_tensor = output_tensor.reshape((num_detections, detection_size))
      #  input(output_tensor.shape)

        #detections = parse_yolo_pose_output(output_tensor, num_classes, num_keypoints)
        detections = parse_yolo_pose_output(output_tensor, 1, 17)

        for det_kps in detections:
            for kp in det_kps:
                if kp['confidence'] > 0.5:
                    cv2.circle(frame, (int(kp['x']*1280), int(kp['y']*1280)), 3, (0, 0, 255), -1)
        
        end = time.perf_counter()
        print(f"Execution time: {end - start:.6f} seconds")

        #cv2.imshow("window1", frame_with_detections)
        cv2.imshow("window1", frame)
    
    vdevice.release()
    #source.release()
    cam.stop()
    cv2.destroyWindow("window1")

except KeyboardInterrupt:
    print("Keyboard Interrupt: Exiting")
    vdevice.release()
    source.release()
    cv2.destroyWindow("window1")
