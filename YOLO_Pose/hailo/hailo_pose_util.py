
import cv2
import time
import queue
import threading
import numpy as np
from pathlib import Path
from PIL import Image
from functools import partial
from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm

from YOLO_Pose.hailo.pose_estimation_utils import PoseEstPostProcessing, output_data_type2dict

# Conditional import for testing purposes, if running directly 
from YOLO_Pose.shared_data import SharedState
from YOLO_Pose.exercise_forms import check_bad_form


OPENCV = 0
PICAM = 1

SYNCHRONOUS = 0
QUEUED = 1


def hailo_init(shared_data, camera_type, inference_method):
    global postprocessing_data
    global postprocess_queue
    global hef_path
    global hailo_model
    global post_processing
    global postprocessing_data
    global postprocess_queue
    global CAMERA_TYPE
    global INFERENCE_METHOD
    
    CAMERA_TYPE = camera_type
    INFERENCE_METHOD = inference_method
    
    # Initialize Hailo hardware and environment
    hef_path = 'models/yolov8m_pose.hef'
    class_num = 1
    output_type_dict = output_data_type2dict(HEF(hef_path), "FLOAT32")
    post_processing = PoseEstPostProcessing(
        max_detections=300,
        score_threshold=0.001,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32]
    )
    hailo_model = HailoSyncInference(
        hef_path=hef_path,
        output_type=output_type_dict
    )
    if INFERENCE_METHOD == QUEUED:        
        postprocessing_data = None
        postprocess_queue = queue.Queue(maxsize=2)

        inference_thread = threading.Thread(target=hailo_model.threaded_runnable_raw_output, args=(post_processing,shared_data), daemon=True)
        inference_thread.start()
        
        
    # Give OpenCV time to boot or inference thread time to process...? 
    # Not sure, but thread will crash without thread sleep
    time.sleep(1)

def get_postprocess():
    return postprocess_queue.get(timeout=0.1)

def postprocess(post_data):
    global hailo_model
    global post_processing
    return hailo_model.postprocess_raw_output(post_data, post_processing)
    
def hailo_sync_infer(frame):
    return hailo_model.run_single_inference(post_processing, frame)

class HailoSyncInference:
    def __init__(self, hef_path: str, batch_size: int = 1, output_type: dict = None):
        self.hef = HEF(hef_path)
        self.params = VDevice.create_params()
        self.params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.target = VDevice(self.params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)
        if output_type is not None:
            self._set_output_type(output_type)
        self.output_type = output_type

    def _set_output_type(self, output_type_dict: dict) -> None:
        for output_name, output_type in output_type_dict.items():
            self.infer_model.output(output_name).set_format_type(getattr(FormatType, output_type))

    def get_input_shape(self) -> tuple:
        return self.hef.get_input_vstream_infos()[0].shape
        
    # run on second thread
    def threaded_runnable_raw_output(self, post_processing: PoseEstPostProcessing, shared_data, single_run=False):
        global postprocessing_data
        global CAMERA_TYPE
        global cam
        global postprocess_queue
        # Initialize camera if using OpenCV
        if CAMERA_TYPE == OPENCV:
            cam = cv2.VideoCapture(0)
            ret, frame = cam.read()
            if not ret:
                print("Switching USB camera port to 8")
                cam = cv2.VideoCapture(8)
                
        while True:
         #   print('got_frame1')
            if CAMERA_TYPE == PICAM:
                frame = cam.capture_array()
            elif CAMERA_TYPE == OPENCV:
                #print(cam.isOpened())
                ret, frame = cam.read()
                if not ret:
                    print("Failed to grab frame")
                    break
         #   print('got_frame2')
            #shared_data.set_value('postprocessing_data', self.run_single_inference(post_processing, frame, do_postprocess=False))
            data=self.run_single_inference(post_processing, frame, do_postprocess=False)
            try:
                postprocess_queue.put(data, timeout=0.5)
            except queue.Full:
         #       print('second queue full')
                continue
            
            if single_run:
                return
            # while shared_data.get_value('postprocessing_data') is not None:
                # print('stalling second')
                # time.sleep(0.1)
            
    
    #run on main thread
    # raw_output: tuple of output from threaded_runnable_raw_output: (result (raw_detections), preprocessed)
    def postprocess_raw_output(self, raw_output, post_processing: PoseEstPostProcessing):
       # print('got_frame3')
        height, width, _ = self.get_input_shape()
       # print('got_frame4')
        results = post_processing.post_process(raw_output[0], height, width, 1)
        #print('got_frame5')
        visualized, keypoints = post_processing.visualize_pose_estimation_result(results, raw_output[1])
       # print('got_frame6')
        #input(keypoints)
        return keypoints, visualized

    def run_single_inference(self, post_processing: PoseEstPostProcessing, frame, do_postprocess=True):
        height, width, _ = self.get_input_shape()
        
        with self.infer_model.configure() as configured_infer_model:
            # Preprocessing
            image = Image.fromarray(frame)
            preprocessed = post_processing.preprocess(image, width, height)
            preprocessed_np = np.array(preprocessed)
    
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape,
                    dtype=(getattr(np, output_type.lower()))
                ) for name, output_type in self.output_type.items()
            }
    
            bindings = configured_infer_model.create_bindings(output_buffers=output_buffers)
            bindings.input().set_buffer(preprocessed_np)
    
            # Inference
            configured_infer_model.run([bindings], timeout=10000)
    
            # Postprocessing
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                    for name in bindings._output_names
                }
            
            if do_postprocess:
                raw_detections = result
                results = post_processing.post_process(raw_detections, height, width, 1)
                visualized, keypoints = post_processing.visualize_pose_estimation_result(results, preprocessed)
                return keypoints, visualized
            else:
                return (result, preprocessed)

    def run(self, post_processing: PoseEstPostProcessing):
        cap = cv2.VideoCapture(8)
        height, width, _ = self.get_input_shape()
        
        count = 0
        pre_times, infer_times, post_times, misc_times, total_times = [],[],[],[],[]
        t = time.perf_counter()
        start_t = t
        with self.infer_model.configure() as configured_infer_model:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera.")
                    break
                    
                # Preprocessing
                image = Image.fromarray(frame)
                preprocessed = post_processing.preprocess(image, width, height)
                preprocessed_np = np.array(preprocessed)
                
                output_buffers = {
                    name: np.empty(
                        self.infer_model.output(name).shape,
                        dtype=(getattr(np, output_type.lower()))
                    ) for name, output_type in self.output_type.items()
                }

                bindings = configured_infer_model.create_bindings(output_buffers=output_buffers)
                bindings.input().set_buffer(preprocessed_np)
                
                # Inference
                configured_infer_model.run([bindings], timeout=10000)

                # Postprocessing
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                        for name in bindings._output_names
                    }
                
                raw_detections = result
                results = post_processing.post_process(raw_detections, height, width, 1)
                visualized, keypoints = post_processing.visualize_pose_estimation_result(results, preprocessed)
                cv2.imshow("Pose Estimation", visualized)
                if cv2.waitKey(1) == ord("q"):
                    break
                    
                # Timing checks
                print(f'FPS: {1/(time.perf_counter()-t)}')
                t = time.perf_counter()
               
        
        cap.release()
        cv2.destroyAllWindows()
