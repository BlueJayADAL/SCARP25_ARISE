#!/usr/bin/env python3

import os
import sys
import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from functools import partial
from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm
from pose_estimation_utils import PoseEstPostProcessing, output_data_type2dict


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

    @profile
    def run(self, post_processing: PoseEstPostProcessing):
        cap = cv2.VideoCapture(8)
        height, width, _ = self.get_input_shape()
        
        t = time.perf_counter()

        with self.infer_model.configure() as configured_infer_model:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera.")
                    break

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

                configured_infer_model.run([bindings], timeout=10000)

                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                        for name in bindings._output_names
                    }

                #poses = post_processing.fast_post_process(result, height, width)
                #visualized = post_processing.visualize_keypoints(poses, preprocessed)
                processed_image, raw_detections = frame, result
                #print(raw_detections)
                results = post_processing.post_process(raw_detections, height, width, 1)
                visualized = post_processing.visualize_pose_estimation_result(results, processed_image)
                cv2.imshow("Pose Estimation", cv2.cvtColor(visualized, cv2.COLOR_BGR2RGB))
                if cv2.waitKey(1) == ord("q"):
                    break
                print(f'FPS: {1/(time.perf_counter()-t)}')
                t=time.perf_counter()
                #input("FULL")

        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Pose Estimation with Hailo on Raspberry Pi 5")
    parser.add_argument("-n", "--net", default="yolov8s_pose.hef", help="Path to HEF file")
    parser.add_argument("-cn", "--class_num", default=1, type=int, help="Number of classes")
    return parser.parse_args()


def main():
    args = parse_args()
    hef_path = args.net
    class_num = args.class_num

    output_type_dict = output_data_type2dict(HEF(hef_path), "FLOAT32")

    post_processing = PoseEstPostProcessing(
        max_detections=300,
        score_threshold=0.001,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32]
    )

    infer = HailoSyncInference(
        hef_path=hef_path,
        output_type=output_type_dict
    )

    infer.run(post_processing)


if __name__ == "__main__":
    main()
