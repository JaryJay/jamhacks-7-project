import json
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = os.getcwd() + "/efficientdet_lite0.tflite"

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(result: any, output_image: mp.Image, timestamp_ms: int):
    print("detection result: {}".format(result.detections))


options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    category_allowlist=["shirt"],
    result_callback=print_result,
)

cap = cv2.VideoCapture(0)
with ObjectDetector.create_from_options(options) as detector:
    print("Created object detector")
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("L + ratio")
            continue
        image.flags.writeable = False
        print(cap.get(cv2.CAP_PROP_POS_MSEC))
        print("===========" + str(type(image)))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detector.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        cv2.imshow("MediaPipe Pose", cv2.flip(image, 2))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
