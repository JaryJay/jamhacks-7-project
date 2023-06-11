import json
import os
import cv2
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn.model import MaskRCNN
import uuid
import argparse
import colorsys
import tensorflow as tf
import numpy as np
import shutil
import random
import argparse


class TestConfig(Config):
    NAME = "Deepfashion2"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 13


config = TestConfig()

model = modellib.MaskRCNN(
    mode="inference", config=config, model_dir=os.getcwd() + "/logs"
)
model.load_weights(os.getcwd() + "/clothing_trained_model.h5", by_name=True)

class_names = [
    "short_sleeved_shirt",
    "long_sleeved_shirt",
    "short_sleeved_outwear",
    "long_sleeved_outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short_sleeved_dress",
    "long_sleeved_dress",
    "vest_dress",
    "sling_dress",
    "",
]

score_threshold = 0.5  # Set the minimum score threshold
max_results = 2  # Set the maximum number of results for each class

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("L + ratio")
        continue
    image = cv2.flip(image, 2)
    image.flags.writeable = False

    results = model.detect([image], verbose=0)

    r = results[0]

    # Create a dictionary to store unique class labels and their corresponding bounding boxes
    unique_labels = {}

    for i in range(r["rois"].shape[0]):
        class_id = r["class_ids"][i]
        class_name = class_names[class_id]
        score = r["scores"][i]
        bbox = r["rois"][i]

        if score >= score_threshold:  # Filter out results below the threshold
            if class_name not in unique_labels:
                unique_labels[class_name] = []
            unique_labels[class_name].append((score, bbox))

    # Draw the limited number of bounding boxes and labels on the image
    for class_name, detections in unique_labels.items():
        detections = sorted(detections, key=lambda x: x[0], reverse=True)[
            :max_results
        ]  # Sort and limit detections

        color = (0, 255, 0)  # Green color for bounding boxes

        for score, bbox in detections:
            cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)

            label = "{}: {:.2f}".format(class_name, score)  # Move label creation here
            y = bbox[0] + 20
            cv2.putText(
                image,
                label,
                (bbox[1], y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    cv2.imshow("Outfit Oracle", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
