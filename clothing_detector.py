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

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("L + ratio")
        continue
    image.flags.writeable = False

    results = model.detect([image], verbose=0)

    print(results)

    cv2.imshow("Outfit Oracle", cv2.flip(image, 2))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
