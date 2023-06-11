import json
import requests
from PIL import Image
from numpy import asarray, int32
import numpy as np


def save_data():
    f = open("data/data.json")

    data = json.load(f)
    print("BONK")

    desiredTrainImgNum = 60
    c = 1000000 // desiredTrainImgNum

    images = []
    labels = []

    for img_num, img_data in enumerate(data["images"]):
        if img_num % c == 0:
            img = Image.open(requests.get(img_data["url"], stream=True).raw)
            img = img.resize((64, 64))
            numpydata = asarray(img)
            images.append(numpydata)

    for annotation_num, annotation_data in enumerate(data["annotations"]):
        if annotation_num % c == 0:
            labelIds = annotation_data["labelId"]
            conversion = np.zeros(228, dtype=int32)
            for labelId in labelIds:
                conversion[int(labelId)] = 1
            labels.append(conversion)
    return np.stack(np.array(images)), np.array(labels)


def load_data():
    images = np.load("data/images.npy", allow_pickle=True)
    labels = np.load("data/labels.npy", allow_pickle=True)
    return images, labels


if __name__ == "__main__":
    images, labels = save_data()
    np.save("data/images.npy", images, allow_pickle=True)
    np.save("data/labels.npy", labels, allow_pickle=True)
