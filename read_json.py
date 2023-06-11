import json
import requests
from PIL import Image
from numpy import asarray


def get_data():
    f = open("data.json")

    data = json.load(f)

    desiredTrainImgNum = 5000
    c = 1000000 // desiredTrainImgNum

    listOfImgs = []
    listOfLabels = []

    for img_num, img_data in enumerate(data["images"]):
        if img_num % c == 0:
            img = Image.open(requests.get(img_data["url"], stream=True).raw)
            img = img.resize((256, 256))
            numpydata = asarray(img)
            listOfImgs.append(numpydata)

    for annotation_num, annotation_data in enumerate(data["annotations"]):
        if annotation_num % c == 0:
            labelIds = annotation_data.labelId
            listOfLabels.append(labelIds)

    result = []
    for i in range(len(listOfImgs)):
        result.append({"img": listOfImgs[i], "labels": listOfLabels[i]})
    return result
