import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import csv


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def _get_instance_segmentation_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Function to preprocess the frame
def preprocess_frame(frame):
    transform = T.Compose([T.ToTensor()])
    return transform(frame)


# Function to draw predictions on the frame
def draw_predictions(frame, boxes, labels, masks):
    for box, label, mask in zip(boxes, labels, masks):
        color = (0, 255, 0)  # Green color for bounding box and mask
        x1, y1, x2, y2 = box.astype(int)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        mask = mask > 0.5  # Threshold the mask probabilities
        mask = mask[0].cpu().numpy().astype("uint8") * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame = cv2.drawContours(frame, contours, -1, color, 2)
        frame = cv2.putText(
            frame, str(label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
    return frame


def main():
    # Load the instance segmentation model
    num_classes = 91  # Number of classes for COCO dataset
    model = get_instance_segmentation_model(num_classes)

    # Set the device for inference
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # Replace with the correct camera index if needed

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Preprocess the frame
        input_frame = preprocess_frame(frame)
        input_frame = input_frame.unsqueeze(0)  # Add a batch dimension
        input_frame = input_frame.to(device)

        # Perform inference
        with torch.no_grad():
            predictions = model(input_frame)

        # Process the predictions
        boxes = predictions[0]["boxes"].cpu().numpy()
        labels = predictions[0]["labels"].cpu().numpy()
        masks = predictions[0]["masks"]

        # Draw predictions on the frame
        frame = draw_predictions(frame, boxes, labels, masks)

        # Display the frame
        cv2.imshow("Instance Segmentation", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
