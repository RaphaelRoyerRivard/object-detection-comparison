import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import cv2
from matplotlib import pyplot as plt
from os import walk


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on", device)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(img_path, threshold):
    img = Image.open(img_path)  # Load the image
    transform = T.Compose([T.ToTensor()])  # Defing PyTorch Transform
    img = transform(img)  # Apply the transform to the image
    img = img.to(device)
    pred = model([img])  # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].detach().cpu().clone().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().clone().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().clone().numpy())
    valid_pred_indexes = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(valid_pred_indexes) == 0:
        return [], []
    pred_t = valid_pred_indexes[-1]  # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


def object_detection_api(images_root_path, threshold=0.5, save_result=True, show_plot=False, rect_th=3, text_size=0.5, text_th=1):
    for path, subfolders, files in walk(images_root_path):
        detected_classes = []
        detection_lines = ""
        print(path)
        filename = path.split("\\")[-1].split("/")[-1]
        if filename.startswith("_"):
            continue
        for file in files:
            if not file.endswith(".png") and not file.endswith(".jpg"):
                continue
            print(file)
            img_path = path + "/" + file
            boxes, pred_cls = get_prediction(img_path, threshold)  # Get predictions
            if show_plot and len(boxes) > 0:
                img = cv2.imread(img_path)  # Read image with cv2
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                for i in range(len(boxes)):
                    cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0),
                                  thickness=rect_th)  # Draw Rectangle with the coordinates
                    text_pos = (int(boxes[i][0][0] + 5), int(boxes[i][0][1] + 20))
                    cv2.putText(img, pred_cls[i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                                thickness=text_th)  # Write the prediction class
                plt.figure(figsize=(20, 30))  # display the output image
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                plt.show()
            if save_result:
                print(file)
                line = file
                for i in range(len(boxes)):
                    box = boxes[i]
                    line += " " + pred_cls[i]
                    if pred_cls[i] not in detected_classes:
                        detected_classes.append(pred_cls[i])
                    for point in box:
                        for coordinate in point:
                            line += " " + str(coordinate)
                detection_lines += line + "\n"

        if len(detected_classes) > 0:
            outfile = open(filename + ".detection.txt", "a")
            for detected_class in detected_classes:
                outfile.write(detected_class + ";")
            outfile.write("\n")
            outfile.write(detection_lines)
            outfile.close()


object_detection_api('../images/fall', threshold=0.8, save_result=False, show_plot=True)
