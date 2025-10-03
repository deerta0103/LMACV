#!/usr/bin/env python3

import re
import os
import cv2
import numpy as np
from tqdm import tqdm
import onnxruntime as ort

# 类别名称字典
CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
           8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
           14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
           22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
           29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
           35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
           40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
           48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
           55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
           62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
           69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
           76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, confidence_thres, iou_thres, target_classes=None):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
            target_classes: List of class names to detect. If None, detect all classes.
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.target_classes = target_classes

        # Create an inference session using the ONNX model and specify execution providers
        self.session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider"])
        # 打印当前推理模型的device
        providers = self.session.get_providers()
        if 'CUDAExecutionProvider' in providers:
            print("ONNX模型推理在GPU上运行。")
        else:
            print("ONNX模型推理在CPU上运行。")

        # Load the class names
        self.classes = CLASSES

        # 如果指定了目标类别，则创建类别ID到名称的映射
        self.target_class_ids = None
        if self.target_classes:
            self.target_class_ids = {idx: name for idx, name in self.classes.items() if name in self.target_classes}
            print(f"只检测以下类别: {self.target_classes}")

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def preprocess(self, image: np.ndarray):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        self.img = image

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            list: List of detections in the format (score, class_id, x_min, y_min, x_max, y_max)
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        detections = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # 如果指定了目标类别，检查当前类别是否在目标类别中
                if self.target_class_ids is not None and class_id not in self.target_class_ids:
                    continue

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                detections.append((max_score, class_id, left, top, left + width, top + height))

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        detections = self.nms(detections)

        return detections

    def nms(self, detections):
        """
        Applies non-maximum suppression to filter out overlapping bounding boxes.

        Args:
            detections: List of detections in the format (score, class_id, x_min, y_min, x_max, y_max)

        Returns:
            list: List of filtered detections
        """
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x[0], reverse=True)
        keep = []

        while detections:
            current = detections.pop(0)
            keep.append(current)
            detections = [d for d in detections if self.iou(current[2:], d[2:]) < self.iou_thres]

        return keep

    def iou(self, box1, box2):
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1: (x_min, y_min, x_max, y_max)
            box2: (x_min, y_min, x_max, y_max)

        Returns:
            float: IoU value
        """
        x_min = max(box1[0], box2[0])
        y_min = max(box1[1], box2[1])
        x_max = min(box1[2], box2[2])
        y_max = min(box1[3], box2[3])

        intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection / float(area1 + area2 - intersection)

    def main(self, data: np.ndarray):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            list: List of detections in the format (score, class_id, x_min, y_min, x_max, y_max)
        """
        # Get the model inputs
        model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Preprocess the image data
        img_data = self.preprocess(data)

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain detections
        return self.postprocess(data, outputs)


def custom_sort_key(filename):
    # 使用正则表达式分割文件名中的字母和数字部分
    parts = re.split(r'(\d+)', filename)
    # 创建一个元组，其中字母部分保持原样，数字部分转换为整数
    sorted_parts = []
    for part in parts:
        if part.isdigit():
            sorted_parts.append(int(part))
    return tuple(sorted_parts)


def save_detections(detections, save_path):
    """
    Saves detections to a file.

    Args:
        detections: List of detections in the format (score, class_id, x_min, y_min, x_max, y_max)
        save_path: Path to save the detections file
    """
    with open(save_path, 'w') as f:
        for detection in detections:
            score, class_id, x_min, y_min, x_max, y_max = detection
            f.write(f"{score},{class_id},{x_min},{y_min},{x_max},{y_max}\n")


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/he/jiance/object_detect_model_m.onnx", help="Input your ONNX model.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--target-classes", type=str, nargs='+', default=None,
                        help="List of target classes to detect (e.g., person car)")
    args = parser.parse_args()

    # 图片读取路径
    image_path = "/home/he/jiance/dpcnn"   # 需要修改为你自己的图片文件夹路径
    save_root_path = "/home/he/jiance"  # 修改为你想要存储检测结果的路径

    # Create an instance of the YOLOv8 class with the specified arguments
    target_classes = ['person', 'car'] if args.target_classes is None else args.target_classes
    detection = YOLOv8(args.model, args.conf_thres, args.iou_thres, target_classes)

    image_list = os.listdir(image_path)
    image_list = sorted(image_list, key=custom_sort_key)
    for image in tqdm(image_list):
        image_dir = os.path.join(image_path, image)
        save_path = os.path.join(save_root_path, 'detections', os.path.splitext(image)[0] + '.txt')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        frame = cv2.imread(image_dir)
        if frame is None:
            print(f"无法读取图像: {image_dir}")
            continue
        print(f"处理图像: {image}, 尺寸: {frame.shape}")
        detections = detection.main(data=frame)
        save_detections(detections, save_path)