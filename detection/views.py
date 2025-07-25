import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.conf import settings

COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")

# @api_view(['POST'])
# def detect_objects(request):
#     uploaded_file = request.FILES['image']
#     path = default_storage.save(uploaded_file.name, uploaded_file)
#     file_path = f"{settings.MEDIA_ROOT}/{path}"

#     image_np = cv2.imread(file_path)
#     image_np = cv2.resize(image_np, (640, 480))
@api_view(['POST'])
def detect_objects(request):
    uploaded_file = request.FILES['image']
    path = default_storage.save(uploaded_file.name, uploaded_file)
    file_path = f"{settings.MEDIA_ROOT}/{path}"

    image_np = cv2.imread(file_path)
    if image_np is None:
        return Response({"error": "Failed to read uploaded image"}, status=400)

    image_np = cv2.resize(image_np, (640, 480))
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)[tf.newaxis, ...]
    detections = detector(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    result_names = []
    for i in range(len(scores)):
        if scores[i] >= 0.5:
            label = COCO_LABELS.get(class_ids[i], 'Unknown')
            result_names.append(label)

    return Response({"objects": list(set(result_names))})
