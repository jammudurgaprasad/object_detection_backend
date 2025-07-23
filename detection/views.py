import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.conf import settings

COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    # (Trimmed for brevity — use your full dictionary here)
    90: 'toothbrush'
}

detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")

@api_view(['POST'])
def detect_objects(request):
    uploaded_file = request.FILES['image']
    path = default_storage.save(uploaded_file.name, uploaded_file)
    file_path = f"{settings.MEDIA_ROOT}/{path}"

    image_np = cv2.imread(file_path)
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
