# core/utils.py
import cv2
import os
import time

def draw_detections(img, boxes, confs, classes, names):
    for (box, conf, cls) in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box
        label = f"{names.get(cls, str(cls))} {conf:.2f}"
        color = (0, 0, 255) if names.get(cls) == "fire" else (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


def is_fire_or_smoke_detected(names, classes):
    """
    بررسی اینکه آیا آتش یا دود شناسایی شده است یا خیر.
    """
    detected_labels = [names.get(cls, str(cls)) for cls in classes]
    detected_fire = any(lbl.lower() == "fire" for lbl in detected_labels)
    detected_smoke = any(lbl.lower() == "smoke" for lbl in detected_labels)
    return (detected_fire or detected_smoke), detected_labels


def save_alert_image(frame, save_dir="alerts"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = f"alert_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(save_dir, filename)
    cv2.imwrite(path, frame)
    return path
