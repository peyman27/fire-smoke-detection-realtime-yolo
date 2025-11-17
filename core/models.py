from ultralytics import YOLO
import os
import glob
import numpy as np
import torch

def find_latest_model(models_dir="models"):
    runs_candidates = glob.glob(os.path.join("runs", "**", "weights", "best.pt"), recursive=True)
    if runs_candidates:
        latest_run = max(runs_candidates, key=os.path.getmtime)
        return os.path.abspath(latest_run)

    fallback = os.path.join(models_dir, "best.pt")
    if os.path.isfile(fallback):
        return os.path.abspath(fallback)

    return None


def load_model(model_path=None, device=None):
    if model_path is None:
        model_path = find_latest_model()
    if model_path is None:
        raise FileNotFoundError("❌ هیچ مدلی پیدا نشد! لطفاً فایل best.pt را در models/ بگذارید یا training انجام دهید.")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[models.py] ✅ Loading model: {model_path} on device {device}")
    model = YOLO(model_path)
    try:
        model.to(device)
    except Exception:
        pass

    return model, model_path


def detect_on_frame(model, frame, conf=0.35, iou=0.45, device=None, imgsz=640):
    kwargs = {"conf": conf, "iou": iou, "imgsz": imgsz}
    if device:
        kwargs["device"] = device

    results = model.predict(frame, **kwargs)
    if len(results) == 0:
        return {"boxes": [], "confs": [], "classes": [], "names": getattr(model, "names", {})}

    r = results[0]
    boxes, confs, classes = [], [], []
    names = getattr(model, "names", {})

    if hasattr(r, "boxes") and r.boxes is not None:
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf_arr = r.boxes.conf.cpu().numpy()
        cls_arr = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, cls in zip(xyxy, conf_arr, cls_arr):
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            confs.append(float(c))
            classes.append(int(cls))

    return {"boxes": boxes, "confs": confs, "classes": classes, "names": names}

