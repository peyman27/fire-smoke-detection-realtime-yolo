import argparse
import time
import cv2
import os
from core.camera import Camera
from core.models import load_model, detect_on_frame
from core.utils import draw_detections, save_alert_image, is_fire_or_smoke_detected
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", "-s", default=0, help="Camera source (0 for webcam or RTSP URL)")
    p.add_argument("--model", "-m", default=None, help="Path to model .pt (if omitted, newest in models/ will be used)")
    p.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    p.add_argument("--save-alerts", action="store_true", help="save alert images when fire/smoke detected")
    p.add_argument("--device", default=None, help="device e.g. 'cpu' or 'cuda:0' (auto if omitted)")
    return p.parse_args()

def main():
    args = parse_args()

    # auto device if not provided
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[main] Using device: {args.device}")

    # load model (will try to auto-find if args.model is None)
    model, model_path = load_model(args.model, device=args.device)
    print(f"[main] Model loaded from: {model_path}")

    # open camera
    print(f"[main] Opening source: {args.source}")
    cam = Camera(src=args.source)
    cv2.namedWindow("Fire & Smoke Detection", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("[main] No frame received, exiting.")
                break

            res = detect_on_frame(model, frame, conf=args.conf, imgsz=args.imgsz, device=args.device)
            boxes = res["boxes"]
            confs = res["confs"]
            classes = res["classes"]
            names = res["names"]

            detected_flag, labels = is_fire_or_smoke_detected(names, classes)

            vis = frame.copy()
            vis = draw_detections(vis, boxes, confs, classes, names)

            if detected_flag:
                cv2.putText(vis, "ALERT: FIRE/SMOKE DETECTED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                if args.save_alerts:
                    saved = save_alert_image(vis)
                    print(f"[ALERT] {labels} — saved: {saved}")

            cv2.imshow("Fire & Smoke Detection", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[main] Quit by user.")
                break

    except KeyboardInterrupt:
        print("[main] Interrupted by user.")
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
