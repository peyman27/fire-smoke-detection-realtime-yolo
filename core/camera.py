# core/camera.py
"""
کلاس ساده برای گرفتن فریم از وبکم یا IP/RTSP camera.
استفاده:
cam = Camera(src=0)           # وبکم پیش‌فرض
cam = Camera(src="rtsp://...")# IP camera
ret, frame = cam.read()
cam.release()
"""
import cv2

class Camera:
    def __init__(self, src=0, width=None, height=None):
        """
        src: 0 یا مسیر RTSP/HTTP یا شماره دستگاه
        width, height: در صورت نیاز برای تنظیم اندازه ورودی
        """
        self.src = src
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"نشد دوربین را باز کرد: {src}")
        # تنظیم رزولوشن اگر خواسته شده
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        """برمی‌گرداند (ret, frame). frame به حال BGR (OpenCV) است."""
        return self.cap.read()

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
