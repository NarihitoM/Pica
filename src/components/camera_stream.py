import threading

import cv2


class CameraStream:
    """Threaded webcam reader so the main loop never blocks on cap.read()."""

    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {camera_id}")

        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = threading.Thread(target=self._update, daemon=True)

    def start(self) -> "CameraStream":
        self._running = True
        self._thread.start()
        return self

    def _update(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


def demo():
    # smoke check: bad camera id must raise, not hang
    try:
        CameraStream(camera_id=9999)
        raise AssertionError("expected RuntimeError for invalid camera id")
    except RuntimeError:
        pass
    print("camera_stream demo OK")


if __name__ == "__main__":
    demo()
