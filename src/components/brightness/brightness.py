import subprocess
import threading

_COMMAND = "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{})"


class BrightnessControl:
    """Sets display brightness via WMI, on a worker thread -- each PowerShell
    call costs a few hundred ms and would otherwise stall the camera loop."""

    def __init__(self, step: int = 5):
        self.step = step
        self._target: int | None = None
        self._applied: int | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def set(self, percent: float):
        level = min(max(int(percent), 0), 100)
        with self._lock:
            if self._applied is not None and abs(level - self._applied) < self.step:
                return
            self._target = level
        self._wake.set()

    def _worker(self):
        while self._running:
            self._wake.wait()
            self._wake.clear()

            with self._lock:
                target = self._target
            if target is None:
                continue

            subprocess.run(
                ["powershell", "-NoProfile", "-Command", _COMMAND.format(target)],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            with self._lock:
                self._applied = target

    def stop(self):
        self._running = False
        self._wake.set()
        self._thread.join(timeout=1)


def demo():
    control = BrightnessControl(step=5)
    control._running = False
    control.set(50)
    assert control._target == 50
    control._applied = 50
    control.set(52)
    assert control._target == 50, "jitter under one step must be ignored"
    control.set(70)
    assert control._target == 70
    control.stop()
    print("brightness demo OK")


if __name__ == "__main__":
    demo()
