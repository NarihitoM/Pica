import subprocess
import threading

_GET = "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"
_SET = "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{})"


def _powershell(command: str) -> str:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return result.stdout.strip()


class BrightnessControl:
    """Steps display brightness up or down via WMI, on a worker thread -- each
    PowerShell call costs a few hundred ms and would otherwise stall the camera loop."""

    def __init__(self, step: int = 10):
        self.step = step
        self._level = self._read_level()
        self._target: int | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _read_level(self) -> int:
        try:
            return int(_powershell(_GET).splitlines()[0])
        except (ValueError, IndexError):
            return 50

    def up(self):
        self._adjust(self.step)

    def down(self):
        self._adjust(-self.step)

    def _adjust(self, delta: int):
        with self._lock:
            level = min(max(self._level + delta, 0), 100)
            if level == self._level:
                return
            self._level = level
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

            _powershell(_SET.format(target))

    def stop(self):
        self._running = False
        self._wake.set()
        self._thread.join(timeout=1)


def demo():
    control = BrightnessControl(step=10)
    control._running = False

    control._level = 50
    control.up()
    assert control._level == 60, control._level
    control.down()
    control.down()
    assert control._level == 40, control._level

    control._level = 95
    control.up()
    assert control._level == 100, "must clamp at 100"
    control.up()
    assert control._level == 100, "already at max, stays put"

    control._level = 5
    control.down()
    assert control._level == 0, "must clamp at 0"

    control.stop()
    print("brightness demo OK")


if __name__ == "__main__":
    demo()
