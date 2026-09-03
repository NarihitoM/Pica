import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("PICA_HOME", tempfile.mkdtemp(prefix="pica-tests-"))


@contextlib.contextmanager
def quiet():
    """Keeps progress prints and argparse usage text out of the test report."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield sink
