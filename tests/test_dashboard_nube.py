"""El dashboard publicado (Streamlit Community Cloud) corre sin OpenCV ni YOLO.

Se prueba en un proceso aparte, con OpenCV bloqueado, que todas las pestañas se dibujan
sin errores y que la pestaña Video tiene su reproductor. Así ya rompió una vez: una
sección nueva importaba un módulo con OpenCV y el dashboard se cortaba antes del video.
"""

import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

SCRIPT = f"""
import sys
sys.modules["cv2"] = None
sys.modules["ultralytics"] = None
sys.modules["torch"] = None
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({str(RAIZ / "dashboard" / "app.py")!r}, default_timeout=120)
at.run()
assert not at.exception, [str(e.value) for e in at.exception]
errores = [e.value for e in at.error]
assert not errores, errores
assert len(at.get("video")) == 1, "la pestaña Video no tiene reproductor"
print("ok")
"""


def test_dashboard_sin_opencv():
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=RAIZ, capture_output=True,
                       text=True, timeout=300)  # fmt: skip
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]
