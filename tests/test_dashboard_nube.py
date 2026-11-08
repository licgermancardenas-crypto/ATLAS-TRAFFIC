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
assert len(at.get("video")) >= 1, "no hay reproductor de video"
etiquetas = [t.label for t in at.tabs]
assert any(t.startswith("Alertas") for t in etiquetas), etiquetas
print("ok")
"""


def test_dashboard_sin_opencv():
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=RAIZ, capture_output=True,
                       text=True, timeout=300)  # fmt: skip
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]


SCRIPT_SEGUIR = f"""
import sys
sys.modules["cv2"] = None
sys.modules["ultralytics"] = None
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
raiz = Path({str(RAIZ)!r})
datos = raiz / "outputs"
if not (datos / "summary.parquet").exists():
    datos = raiz / "demo" / "outputs"
cruces = pd.read_parquet(datos / "cruces_todos.parquet")
clip = sorted(p.parent.name for p in (datos / "runs").glob("*/trayectorias.parquet"))[0]
tid = int(cruces[(cruces.clip_id == clip) & (cruces.clase == "vehiculo")].track_id.iloc[0])
at = AppTest.from_file(str(raiz / "dashboard" / "app.py"), default_timeout=120)
at.query_params["vista"] = "seguir"
at.query_params["clip"] = clip
at.query_params["vehiculo"] = str(tid)
at.run()
assert not at.exception, [str(e.value) for e in at.exception]
assert not at.error, [e.value for e in at.error]
titulos = [s.value for s in at.subheader]
assert any(t.startswith(f"Vehículo #{{tid}}") for t in titulos), titulos
assert len(at.metric) == 5
print("ok")
"""


def test_seguir_un_vehiculo_sin_opencv():
    r = subprocess.run([sys.executable, "-c", SCRIPT_SEGUIR], cwd=RAIZ, capture_output=True,
                       text=True, timeout=300)  # fmt: skip
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]
