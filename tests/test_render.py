"""Fase 5: video anotado de presentación y GIF de muestra."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from atlas_traffic.config import Geometria, Linea
from atlas_traffic.render import dibujar_panel, gif_desde_video, video_anotado

GEO = Geometria(
    encuadre="t",
    resolucion=(200, 160),
    roi=[(10, 10), (190, 10), (190, 150), (10, 150)],
    lineas=[Linea(id="av", puntos=((20, 80), (180, 80)), sentido_positivo="entrante",
                  sentido_permitido="entrante")],
)  # fmt: skip


@pytest.fixture
def video(tmp_path: Path) -> Path:
    ruta = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 160))
    for i in range(30):
        img = np.full((160, 200, 3), 100, np.uint8)
        cv2.rectangle(img, (5 * i, 120), (5 * i + 20, 140), (255, 255, 255), -1)  # algo se mueve
        w.write(img)
    w.release()
    return ruta


def _det():
    filas = [dict(frame=f, track_id=1, clase="vehiculo", x1=90, y1=40 + 3 * f, x2=110,
                  y2=55 + 3 * f) for f in range(0, 30, 2)]  # fmt: skip
    return pd.DataFrame(filas)


def test_video_anotado_tamano_y_cuadros(video, tmp_path):
    cruces = pd.DataFrame({"frame": [10], "track_id": [1], "aproximacion": ["av"],
                           "sentido": ["entrante"]})  # fmt: skip
    intervalos = pd.DataFrame({"t_inicio_s": [0.0], "t_fin_s": [3.0], "congestion": ["HIGH"],
                               "densidad_media": [12.0]})  # fmt: skip
    eventos = pd.DataFrame({"frame": [20]})
    out = video_anotado(video, _det(), GEO, tmp_path / "a.mp4", "t", cruces=cruces,
                        intervalos=intervalos, eventos=eventos, escala=1.5)  # fmt: skip
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 300
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 240
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 29  # hasta el último frame con detección
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    ok, frame = cap.read()
    cap.release()
    assert ok
    assert frame[2, 150, 2] > 150 and frame[2, 150, 0] < 100  # borde rojo del evento


def test_panel_dibuja_insignia_de_congestion():
    img = np.zeros((300, 500, 3), np.uint8)
    out = dibujar_panel(img, "t", 1.0, GEO, {("av", "entrante"): 3},
                        {"congestion": "HIGH", "densidad_media": 10.0}, escala=1.0)  # fmt: skip
    assert out.any()
    sin_nivel = dibujar_panel(img, "t", 1.0, GEO, {}, {"congestion": None}, escala=1.0)
    assert sin_nivel.any()  # sin congestión calibrada no rompe


def test_gif(video, tmp_path):
    from PIL import Image

    gif = gif_desde_video(video, tmp_path / "m.gif", inicio_s=0.5, duracion_s=2.0, ancho=100,
                          fps=5)  # fmt: skip
    img = Image.open(gif)
    assert img.size == (100, 80)
    assert img.n_frames == 10
    with pytest.raises(ValueError):
        gif_desde_video(video, tmp_path / "x.gif", inicio_s=50)
