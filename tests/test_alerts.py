"""Fase 3c: alerta de sentido contrario y sus tres filtros antifalso-positivo."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atlas_traffic.alerts import (
    COLUMNAS_EVENTOS,
    cruce_limpio,
    detectar_sentido_contrario,
    guardar_evidencia,
)
from atlas_traffic.config import AlertasCfg, Geometria, Linea

# línea horizontal y=100; A->B a la derecha => lado positivo = abajo.
# Cruzar hacia abajo (positivo) = "entrante"; es lo único permitido.
LINEA = Linea(id="av", puntos=((0, 100), (400, 100)), sentido_positivo="entrante",
              sentido_permitido="entrante")  # fmt: skip
GEO = Geometria(encuadre="t", resolucion=(400, 400), roi=[(0, 0), (400, 0), (400, 400), (0, 400)],
                lineas=[LINEA])  # fmt: skip
CFG = AlertasCfg(min_frames_historia=15, min_desplazamiento_px=40, frames_limpios=3)


def _track(tid, ys, x=200.0):
    ys = np.asarray(ys, dtype=float)
    f = np.arange(len(ys)) * 2
    return pd.DataFrame({
        "track_id": tid, "frame": f, "ts": f / 60, "clase": "vehiculo", "clase_detalle": "car",
        "cx_ref": x, "cy_ref": ys, "x1": int(x) - 10, "y1": (ys - 8).astype(int),
        "x2": int(x) + 10, "y2": (ys + 8).astype(int),
    })  # fmt: skip


def test_sentido_permitido_no_alerta():
    det = _track(1, np.linspace(40, 160, 20))  # baja: entrante, permitido
    assert detectar_sentido_contrario(det, GEO, CFG, "c").empty


def test_sentido_contrario_alerta():
    det = _track(2, np.linspace(160, 40, 20))  # sube: saliente, prohibido
    ev = detectar_sentido_contrario(det, GEO, CFG, "c")
    assert list(ev.columns) == COLUMNAS_EVENTOS
    assert len(ev) == 1
    assert ev.loc[0, "tipo"] == "sentido_contrario"
    assert ev.loc[0, "aproximacion"] == "av" and ev.loc[0, "sentido"] == "saliente"
    assert ev.loc[0, "clip_id"] == "c" and ev.loc[0, "track_id"] == 2


def test_filtro_historia_corta():
    det = _track(3, np.linspace(160, 40, 10))  # 10 detecciones < 15
    assert detectar_sentido_contrario(det, GEO, CFG, "c").empty


def test_filtro_desplazamiento_minimo():
    # cruza la línea "hacia arriba" pero en total se mueve 30 px (< 40): temblor/quieto
    det = _track(4, list(np.linspace(115, 85, 16)))
    assert detectar_sentido_contrario(det, GEO, CFG, "c").empty
    assert len(detectar_sentido_contrario(det, GEO, CFG.model_copy(
        update={"min_desplazamiento_px": 10}), "c")) == 1  # fmt: skip


def test_filtro_cruce_limpio():
    # solo 2 puntos del lado de origen antes de cruzar (aparece casi sobre la línea)
    ys = [112, 108] + list(np.linspace(90, 20, 18))
    assert detectar_sentido_contrario(_track(5, ys), GEO, CFG, "c").empty


def test_cruce_limpio_directo():
    d = np.array([10, 10, 10, -10, -10, -10], dtype=float)
    assert cruce_limpio(d, 2, 3, 3, 4.0)
    assert not cruce_limpio(d, 2, 3, 4, 4.0)  # no hay 4 de cada lado
    d2 = np.array([10, 2, 10, -10, -10, -10], dtype=float)  # uno dentro de la banda
    assert not cruce_limpio(d2, 2, 3, 3, 4.0)


def test_linea_doble_mano_no_alerta():
    geo = GEO.model_copy(
        update={"lineas": [LINEA.model_copy(update={"sentido_permitido": "ambos"})]}
    )
    assert detectar_sentido_contrario(_track(6, np.linspace(160, 40, 20)), geo, CFG, "c").empty


def test_guardar_evidencia(tmp_path: Path):
    ruta = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), 30, (400, 400))
    for _ in range(40):
        w.write(np.full((400, 400, 3), 90, np.uint8))
    w.release()
    det = _track(7, np.linspace(160, 40, 20))
    ev = detectar_sentido_contrario(det, GEO, CFG, "c")
    ev = guardar_evidencia(ruta, ev, det, GEO, tmp_path / "eventos")
    assert ev.loc[0, "thumb"] == "eventos/7_av.jpg"
    img = cv2.imread(str(tmp_path / ev.loc[0, "thumb"]))
    assert img is not None and img.size > 0
    # sin eventos: no rompe y limpia la carpeta
    vacio = guardar_evidencia(ruta, ev.iloc[0:0], det, GEO, tmp_path / "eventos")
    assert vacio.empty and not list((tmp_path / "eventos").glob("*.jpg"))


def test_peatones_no_disparan_contramano():
    det = _track(8, np.linspace(160, 40, 20)).assign(clase="peaton")
    assert detectar_sentido_contrario(det, GEO, CFG, "c").empty
