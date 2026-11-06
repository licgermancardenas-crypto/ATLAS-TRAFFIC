"""Matriz de giros: origen, destino, recorridos reconstruidos y reparto."""

import numpy as np
import pandas as pd
import pytest

from atlas_traffic.config import Geometria, Linea
from atlas_traffic.counting import contar
from atlas_traffic.turning import movimientos, reparto_giros

# Una avenida que sube (y decrece) y una calle que sale hacia la izquierda.
# Líneas A->B hacia la derecha: lado positivo = abajo; subir = cruce negativo.
GEO = Geometria(
    encuadre="t",
    resolucion=(400, 400),
    roi=[(0, 0), (400, 0), (400, 400), (0, 400)],
    lineas=[
        Linea(id="av_frente", puntos=((150, 300), (250, 300)), sentido_positivo="saliente",
              sentido_permitido="entrante", clases=["vehiculo"]),
        Linea(id="av_fondo", puntos=((150, 100), (250, 100)), sentido_positivo="entrante",
              sentido_permitido="saliente", clases=["vehiculo"]),
        # vertical hacia abajo: lado positivo = izquierda; ir a la izquierda = positivo
        Linea(id="calle_izq", puntos=((100, 150), (100, 250)), sentido_positivo="saliente",
              sentido_permitido="saliente", clases=["vehiculo"]),
    ],
)  # fmt: skip


def _track(tid, puntos, t0=0.0, fps=30, paso=0.1):
    """Recorre los puntos en línea recta a tramos, una detección cada `paso` segundos."""
    filas, t = [], t0
    for (x0, y0), (x1, y1) in zip(puntos, puntos[1:], strict=False):
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 10))
        for k in range(n):
            x, y = x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n
            filas.append(dict(frame=round(t * fps), ts=t, track_id=tid, clase="vehiculo",
                              clase_detalle="car", cx_ref=x, cy_ref=y, x1=x - 10, x2=x + 10,
                              y1=y - 8, y2=y + 8))  # fmt: skip
            t += paso
    return filas


def _mov(*tracks):
    det = pd.DataFrame([f for t in tracks for f in t])
    return movimientos(contar(det, GEO), det, GEO).set_index("track_id")


def test_recorrido_completo_y_giro():
    m = _mov(_track(1, [(200, 350), (200, 50)]),
             _track(2, [(200, 350), (200, 200), (50, 200)]))  # fmt: skip
    assert m.loc[1, ["origen", "destino", "estado"]].tolist() == [
        "av_frente", "av_fondo", "completo"]  # fmt: skip
    assert m.loc[2, ["origen", "destino"]].tolist() == ["av_frente", "calle_izq"]


def test_reconstruye_un_vehiculo_tapado_en_el_medio():
    # entra, se pierde detrás de algo y reaparece 1 s después, hacia donde iba
    m = _mov(_track(1, [(200, 350), (200, 230)]),
             _track(2, [(160, 200), (50, 200)], t0=2.2))  # fmt: skip
    assert len(m) == 1
    assert m.loc[1, ["origen", "destino", "estado"]].tolist() == [
        "av_frente", "calle_izq", "reconstruido"]  # fmt: skip
    assert m.loc[1, "track_id_salida"] == 2


def test_no_reconstruye_si_reaparece_lejos_o_tarde():
    lejos = _mov(_track(1, [(200, 350), (200, 230)]),
                 _track(2, [(390, 120), (390, 50)], t0=2.2))  # fmt: skip
    assert set(lejos["estado"]) == {"solo_entrada"}
    tarde = _mov(_track(1, [(200, 350), (200, 230)]),
                 _track(2, [(160, 200), (50, 200)], t0=10))  # fmt: skip
    assert sorted(tarde["estado"]) == ["solo_entrada", "solo_salida"]


def test_origen_inferido_solo_cerca_de_la_entrada():
    # aparece a 20 px pasada la entrada (estaba en la cola al arrancar el clip)
    cerca = _mov(_track(1, [(200, 280), (200, 50)]))
    assert cerca.loc[1, ["origen", "estado"]].tolist() == ["av_frente", "origen_inferido"]
    # aparece en el medio del cruce: no se sabe de dónde vino
    medio = _mov(_track(1, [(200, 180), (200, 50)]))
    assert medio.loc[1, ["origen", "estado"]].tolist() == [None, "solo_salida"]


def test_reparto_y_volumen_estimado():
    tracks = [_track(i, [(200, 350), (200, 50)], t0=i) for i in range(1, 4)]
    tracks.append(_track(9, [(200, 350), (200, 200), (50, 200)], t0=5))
    tracks.append(_track(20, [(200, 350), (200, 260)], t0=8))  # entró y no se vio salir
    r = reparto_giros(_mov(*tracks).reset_index()).set_index("destino")
    assert r.loc["av_fondo", "observados"] == 3
    assert r.loc["av_fondo", "pct"] == pytest.approx(0.75)
    assert r.loc["av_fondo", "pct_inf"] < 0.75 < r.loc["av_fondo", "pct_sup"]
    # 5 entraron: 4 con recorrido completo + 1 que no se vio salir
    assert r.loc["av_fondo", "entradas_origen"] == 5
    assert r.loc["calle_izq", "volumen_estimado"] == pytest.approx(1.25)


def test_sin_vehiculos():
    vacio = pd.DataFrame(columns=["frame", "ts", "track_id", "clase", "cx_ref", "cy_ref"])
    assert movimientos(pd.DataFrame(), vacio, GEO).empty
    assert reparto_giros(movimientos(pd.DataFrame(), vacio, GEO)).empty
