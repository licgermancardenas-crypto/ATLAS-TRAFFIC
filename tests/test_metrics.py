"""Fase 3: velocidad instantánea, flujo, densidad y ventanas parciales."""

import numpy as np
import pandas as pd
import pytest

from atlas_traffic.config import Geometria, Linea
from atlas_traffic.metrics import COLUMNAS_INTERVALOS, calcular_intervalos, velocidad_instantanea

FPS = 10.0
GEO = Geometria(
    encuadre="t",
    resolucion=(1000, 1000),
    roi=[(0, 0), (500, 0), (500, 500), (0, 500)],
    lineas=[Linea(id="l", puntos=((0, 250), (500, 250)), sentido_positivo="entrante")],
)


def _track(tid, frames, xs, ys, clase="vehiculo"):
    return pd.DataFrame({
        "track_id": tid, "frame": frames, "ts": np.asarray(frames) / FPS, "clase": clase,
        "cx_ref": np.asarray(xs, float), "cy_ref": np.asarray(ys, float),
    })  # fmt: skip


def test_velocidad_instantanea_uniforme_quieta_y_corta():
    f = np.arange(0, 60, 2)
    det = pd.concat([
        _track(1, f, 100 + 3 * f, np.full(len(f), 100)),  # 3 px/frame
        _track(2, f, np.full(len(f), 50), 50 + np.random.default_rng(0).normal(0, 1, len(f))),
        _track(3, [0, 2], [0, 6], [0, 0]),  # 2 frames: más corto que media ventana de 20
    ])  # fmt: skip
    v = velocidad_instantanea(det, 20)
    assert v[det.track_id == 1].dropna().to_numpy() == pytest.approx(3.0)
    assert v[det.track_id == 2].dropna().mean() < 0.3  # temblor de 1 px: ~quieto
    assert v[det.track_id == 3].isna().all()


def _cruce(tid, ts, clase="vehiculo"):
    return {"track_id": tid, "clase": clase, "ts_cruce": ts, "aproximacion": "l",
            "sentido": "entrante"}  # fmt: skip


def test_intervalos_flujo_densidad_y_ventana_parcial():
    # 45 s de video a 10 fps procesando todos los frames -> ventanas [0,30) y [30,45)
    frames = np.arange(0, 450)
    det = pd.concat([
        # 2 vehículos quietos en la ROI durante los primeros 30 s, 1 fuera de la ROI
        _track(1, frames[:300], np.full(300, 100), np.full(300, 100)),
        _track(2, frames[:300], np.full(300, 200), np.full(300, 100)),
        _track(3, frames, np.full(450, 900), np.full(450, 900)),
        # 1 vehículo a 2 px/frame en la ROI entre 30 y 45 s
        _track(4, frames[300:], 10 + 2 * (frames[300:] - 300) / 10, np.full(150, 300)),
    ])  # fmt: skip
    cruces = pd.DataFrame([
        _cruce(10, 5.0), _cruce(10, 6.0),  # mismo track, dos cruces: cuenta una vez
        _cruce(11, 29.9), _cruce(12, 31.0),
        _cruce(20, 10.0, "peaton"),
    ])  # fmt: skip
    iv = calcular_intervalos(det, cruces, GEO, "c", duracion_s=45.0, frames_procesados=frames,
                             fps=FPS, intervalo_s=30, ventana_frames=20)  # fmt: skip
    assert list(iv.columns) == COLUMNAS_INTERVALOS
    assert list(iv["duracion_s"]) == [30.0, 15.0]
    # ventana 1: 2 vehículos únicos en 30 s = 4/min; 1 peatón = 2/min
    assert iv.loc[0, "flujo_veh_min"] == pytest.approx(4.0)
    assert iv.loc[0, "flujo_peat_min"] == pytest.approx(2.0)
    # ventana 2: 1 vehículo en 15 s = 4/min (normalizado por la duración real)
    assert iv.loc[1, "flujo_veh_min"] == pytest.approx(4.0)
    # densidad: el vehículo fuera de la ROI no cuenta
    assert iv.loc[0, "densidad_media"] == pytest.approx(2.0)
    assert iv.loc[1, "densidad_media"] == pytest.approx(1.0)
    assert iv.loc[0, "velocidad_media_px"] == pytest.approx(0.0)
    assert iv.loc[1, "velocidad_media_px"] == pytest.approx(0.2)  # 2 px cada 10 frames
    assert iv["congestion"].isna().all()  # la completa la Fase 4


def test_densidad_cuenta_frames_vacios():
    # un vehículo solo en la mitad de los frames procesados -> densidad 0.5
    frames = np.arange(0, 100, 2)
    det = _track(1, frames[:25], np.full(25, 100), np.full(25, 100))
    iv = calcular_intervalos(det, pd.DataFrame(), GEO, "c", duracion_s=10.0,
                             frames_procesados=frames, fps=FPS)  # fmt: skip
    assert len(iv) == 1
    assert iv.loc[0, "densidad_media"] == pytest.approx(0.5)
    assert iv.loc[0, "flujo_veh_min"] == 0


def test_sin_detecciones():
    iv = calcular_intervalos(
        pd.DataFrame(columns=["track_id", "frame", "ts", "clase", "cx_ref", "cy_ref"]),
        pd.DataFrame(), GEO, "c", duracion_s=20.0, frames_procesados=np.arange(0, 200, 2),
        fps=FPS,
    )  # fmt: skip
    assert iv.loc[0, "densidad_media"] == 0
    assert np.isnan(iv.loc[0, "velocidad_media_px"])
