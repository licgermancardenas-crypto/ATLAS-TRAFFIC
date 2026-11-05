"""Pipeline de Fase 1 ejercitado con FakeDetector sobre un video sintético."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from atlas_traffic.config import AppConfig, Clip, EstabilizacionCfg, RutasCfg, VideoCfg
from atlas_traffic.detect import Deteccion, FakeDetector
from atlas_traffic.pipeline import (
    analizar_clip,
    correr_clip,
    detectar_video,
    tracks_desde_detecciones,
)


@pytest.fixture
def video(tmp_path) -> Path:
    ruta = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(10):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    return ruta


def _det(tid, x, clase="vehiculo", detalle="car"):
    return Deteccion(bbox=(x, 10, x + 10, 20), clase=clase, confianza=0.8, track_id=tid,
                     clase_detalle=detalle)  # fmt: skip


def _guion():
    # track 1 avanza 2 px por frame durante 10 frames; track 2 (peatón) solo en frames 0-2;
    # una detección sin ID que debe descartarse.
    frames = [[_det(1, 2 * i)] for i in range(10)]
    for i in range(3):
        frames[i].append(_det(2, 40, "peaton", "person"))
    frames[0].append(Deteccion(bbox=(0, 0, 5, 5), clase="vehiculo", confianza=0.5, track_id=None))
    return frames


def test_detectar_video_registra_frames_ts_y_descarta_sin_id(video):
    d = detectar_video(video, FakeDetector(_guion())).detecciones
    assert d["track_id"].notna().all()
    assert len(d) == 10 + 3
    assert d.loc[d.frame == 5, "ts"].iat[0] == pytest.approx(0.5)


def test_paso_frames_mantiene_indice_real(video):
    d = detectar_video(video, FakeDetector(_guion()), paso_frames=2).detecciones
    # el detector ve 5 frames (0,2,4,6,8) y consume las 5 primeras entradas del guion
    assert sorted(d.loc[d.track_id == 1, "frame"]) == [0, 2, 4, 6, 8]


def test_max_segundos(video):
    d = detectar_video(video, FakeDetector(_guion()), max_segundos=0.5).detecciones
    assert d["frame"].max() == 4


def test_tracks_un_registro_por_objeto_con_velocidad(video):
    t = tracks_desde_detecciones(
        detectar_video(video, FakeDetector(_guion())).detecciones
    ).set_index("track_id")
    assert list(t.index) == [1, 2]
    assert t.loc[1, "clase"] == "vehiculo" and t.loc[2, "clase_detalle"] == "person"
    assert (t.loc[1, "frame_inicio"], t.loc[1, "frame_fin"]) == (0, 9)
    assert t.loc[1, "velocidad_px"] == pytest.approx(2.0)
    assert t.loc[2, "velocidad_px"] == pytest.approx(0.0)


def test_velocidad_en_px_por_frame_real_aunque_se_salteen_frames():
    d = pd.DataFrame({"track_id": [1, 1, 1], "frame": [0, 2, 4], "cx": [0.0, 4.0, 8.0],
                      "cy": [0.0] * 3, "clase": ["vehiculo"] * 3,
                      "clase_detalle": ["car"] * 3, "confianza": [0.9] * 3})  # fmt: skip
    assert tracks_desde_detecciones(d)["velocidad_px"].iat[0] == pytest.approx(2.0)


def test_tracks_vacio():
    assert tracks_desde_detecciones(pd.DataFrame()).empty


def _clip(video, geometria):
    return Clip(id="c", archivo=video, fecha="2026-11-04", hora_inicio="12:00",
                dia_semana="miercoles", tipo_dia="habil", clima="x",
                geometria=geometria)  # fmt: skip


def test_correr_clip_escribe_parquets(video, tmp_path):
    clip = _clip(video, tmp_path / "no_existe.yaml")
    cfg = AppConfig(
        rutas=RutasCfg(outputs=tmp_path / "out"),
        video=VideoCfg(paso_frames=1),
        estabilizacion=EstabilizacionCfg(activa=False),  # video negro: sin textura
    )
    salida = correr_clip(clip, cfg, FakeDetector(_guion()))
    assert len(pd.read_parquet(salida / "detecciones.parquet")) == 13
    assert not (salida / "estabilizacion.parquet").exists()
    res = analizar_clip(clip, cfg)  # sin geometría: tracks sí, conteo no
    assert res.conteo is None
    assert len(pd.read_parquet(salida / "tracks.parquet")) == 2
    assert not (salida / "cruces.parquet").exists()


def test_analizar_clip_con_geometria_cuenta(video, tmp_path):
    from atlas_traffic.config import Geometria, Linea
    from atlas_traffic.geometry import guardar_geometria

    # el track 1 avanza en x de 5 a 23 (centroide); línea vertical en x=12
    linea = Linea(id="l", puntos=((12, 0), (12, 40)), sentido_positivo="entrante")
    roi = [(0, 0), (64, 0), (64, 48), (0, 48)]
    geo = Geometria(encuadre="g", resolucion=(64, 48), roi=roi, lineas=[linea])
    ruta_geo = tmp_path / "g.yaml"
    guardar_geometria(geo, ruta_geo)
    clip = _clip(video, ruta_geo)
    cfg = AppConfig(rutas=RutasCfg(outputs=tmp_path / "out"), video=VideoCfg(paso_frames=1),
                    estabilizacion=EstabilizacionCfg(activa=False))  # fmt: skip
    correr_clip(clip, cfg, FakeDetector(_guion()))
    res = analizar_clip(clip, cfg)
    # A->B hacia abajo: lado positivo = izquierda; ir hacia la derecha es cruce negativo
    assert res.conteo.loc[("l", "saliente"), "vehiculo"] == 1
    assert res.conteo.to_numpy().sum() == 1
    tracks = pd.read_parquet(res.salida / "tracks.parquet").set_index("track_id")
    assert bool(tracks.loc[1, "cruzo_linea"]) and not tracks.loc[2, "cruzo_linea"]
    assert (res.salida / "conteo.csv").exists()


def test_analizar_sin_detecciones_falla(video, tmp_path):
    cfg = AppConfig(rutas=RutasCfg(outputs=tmp_path / "out"))
    with pytest.raises(FileNotFoundError, match="no hay detecciones"):
        analizar_clip(_clip(video, tmp_path / "g.yaml"), cfg)


def test_velocidad_ventana_uniforme_ruidosa_y_corta():
    from atlas_traffic.pipeline import velocidad_ventana

    f = np.arange(0, 100, 2)
    uniforme = np.column_stack([1.5 * f, np.zeros_like(f)]).astype(float)
    assert velocidad_ventana(f, uniforme, 20) == pytest.approx(1.5)
    # objeto quieto con temblor de 1 px: sumar pasos daría ~1 px/frame; la ventana, mucho menos
    rng = np.random.default_rng(0)
    quieto = rng.normal(0, 1.0, (len(f), 2))
    pasos = np.hypot(*np.diff(quieto, axis=0).T).sum() / (f[-1] - f[0])
    assert velocidad_ventana(f, quieto, 20) < pasos / 4
    assert velocidad_ventana(np.array([0, 4]), np.array([[0.0, 0], [8, 0]]), 20) == 2.0
    assert velocidad_ventana(np.array([3]), np.array([[1.0, 1]]), 20) == 0.0
