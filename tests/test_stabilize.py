"""Estabilización: recupera movimientos de cámara conocidos sobre escenas sintéticas."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from atlas_traffic.config import EstabilizacionCfg
from atlas_traffic.detect import Deteccion, FakeDetector
from atlas_traffic.pipeline import detectar_video, tracks_desde_detecciones
from atlas_traffic.stabilize import (
    Estabilizador,
    homografia_plausible,
    mascara_sin_cajas,
    transformar_puntos,
)

ALTO, ANCHO = 360, 480


def _escena(seed: int = 0) -> np.ndarray:
    """Imagen con textura rica (bloques y círculos aleatorios): buena para ORB."""
    rng = np.random.default_rng(seed)
    img = np.full((ALTO + 200, ANCHO + 200, 3), 128, np.uint8)
    for _ in range(400):
        x, y = rng.integers(0, img.shape[1]), rng.integers(0, img.shape[0])
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(
                img, (x, y), (x + rng.integers(5, 30), y + rng.integers(5, 30)), color, -1
            )
        else:
            cv2.circle(img, (x, y), int(rng.integers(3, 15)), color, -1)
    return img


def _vista(escena: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Recorte de la escena desplazado (dx, dy): simula una cámara que se corre."""
    return escena[100 + dy : 100 + dy + ALTO, 100 + dx : 100 + dx + ANCHO].copy()


def test_recupera_traslacion():
    escena = _escena()
    est = Estabilizador(_vista(escena, 0, 0))
    r = est.estimar(_vista(escena, 12, -7))
    assert r.ok
    # un punto en (100, 100) del frame desplazado está en (112, 93) de la referencia
    np.testing.assert_allclose(
        transformar_puntos(r.H, np.array([[100, 100]])), [[112, 93]], atol=1.0
    )


def test_recupera_rotacion_y_zoom():
    ref = _vista(_escena(), 0, 0)
    M = cv2.getRotationMatrix2D((ANCHO / 2, ALTO / 2), 4, 1.08)  # referencia -> frame
    frame = cv2.warpAffine(ref, M, (ANCHO, ALTO), borderMode=cv2.BORDER_REFLECT)
    r = Estabilizador(ref).estimar(frame)
    assert r.ok
    pts_ref = np.array([[150.0, 120.0], [300.0, 250.0]])
    pts_frame = cv2.transform(pts_ref.reshape(-1, 1, 2), M).reshape(-1, 2)
    np.testing.assert_allclose(transformar_puntos(r.H, pts_frame), pts_ref, atol=1.5)


def test_frame_sin_textura_reutiliza_ultima_h():
    escena = _escena()
    est = Estabilizador(_vista(escena, 0, 0))
    buena = est.estimar(_vista(escena, 5, 0))
    mala = est.estimar(np.zeros((ALTO, ANCHO, 3), np.uint8))
    assert buena.ok and not mala.ok
    np.testing.assert_array_equal(mala.H, buena.H)


def test_referencia_sin_textura_falla():
    with pytest.raises(ValueError, match="textura"):
        Estabilizador(np.zeros((ALTO, ANCHO, 3), np.uint8))


def test_mascara_tapa_cajas():
    m = mascara_sin_cajas((100, 100), [(10, 10, 20, 20)], margen=0)
    assert m[15, 15] == 0 and m[50, 50] == 255


def test_homografia_plausible():
    assert homografia_plausible(np.eye(3))
    assert not homografia_plausible(np.diag([-1.0, 1.0, 1.0]))  # espejo
    assert not homografia_plausible(np.diag([5.0, 5.0, 1.0]))  # escala absurda
    assert not homografia_plausible(None)


def test_pipeline_compensa_camara_en_movimiento(tmp_path: Path):
    """Un objeto quieto en el mundo, filmado con cámara que se corre 2 px por frame.

    En coordenadas del frame parece moverse; en coordenadas de referencia queda quieto,
    y su velocidad es ~0.
    """
    escena, ruta, n = _escena(1), tmp_path / "mov.mp4", 12
    w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), 10, (ANCHO, ALTO))
    guion = []
    for i in range(n):
        w.write(_vista(escena, 2 * i, 0))
        x = 200 - 2 * i  # el objeto está fijo en la escena: en el frame se corre a la izquierda
        guion.append([Deteccion((x, 150, x + 20, 170), "vehiculo", 0.9, 1, "car")])
    w.release()

    res = detectar_video(ruta, FakeDetector(guion), estabilizacion=EstabilizacionCfg())
    d = res.detecciones
    assert res.estabilizacion["ok"].mean() > 0.9
    assert d["cx"].max() - d["cx"].min() == pytest.approx(2 * (n - 1))  # en el frame se mueve
    assert d["cx_ref"].max() - d["cx_ref"].min() < 3.0  # en referencia, quieto (± compresión)
    assert tracks_desde_detecciones(d)["velocidad_px"].iat[0] < 0.5


def test_suavizado_preserva_tendencia_lineal_y_quita_ruido():
    from atlas_traffic.stabilize import suavizar_homografias

    n = 40
    rng = np.random.default_rng(0)
    H = np.tile(np.eye(3), (n, 1, 1))
    H[:, 0, 2] = 2.0 * np.arange(n)  # cámara que se corre 2 px por frame
    suave = suavizar_homografias(H, 9)
    np.testing.assert_allclose(suave[:, 0, 2], H[:, 0, 2], atol=1e-9)  # bordes incluidos
    ruidosa = H.copy()
    ruidosa[:, 0, 2] += rng.normal(0, 1.0, n)
    error_crudo = np.abs(ruidosa[:, 0, 2] - H[:, 0, 2]).mean()
    error_suave = np.abs(suavizar_homografias(ruidosa, 9)[4:-4, 0, 2] - H[4:-4, 0, 2]).mean()
    assert error_suave < error_crudo / 2
    np.testing.assert_array_equal(suavizar_homografias(H, 1), H)
