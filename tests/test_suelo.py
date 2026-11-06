"""Calibración métrica del suelo y velocidad en km/h."""

import cv2
import numpy as np
import pandas as pd
import pytest

from atlas_traffic.config import Geometria, Linea
from atlas_traffic.suelo import (
    CalibracionSuelo,
    Senda,
    a_metros,
    ajustar_suelo,
    cargar_calibracion_suelo,
    centros_franjas,
    guardar_calibracion_suelo,
    transferir_suelo,
)
from atlas_traffic.velocidad import en_suelo, velocidad_kmh, velocidad_por_track

# una cámara inclinada: suelo (m) -> imagen (px), con perspectiva
G = np.array([[30.0, -6.0, 300.0], [2.0, -8.0, 900.0], [0.0, 0.012, 1.0]])

# sendas de una esquina: dos que cruzan la avenida (franjas en X), dos la calle (en Y)
MODELO = {
    "frente": ("x", 0.0, 0.0, 16),
    "fondo": ("x", 17.0, 0.5, 15),
    "izq": ("y", -6.0, 3.0, 8),
    "der": ("y", 20.0, 3.5, 8),
}


def _imagen(G_, xy):
    p = np.c_[xy, np.ones(len(xy))] @ G_.T
    return p[:, :2] / p[:, 2:3]


def _sendas(G_, desde=0, ruido=0.0, ids=MODELO):
    rng = np.random.default_rng(0)
    out = []
    for sid in ids:
        d, q, s0, n = MODELO[sid]
        a = s0 + np.arange(desde, n)
        suelo = np.c_[a, np.full_like(a, q)] if d == "x" else np.c_[np.full_like(a, q), a]
        s = Senda(sid, d, (0, 0), (0, 0))
        s.centros = _imagen(G_, suelo) + rng.normal(0, ruido, (len(a), 2))
        s.indices = np.arange(len(a))
        out.append(s)
    return out


def test_ajuste_recupera_la_homografia():
    guia = {"fondo": (15.0, 0.0), "izq": (-4.0, 2.0), "der": (18.0, 2.0)}
    H, residuo, pos = ajustar_suelo(_sendas(G, ruido=0.3), 1.0, guia)
    assert residuo < 0.6
    puntos = np.array([[0.0, 0.0], [10.0, 5.0], [16.0, 17.0], [-5.0, 10.0]])
    assert np.allclose(a_metros(H, _imagen(G, puntos)), puntos, atol=0.1)
    assert pos["fondo"]["q"] == pytest.approx(17.0, abs=0.1)


def test_transferencia_a_otro_encuadre_con_franjas_tapadas():
    guia = {"fondo": (15.0, 0.0), "izq": (-4.0, 2.0), "der": (18.0, 2.0)}
    _, _, pos = ajustar_suelo(_sendas(G), 1.0, guia)
    # otro encuadre: otra cámara, se ve desde la tercera franja y solo tres sendas
    G2 = np.array([[12.0, -3.0, 150.0], [1.0, -4.0, 450.0], [0.0, 0.01, 1.0]])
    sendas = _sendas(G2, desde=2, ids=["frente", "fondo", "izq"])
    H2, err, desfases = transferir_suelo(sendas, pos, 1.0)
    assert err < 0.1 and desfases == {"frente": 2, "fondo": 2, "izq": 2}
    puntos = np.array([[3.0, 2.0], [12.0, 14.0]])
    assert np.allclose(a_metros(H2, _imagen(G2, puntos)), puntos, atol=0.05)


def test_centros_de_franjas_en_una_imagen():
    img = np.full((200, 600), 60, np.uint8)
    for k in range(10):  # franjas de 20 px separadas 20 px, desde x = 100
        cv2.rectangle(img, (100 + 40 * k, 50), (119 + 40 * k, 150), 230, -1)
    s = centros_franjas(img, Senda("s", "x", (60, 100), (560, 100)))
    assert len(s.centros) == 10
    assert np.allclose(s.centros[:, 0], 110 + 40 * np.arange(10), atol=1.5)
    assert s.indices.tolist() == list(range(10))


def test_guardar_y_cargar(tmp_path):
    cal = CalibracionSuelo(encuadre="e", homografia=np.eye(3).tolist(), periodo_m=1.0,
                           residuo_px=1.2, n_franjas=30)  # fmt: skip
    guardar_calibracion_suelo(cal, tmp_path / "e.yaml")
    assert cargar_calibracion_suelo(tmp_path / "e.yaml") == cal
    assert cargar_calibracion_suelo(tmp_path / "no_existe.yaml") is None


def test_velocidad_de_un_vehiculo_a_36_kmh():
    # 10 m/s a lo largo de Y, una detección cada 2 cuadros a 60 fps, cámara quieta
    fps, H = 60.0, np.linalg.inv(G)
    t = np.arange(0, 3, 2 / fps)
    apoyo = _imagen(G, np.c_[np.full_like(t, 5.0), 2 + 10 * t])
    det = pd.DataFrame({"frame": np.round(t * fps).astype(int), "ts": t, "track_id": 1,
                        "clase": "vehiculo", "x1": apoyo[:, 0] - 20, "x2": apoyo[:, 0] + 20,
                        "y1": apoyo[:, 1] - 30, "y2": apoyo[:, 1]})  # fmt: skip
    det = en_suelo(det, {}, H)
    assert np.allclose(det["y_m"], 2 + 10 * t, atol=1e-6)
    det = det.assign(vel_kmh=velocidad_kmh(det, fps, 0.5), cx_ref=det["x_m"],
                     cy_ref=det["y_m"])  # fmt: skip
    assert det["vel_kmh"].dropna().to_numpy() == pytest.approx(36.0, abs=0.01)
    r = velocidad_por_track(det).iloc[0]
    assert r["velocidad_kmh"] == pytest.approx(36.0, abs=0.01)


def test_velocidad_por_intervalo_y_auto_quieto():
    from atlas_traffic.velocidad import velocidad_por_intervalo

    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    linea = Linea(id="l", puntos=((0, 50), (10, 50)), sentido_positivo="entrante")
    geo = Geometria(encuadre="t", resolucion=(100, 100), roi=roi, lineas=[linea])
    filas = []
    for tid, v in [(1, 0.0), (2, 20.0), (3, 30.0), (4, 40.0)]:
        for k in range(60):
            filas.append(dict(track_id=tid, frame=2 * k, ts=k / 30, cx_ref=50.0, cy_ref=50.0,
                              vel_kmh=v))  # fmt: skip
    r = velocidad_por_intervalo(pd.DataFrame(filas), geo, np.array([0.0, 2.0]))
    assert r["velocidad_kmh_media"].iat[0] == pytest.approx(22.5)
    # el detenido no entra en el percentil 85 de los que se mueven
    assert r["velocidad_kmh_p85"].iat[0] == pytest.approx(np.percentile([20, 30, 40], 85))
