"""Ficha de un vehículo para "seguir un vehículo" en el dashboard."""

import numpy as np
import pandas as pd
import pytest

from atlas_traffic.seguimiento import ficha_vehiculo, recorrido_suavizado


def _tray():
    # vehículo 7: 2 s quieto y después 3 s a 10 m/s (36 km/h) por Y; una muestra cada 1/30 s
    t = np.arange(0, 5, 1 / 30)
    y = np.where(t < 2, 0.0, (t - 2) * 10)
    v = np.where(t < 2, 0.0, 36.0)
    ruido = np.random.default_rng(0).normal(0, 0.05, len(t))  # temblor de la caja
    return pd.DataFrame({"track_id": 7, "ts": t, "x_m": 3.0 + ruido, "y_m": y, "vel_kmh": v})


def test_ficha_con_tiempos_distancia_y_movimiento():
    tracks = pd.DataFrame({"track_id": [7], "clase_detalle": ["car"], "color": ["rojo"],
                           "thumb": ["runs/c/thumbs/7.jpg"], "velocidad_kmh": [36.0],
                           "velocidad_max_kmh": [36.0]})  # fmt: skip
    cruces = pd.DataFrame({
        "track_id": [7, 7], "aproximacion": ["av_frente", "av_fondo"],
        "sentido": ["entrante", "saliente"], "ts_cruce": [3.0, 4.5],
    })  # fmt: skip
    giros = pd.DataFrame({"track_id": [7], "origen": ["av_cordoba_frente"],
                          "destino": ["rodriguez_pena_izq"], "track_id_salida": [7]})  # fmt: skip
    f = ficha_vehiculo(7, _tray(), tracks, cruces, giros)
    assert f.tipo == "car" and f.color == "rojo"
    assert f.en_escena_s == pytest.approx(5.0, abs=0.05)
    assert f.detenido_s == pytest.approx(2.0, abs=0.05)
    # 30 m recorridos: el temblor de la caja no suma metros
    assert f.distancia_m == pytest.approx(30.0, abs=1.0)
    assert f.movimiento == "Córdoba → dobla a Rodríguez Peña"
    assert list(f.cruces["aproximacion"]) == ["av_frente", "av_fondo"]


def test_recorrido_reconstruido_se_encuentra_por_su_pedazo_de_salida():
    giros = pd.DataFrame({"track_id": [3], "origen": ["av_cordoba_frente"],
                          "destino": ["av_cordoba_fondo"], "track_id_salida": [7]})  # fmt: skip
    vacio = pd.DataFrame(columns=["track_id", "ts_cruce"])
    f = ficha_vehiculo(7, _tray(), pd.DataFrame(columns=["track_id"]), vacio, giros)
    assert f.movimiento == "Córdoba → sigue por Córdoba"


def test_vehiculo_que_no_se_vio_salir_y_sin_calibracion():
    giros = pd.DataFrame({"track_id": [7], "origen": ["rodriguez_pena_der"], "destino": [None]})
    sin_metros = _tray().drop(columns=["x_m", "y_m", "vel_kmh"])
    vacio = pd.DataFrame(columns=["track_id", "ts_cruce"])
    f = ficha_vehiculo(7, sin_metros, pd.DataFrame(columns=["track_id"]), vacio, giros)
    assert f.movimiento == "entra por Rodríguez Peña (no se lo vio salir)"
    assert np.isnan(f.distancia_m) and np.isnan(f.detenido_s)
    assert recorrido_suavizado(sin_metros).empty
