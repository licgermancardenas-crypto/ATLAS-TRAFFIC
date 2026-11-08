"""Situaciones de riesgo vehículo-peatón (PET)."""

import numpy as np
import pandas as pd

from atlas_traffic.config import ConflictosCfg
from atlas_traffic.conflictos import detectar_conflictos, gravedad

CFG = ConflictosCfg()
T = np.arange(0, 10, 1 / 30)


def _auto(tid, x, t0, v_ms=10.0):
    """Vehículo que baja por la calle x=const, pasando por y=0 en t0."""
    return pd.DataFrame({"track_id": tid, "ts": T, "x_m": x, "y_m": (T - t0) * v_ms,
                         "vel_kmh": v_ms * 3.6})  # fmt: skip


def _peaton(tid, y, x0=-3.0, t0=0.0, v_ms=1.4):
    """Peatón que cruza la calle (a lo ancho, X) a la altura y."""
    return pd.DataFrame({"track_id": tid, "ts": T, "x_m": x0 + (T - t0) * v_ms, "y_m": y})


def _transito(x):
    # otros autos por el mismo carril: eso hace que el punto sea calzada
    return pd.concat([_auto(100 + k, x, 8 + k * 0.3) for k in range(3)])


def test_auto_que_pasa_un_segundo_despues_del_peaton_es_moderado():
    # el auto pasa por (0, 0) en t=5 s. El último punto del peatón en la calzada (a menos
    # de 1.5 m del carril) es x=1.5 en t≈3.2 s; el auto llega a 2 m de ahí en t≈4.87 s
    # (~1.7 s después). Más allá del carril el peatón ya está en la vereda.
    veh = pd.concat([_auto(1, 0.0, 5.0), _transito(0.0)])
    c = detectar_conflictos(veh, _peaton(50, 0.0), CFG)
    c = c[c["track_id"] == 1].iloc[0]
    assert c["track_id_peaton"] == 50 and c["primero"] == "peaton"
    assert 1.5 < c["pet_s"] < 1.9 and c["gravedad"] == "moderado"
    assert abs(c["vel_kmh"] - 36) < 1


def test_auto_que_pasa_con_el_peaton_al_lado_es_grave():
    # cuando pasa el auto, el peatón todavía está a 1.6 m: PET 0
    veh = pd.concat([_auto(1, 0.0, 3.3), _transito(0.0)])
    c = detectar_conflictos(veh, _peaton(50, 0.0), CFG)
    c = c[c["track_id"] == 1].iloc[0]
    assert c["pet_s"] < 0.1 and c["gravedad"] == "grave" and c["distancia_m"] < 2


def test_lejos_en_el_tiempo_no_es_conflicto():
    veh = pd.concat([_auto(1, 0.0, 8.0), _transito(0.0)])
    c = detectar_conflictos(veh, _peaton(50, 0.0), CFG)
    assert 1 not in set(c["track_id"])


def test_peaton_esperando_en_el_cordon_no_es_conflicto():
    # parado a 1.8 m del carril, donde nunca pasa un auto
    quieto = pd.DataFrame({"track_id": 50, "ts": T, "x_m": -1.8, "y_m": 0.0})
    veh = pd.concat([_auto(1, 0.0, 3.0), _transito(0.0)])
    assert detectar_conflictos(veh, quieto, CFG).empty


def test_objeto_fijo_tomado_por_vehiculo_no_cuenta():
    # caja que tiembla en el lugar (un kiosco): "velocidad" alta por ruido, sin desplazarse
    rng = np.random.default_rng(0)
    fijo = pd.DataFrame({"track_id": 9, "ts": T, "x_m": rng.normal(0, 0.3, len(T)),
                         "y_m": rng.normal(0, 0.3, len(T)), "vel_kmh": 12.0})  # fmt: skip
    c = detectar_conflictos(pd.concat([fijo, _transito(0.0)]), _peaton(50, 0.0), CFG)
    assert 9 not in set(c["track_id"])


def test_peaton_detectado_un_instante_no_cuenta():
    # 5 detecciones en 0.15 s justo donde pasa el auto: un poste o un reflejo
    flash = _peaton(50, 0.0).iloc[60:65]
    veh = pd.concat([_auto(1, 0.0, 2.1), _transito(0.0)])
    assert detectar_conflictos(veh, flash, CFG).empty


def test_gravedad():
    assert gravedad(0.5, CFG) == "grave"
    assert gravedad(1.5, CFG) == "moderado"
    assert gravedad(2.5, CFG) == "leve"
