"""Senda bloqueada y detención prolongada."""

import numpy as np
import pandas as pd

from atlas_traffic.config import InfraccionesCfg
from atlas_traffic.infracciones import apoyo_en_sendas, detectar_infracciones

CFG = InfraccionesCfg()
SENDAS = {"frente": ((0.0, 18.0), (-2.5, 2.5))}
T = np.arange(0, 40, 1 / 10)


def _veh(tid, x, y, vel, t=T, frac=0.0, senda="frente"):
    return pd.DataFrame({"track_id": tid, "ts": t, "x_m": x, "y_m": y, "vel_kmh": vel,
                         "senda": senda, "frac_senda": frac})  # fmt: skip


def _circulando(n, x, t0=0.0):
    # n autos que pasan por el carril x, uno cada 3 s
    return pd.concat([_veh(100 + k, x, (T - t0 - 3 * k) * 8, 30.0) for k in range(n)])


SIN_PEATONES = pd.DataFrame(columns=["track_id", "ts", "x_m", "y_m"])


def test_auto_detenido_sobre_la_senda():
    t = np.arange(10, 22, 0.1)  # 12 s detenido encima
    ev = detectar_infracciones(_veh(1, 5.0, 0.0, 0.5, t, frac=0.9), SIN_PEATONES, SENDAS,
                               {}, CFG)  # fmt: skip
    assert list(ev["tipo"]) == ["senda_bloqueada"]
    assert abs(ev["duracion_s"].iat[0] - 11.9) < 0.2 and ev["gravedad"].iat[0] == "moderado"


def test_con_peatones_en_la_senda_sube_la_gravedad():
    t = np.arange(10, 22, 0.1)
    ped = pd.DataFrame({"track_id": 9, "ts": t, "x_m": 3.0, "y_m": np.linspace(-2, 2, len(t))})
    ev = detectar_infracciones(_veh(1, 5.0, 0.0, 0.5, t, frac=0.9), ped, SENDAS, {}, CFG)
    assert ev["peatones"].iat[0] == 1 and ev["gravedad"].iat[0] == "grave"


def test_pasar_por_la_senda_sin_detenerse_no_cuenta():
    t = np.arange(10, 22, 0.1)
    ev = detectar_infracciones(_veh(1, 5.0, 0.0, 25.0, t, frac=1.0), SIN_PEATONES, SENDAS,
                               {}, CFG)  # fmt: skip
    assert ev.empty


def test_seguimiento_partido_se_une():
    a = _veh(1, 5.0, 0.0, 0.5, np.arange(10, 16, 0.1), frac=0.9)
    b = _veh(2, 5.3, 0.1, 0.5, np.arange(18, 25, 0.1), frac=0.9)  # 2 s después, mismo lugar
    ev = detectar_infracciones(pd.concat([a, b]), SIN_PEATONES, SENDAS, {}, CFG)
    assert len(ev) == 1 and ev["track_ids"].iat[0] == "1,2"


def test_detencion_prolongada_con_transito_al_lado():
    quieto = _veh(1, 10.0, 15.0, 0.5, senda=None)
    ev = detectar_infracciones(pd.concat([quieto, _circulando(8, 12.0)]), SIN_PEATONES,
                               SENDAS, {}, CFG)  # fmt: skip
    assert list(ev["tipo"]) == ["detencion_prolongada"] and ev["duracion_s"].iat[0] > 35


def test_cola_de_semaforo_no_es_detencion_prolongada():
    # todos parados: cola
    cola = pd.concat([_veh(k, 10.0, 15.0 + 5 * k, 0.5, senda=None) for k in range(4)])
    ev = detectar_infracciones(pd.concat([cola, _circulando(8, 12.0)]), SIN_PEATONES,
                               SENDAS, {}, CFG)  # fmt: skip
    assert ev.empty


def test_colectivo_en_la_parada_no_cuenta():
    quieto = _veh(1, 10.0, 15.0, 0.5, senda=None)
    ev = detectar_infracciones(pd.concat([quieto, _circulando(8, 12.0)]), SIN_PEATONES,
                               SENDAS, {1: "bus"}, CFG)  # fmt: skip
    assert ev.empty


def test_apoyo_en_sendas_proyecta_la_parte_baja_de_la_caja():
    # suelo = imagen / 10 (H diagonal); caja de 40x40 px con la base en y=20 px (2 m)
    tray = pd.DataFrame({"frame": [0, 0], "x1": [40, 40], "y1": [-20, 100],
                         "x2": [80, 80], "y2": [20, 140]})  # fmt: skip
    H = np.diag([0.1, 0.1, 1.0])
    a = apoyo_en_sendas(tray, {}, H, SENDAS)
    assert a["frac_senda"].iat[0] == 1.0  # base entre y=0.4 y 2 m: dentro de la senda
    assert a["frac_senda"].iat[1] == 0.0  # base a 10-14 m: fuera
