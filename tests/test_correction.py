"""Corrección de conteos por tasa de detección, con intervalo de confianza."""

import numpy as np
import pandas as pd
import pytest

from atlas_traffic.correction import (
    calcular_tasas,
    corregir,
    corregir_conteo,
    detectar_luz,
    flujo_entrante,
    wilson,
)


def test_wilson_valores_conocidos():
    p, lo, hi = wilson(38, 40)
    assert p == pytest.approx(0.95)
    assert lo == pytest.approx(0.835, abs=0.002) and hi == pytest.approx(0.986, abs=0.002)
    assert wilson(5, 5)[2] == pytest.approx(1.0) and wilson(0, 5)[1] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        wilson(0, 0)


def _registros():
    return pd.DataFrame({
        "clip_id": ["a", "b", "c", "a"],
        "linea": ["frente", "frente", "frente", "senda"],
        "luz": ["dia", "dia", "noche", "dia"],
        "manual": [22, 18, 14, 0],
        "automatico": [21, 17, 9, 0],
    })  # fmt: skip


def test_tasas_por_linea_y_luz():
    t = calcular_tasas(_registros())
    assert t[("frente", "dia")].detectados == 38 and t[("frente", "dia")].reales == 40
    assert t[("frente", "noche")].tasa == pytest.approx(9 / 14)
    assert ("senda", "dia") not in t  # sin vehículos reales: no hay nada que medir


def test_tasa_no_premia_falsos_positivos():
    reg = pd.DataFrame({"clip_id": ["a"], "linea": ["x"], "luz": ["dia"], "manual": [10],
                        "automatico": [12]})  # fmt: skip
    assert calcular_tasas(reg)[("x", "dia")].tasa == 1.0


def test_corregir_con_y_sin_tasa():
    t = calcular_tasas(_registros())[("frente", "dia")]
    c, lo, hi = corregir(19, t)
    assert c == pytest.approx(20.0) and lo < c < hi
    c, lo, hi = corregir(19, None)
    assert c == 19 and np.isnan(lo) and np.isnan(hi)


def test_corregir_conteo_marca_lineas_sin_validar():
    conteo = pd.DataFrame({"vehiculo": [19, 3], "peaton": [0, 0]},
                          index=pd.MultiIndex.from_tuples(
                              [("frente", "entrante"), ("otra", "entrante")],
                              names=["aproximacion", "sentido"]))  # fmt: skip
    out = corregir_conteo(conteo, calcular_tasas(_registros()), "dia")
    assert out.loc[("frente", "entrante"), "vehiculo_corregido"] == pytest.approx(20.0)
    assert bool(out.loc[("frente", "entrante"), "validada"])
    assert not out.loc[("otra", "entrante"), "validada"]
    assert out.loc[("otra", "entrante"), "vehiculo_corregido"] == 3


def test_detectar_luz():
    dia = np.full((100, 100, 3), (110, 112, 115), np.uint8)  # BGR casi neutro
    noche = np.full((100, 100, 3), (50, 77, 93), np.uint8)  # alumbrado de sodio: R >> B
    assert detectar_luz(dia) == "dia" and detectar_luz(noche) == "noche"


def test_flujo_entrante_corregido_por_ventana():
    cruces = pd.DataFrame({
        "ts_cruce": [5.0, 10.0, 40.0, 41.0, 42.0],
        "aproximacion": ["frente", "frente", "frente", "otra", "frente"],
        "sentido": ["entrante", "entrante", "entrante", "entrante", "saliente"],
        "clase": ["vehiculo"] * 5,
    })  # fmt: skip
    t = calcular_tasas(_registros())
    f = flujo_entrante(cruces, t, "dia", np.array([0.0, 30.0, 60.0]))
    assert list(f["flujo_entrante_min"]) == [4.0, 4.0]  # 2 por ventana de 30 s (saliente no)
    assert f.loc[0, "flujo_entrante_corr_min"] == pytest.approx(4 / 0.95)
    assert f.loc[0, "flujo_entrante_ic_bajo"] < f.loc[0, "flujo_entrante_corr_min"]
    assert bool(f.loc[0, "flujo_entrante_validado"])
    assert not f.loc[1, "flujo_entrante_validado"]  # entra un vehículo por una línea sin validar
