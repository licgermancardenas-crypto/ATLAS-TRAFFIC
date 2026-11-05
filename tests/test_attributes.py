"""Fase 3b: clasificación HSV, votación, muestreo y pasada sobre video."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from atlas_traffic.attributes import (
    INDEFINIDO,
    color_de_region,
    extraer_atributos,
    muestrear,
    region_color,
    solapes,
    votar_color,
)
from atlas_traffic.config import ColorCfg

CFG = ColorCfg()

BGR = {
    "blanco": (235, 235, 235),
    "gris": (130, 130, 130),
    "negro": (25, 25, 25),
    "rojo": (30, 30, 200),
    "azul": (200, 60, 20),
    "verde": (40, 170, 40),
    "amarillo": (20, 200, 230),
}


@pytest.mark.parametrize("nombre", list(BGR))
def test_paleta(nombre):
    parche = np.full((20, 20, 3), BGR[nombre], np.uint8)
    assert color_de_region(parche, CFG) == nombre


def test_rojo_del_otro_lado_del_circulo_de_matiz():
    hsv = np.full((10, 10, 3), (172, 200, 200), np.uint8)  # magenta-rojo
    assert color_de_region(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), CFG) == "rojo"


def test_region_vacia_o_sin_votos():
    assert color_de_region(np.zeros((0, 0, 3), np.uint8), CFG) is None
    violeta = cv2.cvtColor(np.full((5, 5, 3), (145, 220, 220), np.uint8), cv2.COLOR_HSV2BGR)
    assert color_de_region(violeta, CFG) is None  # matiz intermedio: no vota


def test_votacion():
    assert votar_color(["rojo"] * 7 + ["gris"] * 3, 0.5) == ("rojo", 0.7)
    assert votar_color(["rojo", "gris", "negro"], 0.5)[0] == INDEFINIDO  # 0.33: se abstiene
    assert votar_color([], 0.5) == (INDEFINIDO, 0.0)


def test_region_central_sesgada_hacia_arriba_en_vehiculos():
    assert region_color((0, 0, 100, 100), "vehiculo") == (25, 10, 75, 60)
    assert region_color((0, 0, 100, 100), "peaton") == (25, 25, 75, 75)


def _det(filas):
    cols = ["track_id", "frame", "clase", "x1", "y1", "x2", "y2"]
    return pd.DataFrame(filas, columns=cols)


def test_solapes():
    d = _det([(1, 0, "vehiculo", 0, 0, 10, 10), (2, 0, "vehiculo", 5, 0, 15, 10),
              (3, 1, "vehiculo", 0, 0, 10, 10)])  # fmt: skip
    np.testing.assert_allclose(solapes(d), [0.5, 0.5, 0.0])


def test_muestreo_descarta_borde_y_solape_y_elige_thumb():
    filas = [(1, f, "vehiculo", 10 + f, 10, 30 + f, 30) for f in range(20)]
    filas += [(1, 20, "vehiculo", 0, 10, 40, 40)]  # toca el borde: no vota
    filas += [(2, 5, "vehiculo", 12, 12, 32, 32)]  # se solapa con el track 1 en el frame 5
    m = muestrear(_det(filas), ancho=200, alto=200, cfg=CFG).set_index(["track_id", "frame"])
    votos_1 = m.loc[1][m.loc[1]["votar"]]
    assert len(votos_1) == CFG.max_frames  # 10 repartidos entre los válidos
    assert 20 not in votos_1.index and 5 not in votos_1.index
    assert m.loc[1]["es_thumb"].sum() == 1
    # el track 2 solo tiene una detección, solapada: no vota, pero igual tiene recorte
    assert not m.loc[(2, 5), "votar"] and m.loc[(2, 5), "es_thumb"]


def test_extraer_atributos_sobre_video(tmp_path: Path):
    # video gris con un rectángulo rojo (track 1) y uno blanco (track 2) que se mueven
    ruta = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), 10, (200, 120))
    filas = []
    for f in range(15):
        img = np.full((120, 200, 3), (90, 90, 90), np.uint8)
        x = 20 + 3 * f
        cv2.rectangle(img, (x, 20), (x + 40, 60), BGR["rojo"], -1)
        cv2.rectangle(img, (x + 60, 50), (x + 100, 90), BGR["blanco"], -1)
        w.write(img)
        filas += [
            (1, f, "vehiculo", x, 20, x + 40, 60),
            (2, f, "vehiculo", x + 60, 50, x + 100, 90),
        ]
        filas += [(3, f, "peaton", 5, 95, 15, 115)]
    w.release()
    out = extraer_atributos(ruta, _det(filas), CFG, tmp_path / "thumbs").set_index("track_id")
    assert out.loc[1, "color"] == "rojo" and out.loc[1, "color_confianza"] == 1.0
    assert out.loc[2, "color"] == "blanco"
    assert pd.isna(out.loc[3, "color"])  # peatones sin color
    for tid in (1, 2, 3):
        assert (tmp_path / out.loc[tid, "thumb"]).exists()
