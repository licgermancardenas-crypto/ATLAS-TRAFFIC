"""Fase 2: cruce, sentido, conteo único, banda muerta y filtro por ROI."""

import numpy as np
import pandas as pd
import pytest

from atlas_traffic.config import Geometria, Linea
from atlas_traffic.counting import anotar_tracks, contar, cruces_de_track, resumen_conteo

# línea horizontal y=100 entre x=100 y x=300. A->B hacia la derecha: lado positivo = abajo.
LINEA = Linea(id="av", puntos=((100, 100), (300, 100)), sentido_positivo="entrante")
GEO = Geometria(
    encuadre="t",
    resolucion=(400, 400),
    roi=[(50, 20), (350, 20), (350, 300), (50, 300)],
    lineas=[
        LINEA,
        Linea(id="calle", puntos=((320, 50), (320, 250)), sentido_positivo="saliente"),
    ],
)


def _cruces(ys, xs=None, margen=4.0):
    ys = np.asarray(ys, dtype=float)
    xs = np.full_like(ys, 200.0) if xs is None else np.asarray(xs, dtype=float)
    frames = np.arange(len(ys)) * 2
    return cruces_de_track(frames, frames / 60, np.column_stack([xs, ys]), LINEA, margen)


def test_cruce_positivo_y_negativo():
    assert [c[2] for c in _cruces([60, 80, 95, 105, 120, 140])] == [True]  # baja: positivo
    assert [c[2] for c in _cruces([140, 120, 105, 95, 80])] == [False]  # sube: negativo


def test_instante_interpolado():
    ((frame, ts, _),) = _cruces([90, 110])  # cruza a mitad de camino entre frame 0 y 2
    assert frame == 1 and ts == pytest.approx(1 / 60)


def test_sin_cruce_si_no_pasa_la_linea():
    assert _cruces([20, 40, 60, 80, 90]) == []


def test_banda_muerta_ignora_temblor_sobre_la_linea():
    # auto detenido sobre la línea, oscilando ±3 px: no cuenta
    assert _cruces([97, 103, 98, 102, 97, 103, 99]) == []
    # con margen 0 el mismo temblor genera cruces falsos
    assert len(_cruces([97, 103, 98, 102, 97, 103, 99], margen=0)) > 0


def test_cruce_a_traves_de_la_banda_muerta_cuenta_una_vez():
    assert len(_cruces([80, 97, 99, 101, 103, 98, 102, 120])) == 1


def test_fuera_del_segmento_no_cuenta():
    # pasa de arriba a abajo de la recta y=100, pero en x=400 (el segmento llega a 300)
    assert _cruces([60, 140], xs=[400, 400]) == []
    # en diagonal, cortando el segmento
    assert len(_cruces([60, 140], xs=[250, 330])) == 1


def _det(tracks: dict[int, list[tuple[float, float]]], clase="vehiculo") -> pd.DataFrame:
    filas = []
    for tid, pts in tracks.items():
        for i, (x, y) in enumerate(pts):
            filas.append(dict(frame=2 * i, ts=2 * i / 60, track_id=tid, clase=clase,
                              clase_detalle="car", cx_ref=x, cy_ref=y))  # fmt: skip
    return pd.DataFrame(filas)


def test_contar_sentido_y_conteo_unico():
    det = _det({
        1: [(200, 60), (200, 120)],  # baja: entrante (positivo)
        2: [(200, 140), (200, 60)],  # sube: saliente
        3: [(200, 60), (200, 120), (200, 60), (200, 120)],  # baja, sube, baja: 1 + 1
    })  # fmt: skip
    c = contar(det, GEO)
    por_track = c.groupby("track_id")["sentido"].apply(sorted).to_dict()
    assert por_track == {1: ["entrante"], 2: ["saliente"], 3: ["entrante", "saliente"]}


def test_un_track_puede_cruzar_dos_lineas():
    # entra por av (baja) y sale por calle (cruza x=320 hacia la derecha)
    det = _det({7: [(200, 60), (200, 150), (300, 150), (340, 150)]})
    c = contar(det, GEO)
    assert list(zip(c.aproximacion, c.sentido, strict=True)) == [
        ("av", "entrante"),
        (
            "calle",
            "entrante",
        ),  # A->B hacia abajo: lado positivo = izquierda; ir a la derecha = negativo
    ]


def test_track_que_nunca_entra_a_la_roi_se_descarta():
    roi_chica = GEO.model_copy(update={"roi": [(150, 110), (250, 110), (250, 200), (150, 200)]})
    det = _det({1: [(200, 60), (200, 105)], 2: [(200, 60), (200, 150)]})
    assert set(contar(det, roi_chica)["track_id"]) == {2}


def test_resumen_incluye_ceros_y_clases():
    det = pd.concat([
        _det({1: [(200, 60), (200, 120)]}),
        _det({2: [(200, 60), (200, 120)]}, clase="peaton"),
    ])  # fmt: skip
    r = resumen_conteo(contar(det, GEO), GEO)
    assert r.loc[("av", "entrante"), "vehiculo"] == 1
    assert r.loc[("av", "entrante"), "peaton"] == 1
    assert r.loc[("calle", "saliente")].sum() == 0
    assert len(r) == 4


def test_contar_vacio():
    assert contar(pd.DataFrame(), GEO).empty
    assert resumen_conteo(contar(pd.DataFrame(), GEO), GEO).to_numpy().sum() == 0


def test_anotar_tracks():
    det = _det({1: [(200, 60), (200, 120)], 2: [(60, 60), (70, 60)]})
    tracks = pd.DataFrame({"track_id": [1, 2], "clase": ["vehiculo"] * 2})
    t = anotar_tracks(tracks, contar(det, GEO)).set_index("track_id")
    assert bool(t.loc[1, "cruzo_linea"]) and t.loc[1, "aproximacion"] == "av"
    assert t.loc[1, "sentido"] == "entrante" and t.loc[1, "n_cruces"] == 1
    assert not t.loc[2, "cruzo_linea"] and pd.isna(t.loc[2, "aproximacion"])
    # idempotente: anotar dos veces no duplica columnas
    assert list(anotar_tracks(t.reset_index(), contar(det, GEO)).columns) == list(
        t.reset_index().columns
    )


def test_track_que_aparece_dentro_de_la_banda_y_sale_cuenta():
    # detectado tarde, a 1 px de la línea del lado negativo, y después se aleja hacia el
    # lado positivo: cruzó
    assert [c[2] for c in _cruces([99, 104, 120, 140])] == [True]
    # aparece en la banda y se queda temblando: no cuenta
    assert _cruces([99, 102, 98, 101, 99]) == []


def test_cada_linea_cuenta_solo_sus_clases():
    calzada = LINEA.model_copy(update={"clases": ["vehiculo"]})
    senda = Linea(id="senda", puntos=((100, 200), (300, 200)), sentido_positivo="entrante",
                  clases=["peaton"])  # fmt: skip
    geo = GEO.model_copy(update={"lineas": [calzada, senda]})
    det = pd.concat([
        _det({1: [(200, 60), (200, 250)]}),  # auto: cruza las dos líneas
        _det({2: [(200, 60), (200, 250)]}, clase="peaton"),  # peatón: cruza las dos
    ])  # fmt: skip
    c = contar(det, geo)
    assert set(zip(c.track_id, c.aproximacion, strict=True)) == {(1, "av"), (2, "senda")}
