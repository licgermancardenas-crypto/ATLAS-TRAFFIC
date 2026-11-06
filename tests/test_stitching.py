"""Unión de tracks cortados por una oclusión."""

import pandas as pd

from atlas_traffic.stitching import aplicar_uniones, unir_tracks


def _tramo(tid, frames, x0, vx, clase="vehiculo", alto=30, y=100.0, fps=60):
    filas = []
    for f in frames:
        x = x0 + vx * (f / fps)
        filas.append(dict(frame=f, ts=f / fps, track_id=tid, clase=clase, cx_ref=x, cy_ref=y,
                          x1=x - 10, x2=x + 10, y1=y - alto / 2, y2=y + alto / 2))  # fmt: skip
    return filas


def _det(*tramos):
    return pd.DataFrame([f for t in tramos for f in t])


def test_une_el_mismo_objeto_tras_una_oclusion_breve():
    # va a 60 px/s; desaparece 0.3 s y reaparece donde tenía que estar
    det = _det(_tramo(1, range(0, 60, 2), 0, 60), _tramo(2, range(78, 140, 2), 0, 60))
    assert unir_tracks(det, hueco_max_s=0.5, factor_distancia=0.5) == {2: 1}
    assert set(aplicar_uniones(det, {2: 1})["track_id"]) == {1}


def test_no_une_si_reaparece_en_otro_lugar_o_tarde():
    lejos = _det(_tramo(1, range(0, 60, 2), 0, 60), _tramo(2, range(78, 140, 2), 200, 60))
    assert unir_tracks(lejos, hueco_max_s=0.5, factor_distancia=0.5) == {}
    tarde = _det(_tramo(1, range(0, 60, 2), 0, 60), _tramo(2, range(120, 180, 2), 0, 60))
    assert unir_tracks(tarde, hueco_max_s=0.5, factor_distancia=0.5) == {}


def test_no_une_clases_distintas_ni_tracks_simultaneos():
    clases = _det(_tramo(1, range(0, 60, 2), 0, 60),
                  _tramo(2, range(78, 140, 2), 0, 60, clase="peaton"))  # fmt: skip
    assert unir_tracks(clases, hueco_max_s=0.5) == {}
    # dos personas caminando juntas: los tracks se superponen en el tiempo
    juntos = _det(_tramo(1, range(0, 60, 2), 0, 60), _tramo(2, range(40, 100, 2), 0, 60))
    assert unir_tracks(juntos, hueco_max_s=0.5) == {}


def test_cadena_de_tres_pedazos_conserva_el_primer_numero():
    det = _det(_tramo(5, range(0, 40, 2), 0, 60), _tramo(9, range(50, 90, 2), 0, 60),
               _tramo(7, range(100, 140, 2), 0, 60))  # fmt: skip
    assert unir_tracks(det, hueco_max_s=0.5, factor_distancia=0.5) == {9: 5, 7: 5}


def test_cada_pedazo_continua_a_uno_solo():
    # dos candidatos para continuar al track 1: gana el que mejor encaja
    det = _det(_tramo(1, range(0, 60, 2), 0, 60), _tramo(2, range(70, 120, 2), 0, 60),
               _tramo(3, range(70, 120, 2), 8, 60))  # fmt: skip
    assert unir_tracks(det, hueco_max_s=0.5, factor_distancia=0.5) == {2: 1}


def test_sin_detecciones():
    assert unir_tracks(pd.DataFrame()) == {}


def test_pasada_de_personas_une_huecos_largos_solo_si_caminan():
    # una persona camina a 12 px/s y el tracker la pierde 1.2 s
    caminando = _det(_tramo(1, range(0, 120, 2), 0, 12, clase="peaton"),
                     _tramo(2, range(192, 300, 2), 0, 12, clase="peaton"))  # fmt: skip
    assert unir_tracks(caminando, 0.5, 0.5) == {}  # la pasada estricta no la une
    kw = dict(clases=("peaton",), velocidad_min_px_s=4.0)
    assert unir_tracks(caminando, 1.5, 1.5, **kw) == {2: 1}
    # quieta (esperando en la esquina): no se une, no está cruzando nada
    quieta = _det(_tramo(1, range(0, 120, 2), 0, 0, clase="peaton"),
                  _tramo(2, range(192, 300, 2), 0, 0, clase="peaton"))  # fmt: skip
    assert unir_tracks(quieta, 1.5, 1.5, **kw) == {}
    # los vehículos quedan afuera de esta pasada
    autos = _det(_tramo(1, range(0, 120, 2), 0, 12), _tramo(2, range(192, 300, 2), 0, 12))
    assert unir_tracks(autos, 1.5, 1.5, **kw) == {}
