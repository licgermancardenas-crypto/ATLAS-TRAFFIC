"""Fase 1b: convención de lado, ROI, editor (sin GUI), guardado y render."""

from pathlib import Path

import numpy as np
import pytest

from atlas_traffic.config import Linea, load_geometria
from atlas_traffic.geometry import (
    EditorGeometria,
    dentro_roi,
    guardar_geometria,
    lado,
    normal_positiva,
)
from atlas_traffic.render import dibujar_geometria

FIXTURES = Path(__file__).parent / "fixtures"
HORIZONTAL = Linea(id="h", puntos=((100, 100), (200, 100)), sentido_positivo="entrante")


def test_lado_y_normal_son_coherentes():
    # en coordenadas de imagen (y hacia abajo), para A->B hacia la derecha el lado
    # positivo es abajo
    assert lado(HORIZONTAL, (150, 120)) > 0
    assert lado(HORIZONTAL, (150, 80)) < 0
    assert lado(HORIZONTAL, (150, 100)) == 0
    n = normal_positiva(HORIZONTAL)
    np.testing.assert_allclose(n, [0, 1])
    assert lado(HORIZONTAL, np.array([150, 100]) + 10 * n) > 0


def test_lado_vectorizado():
    pts = np.array([[150, 120], [150, 80], [10, 100]])
    np.testing.assert_array_equal(np.sign(lado(HORIZONTAL, pts)), [1, -1, 0])


def test_dentro_roi():
    roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
    pts = np.array([[50, 50], [150, 50], [100, 50]])
    np.testing.assert_array_equal(dentro_roi(roi, pts), [True, False, True])  # borde = adentro


def _editor_completo() -> EditorGeometria:
    ed = EditorGeometria(encuadre="enc", resolucion=(400, 300))
    for p in [(10, 10), (390, 10), (390, 290), (10, 290)]:
        ed.clic(*p)
    ed.cerrar_roi()
    ed.clic(50, 150)
    ed.clic(150, 150)
    ed.confirmar_linea("av_norte", "entrante")
    return ed


def test_editor_flujo_completo():
    ed = _editor_completo()
    geo = ed.geometria()
    assert geo.encuadre == "enc" and len(geo.roi) == 4
    assert geo.lineas[0].puntos == ((50, 150), (150, 150))
    assert geo.lineas[0].sentido_positivo == "entrante"
    assert geo.lineas[0].sentido_permitido == "ambos"  # default: no genera alertas
    ed.clic(50, 200)
    ed.clic(150, 200)
    assert ed.confirmar_linea("calle", "saliente", "entrante").sentido_permitido == "entrante"


def test_editor_valida():
    ed = EditorGeometria(encuadre="e", resolucion=(100, 100))
    ed.clic(1, 1)
    ed.clic(50, 1)
    with pytest.raises(ValueError, match="3 puntos"):
        ed.cerrar_roi()
    ed.clic(50, 50)
    ed.cerrar_roi()
    with pytest.raises(ValueError, match="dos extremos"):
        ed.confirmar_linea("x", "entrante")
    ed.clic(10, 10)
    ed.clic(10, 10)  # mismo punto dos veces: se ignora el segundo
    assert not ed.linea_lista
    ed.clic(20, 10)
    ed.confirmar_linea("x", "entrante")
    ed.clic(30, 30)
    ed.clic(40, 30)
    with pytest.raises(ValueError, match="ya existe"):
        ed.confirmar_linea("x", "saliente")


def test_editor_recorta_clics_fuera_del_frame():
    ed = EditorGeometria(encuadre="e", resolucion=(100, 80))
    ed.clic(-5, 500)
    assert ed.roi == [(0, 80)]


def test_editor_deshacer_retrocede_por_etapas():
    ed = _editor_completo()
    ed.clic(10, 20)
    ed.deshacer()  # punto pendiente
    assert ed.pendiente == [] and len(ed.lineas) == 1
    ed.deshacer()  # la línea
    assert ed.lineas == [] and ed.etapa == "lineas"
    ed.deshacer()  # reabre la ROI
    assert ed.etapa == "roi" and len(ed.roi) == 4
    ed.deshacer()  # un vértice
    assert len(ed.roi) == 3
    ed.reiniciar()
    assert ed.roi == [] and ed.etapa == "roi"


def test_guardar_y_recargar(tmp_path):
    geo = _editor_completo().geometria()
    ruta = tmp_path / "geometria" / "enc.yaml"
    guardar_geometria(geo, ruta)
    assert load_geometria(ruta) == geo
    ed = EditorGeometria.desde_geometria(geo)
    assert ed.etapa == "lineas" and ed.geometria() == geo


def test_render_dibuja_en_la_posicion_correcta():
    geo = load_geometria(FIXTURES / "encuadre_test.yaml")
    frame = np.zeros((1080, 1920, 3), np.uint8)
    out = dibujar_geometria(frame, geo.roi, geo.lineas)
    av_norte = geo.lineas[0]  # (600,300)-(1200,300), entrante
    assert out[300, 900].any()  # píxel sobre la línea
    assert not out[600, 50].any()  # fuera de la ROI y de las líneas
    assert out[600, 900].any()  # dentro de la ROI (relleno semitransparente)
    # la flecha sale del punto medio hacia el lado positivo (abajo para A->B a la derecha)
    assert out[300 + 40, 900].any() and lado(av_norte, (900, 340)) > 0


def test_render_con_homografia_desplaza_la_geometria():
    geo = load_geometria(FIXTURES / "encuadre_test.yaml")
    H = np.array([[1, 0, 50], [0, 1, 0], [0, 0, 1]], dtype=float)  # 50 px a la derecha
    out = dibujar_geometria(np.zeros((1080, 1920, 3), np.uint8), [], geo.lineas[:1], H=H)
    assert out[300, 1240].any()  # el extremo B (1200) quedó en 1250
    assert not out[300, 620].any()  # antes del nuevo extremo A (650) no hay línea
