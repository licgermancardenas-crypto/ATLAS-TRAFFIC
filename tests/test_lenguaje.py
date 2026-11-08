"""Búsqueda en lenguaje natural."""

import datetime as dt
import subprocess
import sys
from pathlib import Path

import pytest

from atlas_traffic.config import load_camaras
from atlas_traffic.lenguaje import interpretar

RAIZ = Path(__file__).resolve().parents[1]
CAMARAS = load_camaras(RAIZ / "configs" / "camaras.yaml")
LINEAS = ["av_cordoba_frente", "av_cordoba_fondo", "rodriguez_pena_izq", "ciclovia_cordoba",
          "senda_cordoba", "senda_rodriguez_pena_der"]  # fmt: skip
COLORES = ["amarillo", "azul", "blanco", "gris", "negro", "rojo"]


def _i(texto):
    return interpretar(texto, CAMARAS, LINEAS, COLORES, anio=2026)


def test_el_ejemplo_del_pedido():
    i = _i("busco sospechoso en bicicleta por avenida cordoba al rededor de las 7:30 pm")
    f = i.filtro
    assert f.clases == ["person"] and f.aproximaciones == ["ciclovia_cordoba"]
    assert f.franja == (dt.time(19, 0), dt.time(20, 0))  # "alrededor": ±30 min
    avisos = " ".join(i.avisos)
    assert "conductas" in avisos and "bicicletas" in avisos
    assert i.criterios is None  # una bicicleta no se puede guardar como vehículo de interés


def test_auto_rojo_por_cordoba():
    i = _i("Auto rojo por Córdoba a las 19:30")
    f = i.filtro
    assert f.clases == ["car"] and f.colores == ["rojo"]
    assert f.aproximaciones == ["av_cordoba_frente", "av_cordoba_fondo", "ciclovia_cordoba",
                                "senda_cordoba"]  # fmt: skip
    assert f.franja == (dt.time(19, 15), dt.time(19, 45))  # hora exacta: ±15 min
    assert i.criterios["calles"] == ["av_cordoba"] and i.criterios["hora_desde"] == "19:15"


@pytest.mark.parametrize("texto, franja", [
    ("autos de 19 a 20", ((19, 0), (20, 0))),
    ("entre las 12 y las 12:30", ((12, 0), (12, 30))),
    ("despues de las 19", ((19, 0), (23, 59))),
    ("antes de las 8", ((0, 0), (8, 0))),
    ("a las 7 de la tarde", ((18, 45), (19, 15))),
    ("motos de noche", ((19, 0), (5, 59))),
    ("a la mañana", ((6, 0), (11, 59))),
    ("19 hs", ((18, 45), (19, 15))),
])  # fmt: skip
def test_horarios(texto, franja):
    (h0, m0), (h1, m1) = franja
    assert _i(texto).filtro.franja == (dt.time(h0, m0), dt.time(h1, m1))


def test_hora_ambigua_avisa():
    i = _i("a las 7:30")
    assert i.filtro.franja == (dt.time(7, 15), dt.time(7, 45))
    assert any("19:30" in a for a in i.avisos)


def test_velocidad_no_se_confunde_con_hora():
    i = _i("autos blancos a más de 35 km/h")
    assert i.filtro.velocidad_min_kmh == 35 and i.filtro.franja is None
    assert _i("moto rápida").filtro.velocidad_min_kmh == 40


def test_fecha_barrio_y_taxi():
    i = _i("taxis en Recoleta el 4/11")
    assert i.filtro.clases == ["car"] and i.filtro.colores == ["amarillo"]
    assert i.filtro.fechas == (dt.date(2026, 11, 4), dt.date(2026, 11, 4))
    assert "encuadre_m1" in i.filtro.encuadres
    assert _i("4 de noviembre").filtro.fechas[0] == dt.date(2026, 11, 4)


def test_lo_que_no_se_puede_se_avisa():
    i = _i("camioneta gris ford patente AB123CD")
    assert i.filtro.clases == ["car"] and i.filtro.colores == ["gris"]
    avisos = " ".join(i.avisos)
    assert "marca" in avisos and "patentes" in avisos and "camionetas" in avisos
    assert any("verde" in a for a in _i("auto verde").avisos)  # color sin datos


def test_sin_criterios():
    i = _i("hola")
    assert not i.entendido and i.avisos and i.filtro.clases == []


SCRIPT = f"""
import sys
sys.modules["cv2"] = None
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({str(RAIZ / "dashboard" / "app.py")!r}, default_timeout=120)
at.query_params["vista"] = "asistente"
at.run()
at.chat_input[0].set_value("auto rojo por Córdoba alrededor de las 7:30 pm").run()
assert not at.exception, [str(e.value) for e in at.exception]
assert not at.error, [e.value for e in at.error]
textos = [m.value for m in at.markdown]
assert any(t.startswith("**Entendí:** tipo: auto") for t in textos), textos
assert any(t.startswith("**Encontré") or "No encontré" in t for t in textos), textos
print("ok")
"""


def test_asistente_en_el_dashboard_sin_opencv():
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=RAIZ, capture_output=True,
                       text=True, timeout=300)  # fmt: skip
    assert r.returncode == 0 and "ok" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]
