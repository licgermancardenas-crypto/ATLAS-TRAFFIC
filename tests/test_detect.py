"""Fase 0b: el protocolo de detección se ejercita con FakeDetector, sin Ultralytics."""

import subprocess
import sys

import numpy as np
import pytest

from atlas_traffic.detect import (
    Deteccion,
    Detector,
    FakeDetector,
    crear_detector,
    disponibles,
    registrar,
)

FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)


def _det(tid, x=100, clase="vehiculo"):
    return Deteccion(bbox=(x, 100, x + 50, 140), clase=clase, confianza=0.9, track_id=tid)


def test_fake_cumple_el_protocolo():
    assert isinstance(FakeDetector(), Detector)


def test_fake_reproduce_el_guion_y_reset_rebobina():
    guion = [[_det(1)], [_det(1, x=110), _det(2, clase="peaton")], []]
    det: Detector = FakeDetector(guion)
    salidas = [det.track(FRAME) for _ in range(4)]
    assert [len(s) for s in salidas] == [1, 2, 0, 0]  # pasado el guion: vacío
    assert salidas[1][1].clase == "peaton"
    det.reset()
    assert det.track(FRAME) == guion[0]


def test_fake_no_expone_su_estado_interno():
    det = FakeDetector([[_det(1)]])
    det.track(FRAME).clear()
    det.reset()
    assert len(det.track(FRAME)) == 1


def test_fake_exige_frame_ndarray():
    with pytest.raises(TypeError):
        FakeDetector().track("no soy un frame")


def test_deteccion_valida_bbox_clase_y_confianza():
    assert _det(1).centroide == (125.0, 120.0)
    with pytest.raises(ValueError, match="bbox"):
        Deteccion(bbox=(10, 10, 5, 20), clase="vehiculo", confianza=0.5, track_id=None)
    with pytest.raises(ValueError, match="vocabulario"):
        Deteccion(bbox=(0, 0, 5, 5), clase="car", confianza=0.5, track_id=None)  # clase COCO
    with pytest.raises(ValueError, match="confianza"):
        Deteccion(bbox=(0, 0, 5, 5), clase="peaton", confianza=1.5, track_id=None)


def test_registro_por_nombre():
    assert "fake" in disponibles()
    det = crear_detector("fake", guion=[[_det(7)]])
    assert det.track(FRAME)[0].track_id == 7
    with pytest.raises(KeyError, match="desconocido"):
        crear_detector("no-existe")
    with pytest.raises(ValueError, match="ya registrado"):
        registrar("fake")(FakeDetector)


def test_ultralytics_no_se_importa():
    """Importar el paquete y usar el protocolo no carga Ultralytics (ni torch)."""
    codigo = (
        "import sys, numpy as np\n"
        "import atlas_traffic, atlas_traffic.cli, atlas_traffic.detect as d\n"
        "for m in ['counting','metrics','attributes','alerts','congestion','aggregate',"
        "'render','geometry','clips','config']:\n"
        "    __import__('atlas_traffic.' + m)\n"
        "d.crear_detector('fake').track(np.zeros((2,2,3), np.uint8))\n"
        "assert 'ultralytics' not in sys.modules, 'ultralytics importado'\n"
        "assert 'torch' not in sys.modules, 'torch importado'\n"
    )
    res = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
