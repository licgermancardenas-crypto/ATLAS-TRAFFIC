from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas_traffic.clips import get_clip, load_clips
from atlas_traffic.config import (
    AppConfig,
    Calibracion,
    Clip,
    ClipsFile,
    Geometria,
    load_app_config,
    load_calibracion,
    load_geometria,
    save_calibracion,
)

RAIZ = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"

CLIP_OK = dict(
    id="c1",
    archivo="data/raw/c1.mp4",
    fecha="2026-09-15",
    hora_inicio="08:00",
    dia_semana="martes",
    tipo_dia="habil",
    clima="despejado",
    geometria="configs/geometria/encuadre_a.yaml",
)


def test_default_yaml_del_repo_es_valido():
    cfg = load_app_config(RAIZ / "configs/default.yaml")
    assert cfg.deteccion.detector == "ultralytics"
    assert cfg.metricas.intervalo_s == 30
    assert cfg == AppConfig()  # el YAML documenta exactamente los defaults


def test_clips_yaml_del_repo_es_valido():
    clips = load_clips(RAIZ / "configs/clips.yaml")
    assert len(clips) >= 1
    assert get_clip(clips, clips[0].id) is clips[0]
    with pytest.raises(KeyError):
        get_clip(clips, "no-existe")


def test_campo_desconocido_es_error():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"deteccion": {"detectro": "x"}})


def test_dia_semana_acepta_acentos_y_valida_contra_fecha():
    clip = Clip.model_validate(
        {**CLIP_OK, "fecha": "2026-09-19", "dia_semana": "Sábado", "tipo_dia": "finde"}
    )
    assert clip.dia_semana == "sabado"
    with pytest.raises(ValidationError, match="es martes"):
        Clip.model_validate({**CLIP_OK, "dia_semana": "lunes"})


@pytest.mark.parametrize("hora", ["8:00", "24:00", "08:60", "0800"])
def test_hora_inicio_invalida(hora):
    with pytest.raises(ValidationError):
        Clip.model_validate({**CLIP_OK, "hora_inicio": hora})


def test_ids_de_clip_repetidos():
    with pytest.raises(ValidationError, match="repetidos"):
        ClipsFile.model_validate({"clips": [CLIP_OK, CLIP_OK]})


def test_geometria_fixture():
    geo = load_geometria(FIXTURES / "encuadre_test.yaml")
    assert len(geo.lineas) == 3
    assert {linea.sentido_positivo for linea in geo.lineas} == {"entrante", "saliente"}


def test_geometria_rechaza_puntos_fuera_y_lineas_degeneradas():
    base = {"encuadre": "e", "resolucion": [100, 100], "roi": [[0, 0], [100, 0], [100, 100]]}
    linea = {"id": "l", "puntos": [[10, 10], [50, 10]], "sentido_positivo": "entrante"}
    Geometria.model_validate({**base, "lineas": [linea]})
    with pytest.raises(ValidationError, match="fuera"):
        Geometria.model_validate({**base, "lineas": [{**linea, "puntos": [[10, 10], [500, 10]]}]})
    with pytest.raises(ValidationError, match="iguales"):
        Geometria.model_validate({**base, "lineas": [{**linea, "puntos": [[10, 10], [10, 10]]}]})
    with pytest.raises(ValidationError, match="repetidos"):
        Geometria.model_validate({**base, "lineas": [linea, linea]})


def test_calibracion_roundtrip(tmp_path):
    cal = Calibracion(
        velocidad_p90_por_encuadre={"encuadre_a": 7.5},
        densidad={"p33": 2.0, "p66": 5.0},
        velocidad_norm={"p33": 0.4, "p66": 0.7},
    )
    ruta = tmp_path / "calibracion.yaml"
    save_calibracion(cal, ruta)
    assert load_calibracion(ruta) == cal
    with pytest.raises(ValidationError):
        Calibracion.model_validate({**cal.model_dump(), "densidad": {"p33": 5, "p66": 2}})
