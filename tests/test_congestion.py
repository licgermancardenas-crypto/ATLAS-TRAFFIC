"""Fase 4: niveles, matriz, calibración por percentiles, aplicación y comando calibrate."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from atlas_traffic.cli import app
from atlas_traffic.config import Calibracion, Cortes, load_calibracion
from atlas_traffic.congestion import MATRIZ, aplicar, calibrar, clasificar, nivel

CAL = Calibracion(
    velocidad_p90_por_encuadre={"e": 2.0},
    densidad=Cortes(p33=3.0, p66=6.0),
    velocidad_norm=Cortes(p33=0.3, p66=0.6),
)


def test_nivel():
    c = Cortes(p33=1.0, p66=2.0)
    assert [nivel(v, c) for v in (0.5, 1.0, 1.5, 2.0, 9.0)] == [
        "baja", "media", "media", "alta", "alta",
    ]  # fmt: skip
    assert nivel(float("nan"), c) is None


@pytest.mark.parametrize(
    ("densidad", "vnorm", "esperado"),
    [
        (1, 0.1, "LOW"), (1, 0.5, "LOW"), (1, 0.9, "LOW"),  # densidad baja: siempre LOW
        (4, 0.9, "LOW"), (4, 0.5, "MODERATE"), (4, 0.1, "MODERATE"),
        (8, 0.9, "MODERATE"), (8, 0.5, "HIGH"), (8, 0.1, "HIGH"),
    ],
)  # fmt: skip
def test_matriz_del_brief(densidad, vnorm, esperado):
    assert clasificar(densidad, vnorm, CAL) == esperado


def test_matriz_completa():
    assert len(MATRIZ) == 9 and set(MATRIZ.values()) == {"LOW", "MODERATE", "HIGH"}


def test_sin_velocidad():
    assert clasificar(1, float("nan"), CAL) == "LOW"  # poca densidad: no hace falta velocidad
    assert clasificar(8, float("nan"), CAL) is None  # mucha densidad sin velocidad: no se decide


def test_calibrar_percentiles_y_normalizacion_por_encuadre():
    # dos encuadres con escalas distintas en px: la velocidad normalizada debe igualarlas
    iv = pd.DataFrame({
        "encuadre": ["a"] * 3 + ["b"] * 3,
        "densidad_media": [1, 2, 3, 4, 5, 6],
        "velocidad_media_px": [1, 2, 3, 2, 4, 6],
    })  # fmt: skip
    vel = {"a": np.linspace(0, 10 / 0.9 * 0.9, 101), "b": np.linspace(0, 20, 101)}
    cal = calibrar(iv, vel, ["c1", "c2"])
    assert cal.velocidad_p90_por_encuadre["a"] == pytest.approx(9.0)
    assert cal.velocidad_p90_por_encuadre["b"] == pytest.approx(18.0)
    vnorm = np.array([1, 2, 3, 2, 4, 6]) / np.array([9, 9, 9, 18, 18, 18])
    assert cal.velocidad_norm.p33 == pytest.approx(np.percentile(vnorm, 33))
    assert cal.densidad.p66 == pytest.approx(np.percentile([1, 2, 3, 4, 5, 6], 66))
    assert cal.clips == ["c1", "c2"] and cal.n_intervalos == 6


def test_calibrar_falla_sin_velocidades_de_un_encuadre():
    iv = pd.DataFrame({"encuadre": ["a", "z"], "densidad_media": [1, 2],
                       "velocidad_media_px": [1, 2]})  # fmt: skip
    with pytest.raises(ValueError, match="encuadres"):
        calibrar(iv, {"a": np.ones(10)})


def test_aplicar():
    iv = pd.DataFrame({"densidad_media": [1.0, 8.0], "velocidad_media_px": [1.8, 0.2],
                       "congestion": [None, None]})  # fmt: skip
    out = aplicar(iv, "e", CAL)
    assert list(out["velocidad_norm"]) == pytest.approx([0.9, 0.1])
    assert list(out["congestion"]) == ["LOW", "HIGH"]
    assert aplicar(iv, "otro", CAL)["congestion"].isna().all()  # encuadre sin calibrar


def test_comando_calibrate_de_punta_a_punta(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    linea = {"id": "l", "puntos": [[0, 50], [100, 50]], "sentido_positivo": "entrante"}
    roi = [[0, 0], [100, 0], [100, 100]]
    geo = {"encuadre": "e", "resolucion": [100, 100], "roi": roi, "lineas": [linea]}
    (tmp_path / "geo.yaml").write_text(yaml.safe_dump(geo), encoding="utf-8")
    clips = {"clips": [
        {"id": cid, "archivo": f"{cid}.mp4", "fecha": "2026-11-04", "hora_inicio": "12:00",
         "dia_semana": "miercoles", "tipo_dia": "habil", "clima": "x", "geometria": "geo.yaml"}
        for cid in ("fluido", "cola")
    ]}  # fmt: skip
    (tmp_path / "clips.yaml").write_text(yaml.safe_dump(clips), encoding="utf-8")
    (tmp_path / "default.yaml").write_text(yaml.safe_dump({"rutas": {
        "clips": "clips.yaml", "calibracion": "calibracion.yaml", "outputs": "out"}}),
        encoding="utf-8")  # fmt: skip
    datos = {"fluido": ([2.0, 3.0], [2.0, 1.8]), "cola": ([12.0, 14.0], [0.2, 0.3])}
    for cid, (dens, vel) in datos.items():
        d = tmp_path / "out" / "runs" / cid
        d.mkdir(parents=True)
        pd.DataFrame({"clip_id": cid, "t_inicio_s": [0, 30], "t_fin_s": [30, 60],
                      "densidad_media": dens, "velocidad_media_px": vel,
                      "congestion": [None, None]}).to_parquet(d / "intervalos.parquet")  # fmt: skip
        pd.DataFrame({"vel": np.linspace(0, 2.5, 50)}).to_parquet(d / "velocidades_roi.parquet")
    res = CliRunner().invoke(app, ["calibrate", "--config", "default.yaml"])
    assert res.exit_code == 0, res.output
    cal = load_calibracion(tmp_path / "calibracion.yaml")
    assert cal.clips == ["cola", "fluido"] and cal.n_intervalos == 4
    cola = pd.read_parquet(tmp_path / "out/runs/cola/intervalos.parquet")
    fluido = pd.read_parquet(tmp_path / "out/runs/fluido/intervalos.parquet")
    assert set(cola["congestion"]) == {"HIGH"}
    assert set(fluido["congestion"]) == {"LOW"}


@pytest.mark.parametrize("con_grupo", [True, False])
def test_grupo_de_normalizacion(tmp_path: Path, monkeypatch, con_grupo):
    monkeypatch.chdir(tmp_path)
    linea = {"id": "l", "puntos": [[0, 50], [100, 50]], "sentido_positivo": "entrante"}
    for enc in ("a", "b"):
        geo = {"encuadre": enc, "resolucion": [100, 100], "roi": [[0, 0], [100, 0], [100, 100]],
               "lineas": [linea]}  # fmt: skip
        if con_grupo:
            geo["grupo_normalizacion"] = "balcon"
        Path(f"geo_{enc}.yaml").write_text(yaml.safe_dump(geo), encoding="utf-8")
    clips = {"clips": [
        {"id": f"c{enc}", "archivo": f"{enc}.mp4", "fecha": "2026-11-04", "hora_inicio": "12:00",
         "dia_semana": "miercoles", "tipo_dia": "habil", "clima": "x",
         "geometria": f"geo_{enc}.yaml"} for enc in ("a", "b")
    ]}  # fmt: skip
    Path("clips.yaml").write_text(yaml.safe_dump(clips), encoding="utf-8")
    Path("default.yaml").write_text(yaml.safe_dump({"rutas": {
        "clips": "clips.yaml", "calibracion": "cal.yaml", "outputs": "out"}}),
        encoding="utf-8")  # fmt: skip
    # el encuadre "b" solo tiene una cola lenta: su flujo libre propio sería engañoso
    for enc, vmax in (("a", 3.0), ("b", 0.5)):
        d = tmp_path / "out" / "runs" / f"c{enc}"
        d.mkdir(parents=True)
        pd.DataFrame({"clip_id": f"c{enc}", "t_inicio_s": [0], "t_fin_s": [30],
                      "densidad_media": [5.0], "velocidad_media_px": [vmax / 2],
                      "congestion": [None]}).to_parquet(d / "intervalos.parquet")  # fmt: skip
        pd.DataFrame({"vel": np.linspace(0, vmax, 50)}).to_parquet(d / "velocidades_roi.parquet")
    res = CliRunner().invoke(app, ["calibrate", "--config", "default.yaml"])
    assert res.exit_code == 0, res.output
    cal = load_calibracion(tmp_path / "cal.yaml")
    p90 = cal.velocidad_p90_por_encuadre
    if con_grupo:
        assert p90["a"] == p90["b"]  # comparten la referencia de flujo libre
        assert cal.grupos == {"a": "balcon", "b": "balcon"}
    else:
        assert p90["a"] > p90["b"]  # cada encuadre con la suya (brief)
        assert cal.grupos == {"a": "a", "b": "b"}
