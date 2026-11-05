"""Fase 6: batch de punta a punta sobre un dataset sintético (sin YOLO).

Dos clips que comparten encuadre: uno fluido (pocos autos rápidos que cruzan la línea) y
uno denso y lento (muchos autos casi quietos). batch tiene que procesar ambos, calibrar,
clasificar el denso como HIGH y el fluido como LOW, y consolidar summary, tracks y eventos.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from typer.testing import CliRunner

from atlas_traffic.cli import app
from atlas_traffic.config import load_app_config
from atlas_traffic.detect import Deteccion, FakeDetector
from atlas_traffic.pipeline import correr_clip

FPS, N = 10, 100  # 10 s de video


def _video(ruta: Path) -> None:
    w = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (200, 200))
    for _ in range(N):
        w.write(np.full((200, 200, 3), 90, np.uint8))
    w.release()


def _guion_fluido():
    # 3 autos que bajan rápido (4 px/frame) y cruzan la línea y=100, en momentos distintos
    frames = [[] for _ in range(N)]
    for k, t0 in enumerate((0, 30, 60)):
        for f in range(t0, min(N, t0 + 40)):
            y = 20 + 4 * (f - t0)
            frames[f].append(Deteccion((90, y - 5, 110, y + 5), "vehiculo", 0.9, k + 1, "car"))
    return frames


def _guion_denso():
    # 12 autos casi quietos (0.05 px/frame) todo el clip
    frames = [[] for _ in range(N)]
    for k in range(12):
        x, y0 = 15 + 15 * (k % 6), 40 + 60 * (k // 6)
        for f in range(N):
            y = int(y0 + 0.05 * f)
            frames[f].append(Deteccion((x, y, x + 10, y + 8), "vehiculo", 0.9, k + 1, "car"))
    return frames


def test_batch_de_punta_a_punta(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    linea = {"id": "av", "puntos": [[10, 100], [190, 100]], "sentido_positivo": "entrante",
             "sentido_permitido": "entrante"}  # fmt: skip
    geo = {"encuadre": "enc", "resolucion": [200, 200],
           "roi": [[0, 0], [200, 0], [200, 200], [0, 200]], "lineas": [linea]}  # fmt: skip
    Path("geo.yaml").write_text(yaml.safe_dump(geo), encoding="utf-8")
    meta = {"fecha": "2026-11-04", "dia_semana": "miercoles", "tipo_dia": "habil",
            "clima": "nublado", "geometria": "geo.yaml"}  # fmt: skip
    clips = {"clips": [
        {"id": "fluido", "archivo": "fluido.mp4", "hora_inicio": "08:00", **meta},
        {"id": "denso", "archivo": "denso.mp4", "hora_inicio": "18:30", **meta},
    ]}  # fmt: skip
    Path("clips.yaml").write_text(yaml.safe_dump(clips), encoding="utf-8")
    config = {
        "video": {"paso_frames": 1},
        "estabilizacion": {"activa": False},  # video sintético sin textura
        "rutas": {"clips": "clips.yaml", "calibracion": "calibracion.yaml", "outputs": "out"},
    }
    Path("default.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = load_app_config(Path("default.yaml"))
    assert cfg.video.paso_frames == 1

    # detección "real" con el detector falso; batch después reanaliza todo
    from atlas_traffic.clips import load_clips

    for c, guion in zip(load_clips(Path("clips.yaml")), (_guion_fluido(), _guion_denso()),
                        strict=True):  # fmt: skip
        _video(c.archivo)
        correr_clip(c, cfg, FakeDetector(guion))

    res = CliRunner().invoke(app, ["batch", "--config", "default.yaml", "--reusar-detecciones"])
    assert res.exit_code == 0, res.output

    summary = pd.read_parquet("out/summary.parquet").set_index("clip_id")
    assert set(summary.index) == {"fluido", "denso"}
    assert summary.loc["denso", "congestion"] == "HIGH"
    assert summary.loc["fluido", "congestion"] == "LOW"
    assert summary.loc["denso", "hora_inicio"] == "18:30"
    assert summary.loc["denso", "inicio"] == pd.Timestamp("2026-11-04 18:30")
    assert summary.loc["fluido", "hora_del_dia"] == 8
    assert summary.loc["fluido", "encuadre"] == "enc"

    tracks = pd.read_parquet("out/tracks_todos.parquet")
    cruzaron = tracks[(tracks.clip_id == "fluido") & tracks.cruzo_linea]
    assert len(cruzaron) == 3
    assert (cruzaron["hora_cruce"].dt.hour == 8).all()
    assert all(t.startswith("runs/fluido/thumbs/") for t in cruzaron["thumb"])
    assert Path("out", cruzaron["thumb"].iat[0]).exists()
    assert pd.read_parquet("out/eventos_todos.parquet").empty  # nadie a contramano
    assert Path("calibracion.yaml").exists()


def test_batch_sigue_si_falta_un_video_y_termina_con_error(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    clips = {"clips": [{"id": "x", "archivo": "no_existe.mp4", "fecha": "2026-11-04",
                        "hora_inicio": "08:00", "dia_semana": "miercoles", "tipo_dia": "habil",
                        "clima": "x", "geometria": "geo.yaml"}]}  # fmt: skip
    Path("clips.yaml").write_text(yaml.safe_dump(clips), encoding="utf-8")
    Path("default.yaml").write_text(yaml.safe_dump({
        "deteccion": {"detector": "fake"},
        "rutas": {"clips": "clips.yaml", "outputs": "out"}}), encoding="utf-8")  # fmt: skip
    res = CliRunner().invoke(app, ["batch", "--config", "default.yaml"])
    assert res.exit_code == 1
    assert "no existe el video" in res.output
