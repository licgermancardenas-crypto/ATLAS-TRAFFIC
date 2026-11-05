"""Orquestación de una corrida sobre un clip.

Fase 1: video -> detector (protocolo) -> detecciones.parquet (una fila por detección y
frame) -> tracks.parquet (una fila por objeto). Las fases siguientes agregan columnas a
tracks (cruce, sentido, color) y generan intervalos y eventos a partir de las mismas
detecciones crudas.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atlas_traffic.config import AppConfig, Clip
from atlas_traffic.detect import Detector

COLUMNAS_DETECCIONES = [
    "frame", "ts", "track_id", "clase", "clase_detalle", "confianza",
    "x1", "y1", "x2", "y2", "cx", "cy",
]  # fmt: skip


@dataclass(frozen=True)
class InfoVideo:
    fps: float
    n_frames: int
    ancho: int
    alto: int


def info_video(ruta: Path) -> InfoVideo:
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f"no se pudo abrir el video: {ruta}")
    try:
        return InfoVideo(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            ancho=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            alto=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def detectar_video(
    ruta: Path,
    detector: Detector,
    paso_frames: int = 1,
    max_segundos: float | None = None,
    progreso: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Corre el detector sobre el video y devuelve las detecciones con track_id."""
    info = info_video(ruta)
    limite = info.n_frames
    if max_segundos is not None:
        limite = min(limite, int(max_segundos * info.fps))
    detector.reset()
    filas: list[tuple] = []
    cap = cv2.VideoCapture(str(ruta))
    try:
        frame_idx = 0
        while frame_idx < limite:
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % paso_frames == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                for d in detector.track(frame):
                    if d.track_id is None:
                        continue  # sin ID no hay objeto único que contar
                    x1, y1, x2, y2 = d.bbox
                    cx, cy = d.centroide
                    filas.append((
                        frame_idx, frame_idx / info.fps, d.track_id, d.clase, d.clase_detalle,
                        d.confianza, x1, y1, x2, y2, cx, cy,
                    ))  # fmt: skip
                if progreso:
                    progreso(frame_idx, limite)
            frame_idx += 1
    finally:
        cap.release()
    df = pd.DataFrame(filas, columns=COLUMNAS_DETECCIONES)
    return df.astype({"frame": "int32", "track_id": "int32", "x1": "int32", "y1": "int32",
                      "x2": "int32", "y2": "int32"})  # fmt: skip


def tracks_desde_detecciones(det: pd.DataFrame) -> pd.DataFrame:
    """Una fila por track_id. Clase por mayoría; velocidad = recorrido / frames abarcados."""
    columnas = ["track_id", "clase", "clase_detalle", "frame_inicio", "frame_fin",
                "n_detecciones", "confianza_media", "velocidad_px"]  # fmt: skip
    if det.empty:
        return pd.DataFrame(columns=columnas)
    det = det.sort_values(["track_id", "frame"])
    paso = det.groupby("track_id")[["cx", "cy"]].diff()
    det = det.assign(_desp=np.hypot(paso["cx"], paso["cy"]).fillna(0.0))
    g = det.groupby("track_id")
    tracks = pd.DataFrame({
        "clase": g["clase"].agg(lambda s: s.mode().iat[0]),
        "clase_detalle": g["clase_detalle"].agg(lambda s: s.mode().iat[0]),
        "frame_inicio": g["frame"].min(),
        "frame_fin": g["frame"].max(),
        "n_detecciones": g.size(),
        "confianza_media": g["confianza"].mean(),
        "recorrido": g["_desp"].sum(),
    })  # fmt: skip
    abarcados = (tracks["frame_fin"] - tracks["frame_inicio"]).clip(lower=1)
    tracks["velocidad_px"] = tracks.pop("recorrido") / abarcados
    return tracks.reset_index()[columnas]


def correr_clip(
    clip: Clip,
    cfg: AppConfig,
    detector: Detector,
    max_segundos: float | None = None,
    progreso: Callable[[int, int], None] | None = None,
) -> Path:
    """Procesa un clip y escribe outputs/runs/<clip_id>/. Devuelve la carpeta de salida."""
    salida = cfg.rutas.outputs / "runs" / clip.id
    salida.mkdir(parents=True, exist_ok=True)
    det = detectar_video(clip.archivo, detector, cfg.video.paso_frames, max_segundos, progreso)
    det.to_parquet(salida / "detecciones.parquet", index=False)
    tracks_desde_detecciones(det).to_parquet(salida / "tracks.parquet", index=False)
    return salida
