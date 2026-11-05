"""Video anotado: cajas, IDs, ROI, líneas de conteo y contador por aproximación.

Fase 1: render de debug de cajas + track_id para verificar estabilidad de IDs.
Fase 5 completa el overlay (ROI, líneas, contadores).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

COLOR_CLASE = {"vehiculo": (0, 200, 255), "peaton": (255, 120, 0)}  # BGR


def _color_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def dibujar_cajas(frame: np.ndarray, dets: pd.DataFrame) -> np.ndarray:
    out = frame.copy()
    for r in dets.itertuples(index=False):
        color = _color_id(int(r.track_id))
        cv2.rectangle(out, (r.x1, r.y1), (r.x2, r.y2), color, 2)
        etiqueta = f"{r.track_id} {r.clase_detalle}"
        cv2.putText(out, etiqueta, (r.x1, max(10, r.y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)  # fmt: skip
    return out


def video_debug(video: Path, detecciones: pd.DataFrame, salida: Path) -> Path:
    """Escribe un mp4 con las cajas y los IDs de cada frame procesado."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ultimo = int(detecciones["frame"].max()) if not detecciones.empty else -1
    por_frame = dict(tuple(detecciones.groupby("frame")))
    writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        i = 0
        vigentes = detecciones.iloc[0:0]
        while i <= ultimo:
            ok, frame = cap.read()
            if not ok:
                break
            # con paso_frames > 1 hay frames sin inferencia: se mantienen las últimas cajas
            vigentes = por_frame.get(i, vigentes)
            frame = dibujar_cajas(frame, vigentes)
            writer.write(frame)
            i += 1
    finally:
        cap.release()
        writer.release()
    return salida


def video_estabilizado(video: Path, estabilizacion: pd.DataFrame, salida: Path) -> Path:
    """Escribe un mp4 con cada frame procesado alineado al frame de referencia.

    Sirve para verificar a ojo la estabilización: los edificios deberían quedar quietos.
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hs = {
        int(r.frame): np.array(
            [getattr(r, f"h{i}{j}") for i in range(3) for j in range(3)]
        ).reshape(3, 3)
        for r in estabilizacion.itertuples(index=False)
    }
    ultimo = max(hs) if hs else -1
    writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        H = np.eye(3)
        for i in range(ultimo + 1):
            ok, frame = cap.read()
            if not ok:
                break
            H = hs.get(i, H)
            writer.write(cv2.warpPerspective(frame, H, (w, h)))
    finally:
        cap.release()
        writer.release()
    return salida
