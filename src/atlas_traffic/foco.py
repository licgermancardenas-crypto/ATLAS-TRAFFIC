"""Material para "seguir un vehículo" en el dashboard (se genera junto con el render).

- Video de foco: una cámara virtual que sigue a un solo vehículo (recorte centrado en él,
  suavizado), con su estela y su velocidad en cada momento. Los demás no se marcan.
- Plano: el fondo de la esquina visto desde arriba, en metros (con la calibración del
  suelo), para dibujar encima el recorrido del vehículo.

El dashboard publicado corre sin OpenCV ni los videos originales: todo esto se prepara
acá y se publica como archivos livianos.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atlas_traffic.config import EXTENSION_PLANO, PX_POR_M_PLANO
from atlas_traffic.suelo import fondo_estabilizado

__all__ = ["plano_suelo", "videos_foco"]


def plano_suelo(
    video: Path, homografias: dict[int, np.ndarray], tam: tuple[int, int], H_suelo, salida: Path
) -> Path:
    """Fondo limpio de la esquina llevado al suelo, en metros (ver EXTENSION_PLANO)."""
    fondo = fondo_estabilizado(video, homografias, tam)
    (x0, x1), (y0, y1) = EXTENSION_PLANO["x"], EXTENSION_PLANO["y"]
    k = PX_POR_M_PLANO
    # metros -> píxeles del plano (Y hacia arriba en el plano = hacia el fondo de la calle)
    A = np.array([[k, 0, -x0 * k], [0, -k, y1 * k], [0, 0, 1]], dtype=float)
    img = cv2.warpPerspective(fondo, A @ np.asarray(H_suelo, float),
                              (int((x1 - x0) * k), int((y1 - y0) * k)))  # fmt: skip
    salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(salida), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return salida


def _suavizar(v: np.ndarray, n: int) -> np.ndarray:
    if len(v) < 3:
        return v
    n = min(n, len(v) // 2 * 2 + 1)
    pad = np.pad(v, n // 2, mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="valid")


class _Escritor:
    """mp4 H.264 (reproducible en el navegador) escrito cuadro a cuadro por ffmpeg."""

    def __init__(self, ruta: Path, lado: int, fps: float) -> None:
        cmd = [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", f"{lado}x{lado}", "-r", f"{fps:.3f}", "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "31", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-an", str(ruta)]  # fmt: skip
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def escribir(self, img: np.ndarray) -> None:
        self._p.stdin.write(np.ascontiguousarray(img).tobytes())

    def cerrar(self) -> None:
        self._p.stdin.close()
        self._p.wait()


def _plan_de_track(g: pd.DataFrame, ancho_video: int, margen: int, max_frames: int) -> dict:
    """Cuadros a escribir, centro suavizado del recorte y tamaño del recorte."""
    g = g.sort_values("frame")
    f = g["frame"].to_numpy()
    cx = ((g["x1"] + g["x2"]) / 2).to_numpy(float)
    cy = ((g["y1"] + g["y2"]) / 2).to_numpy(float)
    cuadros = np.arange(max(0, f[0] - margen), f[-1] + margen + 1, 2)
    if len(cuadros) > max_frames:  # vehículos que esperan mucho: alrededor de su cruce
        centro = int(g["frame_cruce"].iat[0]) if "frame_cruce" in g else int(np.median(f))
        ini = max(cuadros[0], centro - int(max_frames * 1.2))
        cuadros = cuadros[(cuadros >= ini)][:max_frames]
    # zoom: el recorte acompaña el tamaño de la caja (suavizado), así el vehículo se ve
    # del mismo tamaño aunque se aleje
    lado_caja = np.maximum(g["x2"] - g["x1"], g["y2"] - g["y1"]).to_numpy(float)
    lados = np.clip(3.0 * _suavizar(np.interp(cuadros, f, lado_caja), 31),
                    0.12 * ancho_video, 0.6 * ancho_video)  # fmt: skip
    return {
        "cuadros": cuadros,
        "cx": _suavizar(np.interp(cuadros, f, cx), 15),
        "cy": _suavizar(np.interp(cuadros, f, cy), 15),
        "lados": lados,
    }


def videos_foco(
    video: Path,
    trayectorias: pd.DataFrame,
    track_ids: list[int],
    salida: Path,
    lado_px: int = 240,
    max_segundos: float = 15.0,
) -> list[Path]:
    """Un mp4 por vehículo de `track_ids`, con la cámara siguiéndolo. Lee el video una vez.

    `trayectorias`: detecciones de vehículos (frame, x1..y2 en el cuadro, vel_kmh opcional).
    """
    salida.mkdir(parents=True, exist_ok=True)
    for viejo in salida.glob("*.mp4"):
        viejo.unlink()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    margen = int(0.4 * fps)
    planes, por_track = {}, dict(tuple(trayectorias.groupby("track_id")))
    for tid in track_ids:
        if tid in por_track and len(por_track[tid]) >= 4:
            planes[tid] = _plan_de_track(por_track[tid], ancho, margen,
                                         int(max_segundos * fps / 2))  # fmt: skip
    if not planes:
        cap.release()
        return []
    # qué vehículos escriben en cada cuadro
    agenda: dict[int, list[int]] = {}
    for tid, plan in planes.items():
        for fr in plan["cuadros"]:
            agenda.setdefault(int(fr), []).append(tid)
    escritores: dict[int, _Escritor] = {}
    hechos: list[Path] = []
    pasos = {tid: 0 for tid in planes}
    ultimo = max(agenda)
    i = 0
    try:
        while i <= ultimo:
            ok, frame = cap.read()
            if not ok:
                break
            for tid in agenda.get(i, []):
                plan, k = planes[tid], pasos[tid]
                if tid not in escritores:
                    escritores[tid] = _Escritor(salida / f"{tid}.mp4", lado_px, fps / 2)
                escritores[tid].escribir(_cuadro_foco(frame, plan, k, por_track[tid], lado_px,
                                                      tid, fps))  # fmt: skip
                pasos[tid] = k + 1
                if k + 1 == len(plan["cuadros"]):
                    escritores.pop(tid).cerrar()
                    hechos.append(salida / f"{tid}.mp4")
            i += 1
    finally:
        cap.release()
        for tid, e in escritores.items():
            e.cerrar()
            hechos.append(salida / f"{tid}.mp4")
    return hechos


def _cuadro_foco(frame, plan, k, g: pd.DataFrame, lado_px: int, tid: int, fps: float):
    """Recorte centrado en el vehículo con su caja, su estela y su velocidad."""
    alto, ancho = frame.shape[:2]
    lado = plan["lados"][k]
    cx, cy = plan["cx"][k], plan["cy"][k]
    x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    n = int(round(lado))
    # recorte con relleno si se sale del cuadro
    recorte = np.zeros((n, n, 3), np.uint8)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(ancho, x0 + n), min(alto, y0 + n)
    if xb > xa and yb > ya:
        recorte[ya - y0 : yb - y0, xa - x0 : xb - x0] = frame[ya:yb, xa:xb]
    esc = lado_px / n
    img = cv2.resize(recorte, (lado_px, lado_px), interpolation=cv2.INTER_AREA
                     if esc < 1 else cv2.INTER_CUBIC)  # fmt: skip
    fr = plan["cuadros"][k]
    # estela: centros de la caja en los últimos ~2 s (coordenadas del cuadro)
    pasado = g[(g["frame"] <= fr) & (g["frame"] >= fr - 2 * fps)].sort_values("frame")
    if len(pasado) >= 2:
        pts = np.c_[(pasado["x1"] + pasado["x2"]) / 2, pasado["y2"]].astype(float)
        pts = ((pts - [x0, y0]) * esc).astype(np.int32)
        cv2.polylines(img, [pts], False, (0, 220, 255), 2, cv2.LINE_AA)
    actual = g.iloc[(g["frame"] - fr).abs().argsort().iloc[0]]
    if abs(actual["frame"] - fr) <= 4:
        p1 = (int((actual["x1"] - x0) * esc), int((actual["y1"] - y0) * esc))
        p2 = (int((actual["x2"] - x0) * esc), int((actual["y2"] - y0) * esc))
        cv2.rectangle(img, p1, p2, (80, 220, 80), 2, cv2.LINE_AA)
    texto = f"#{tid}"
    v = actual.get("vel_kmh", np.nan) if hasattr(actual, "get") else np.nan
    if pd.notna(v):
        texto += f"  {v:.0f} km/h"
    for color, grosor in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, texto, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, grosor,
                    cv2.LINE_AA)  # fmt: skip
        cv2.putText(img, f"{fr / fps:.1f} s", (8, lado_px - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, grosor, cv2.LINE_AA)  # fmt: skip
    return img
