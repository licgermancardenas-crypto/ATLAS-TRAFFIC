"""Video anotado: cajas, IDs, ROI, líneas de conteo y contador por aproximación.

Fase 1: render de debug de cajas + track_id para verificar estabilidad de IDs.
Fase 1b: ROI y líneas de conteo, con la flecha del cruce positivo. En el video de debug
la geometría (definida en el frame de referencia) se lleva a cada frame con la inversa
de la homografía de estabilización: si queda pegada a la calle, todo encaja.
Fase 5 completa el overlay (contadores por aproximación).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atlas_traffic.config import Geometria, Linea

COLOR_CLASE = {"vehiculo": (0, 200, 255), "peaton": (255, 120, 0)}  # BGR
COLOR_ROI = (80, 220, 80)
COLOR_LINEA = {"entrante": (60, 60, 255), "saliente": (255, 160, 40)}
COLOR_PENDIENTE = (0, 255, 255)
COLUMNAS_H = [f"h{i}{j}" for i in range(3) for j in range(3)]


def _color_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def _texto(img: np.ndarray, texto: str, org: tuple[int, int], color, escala: float = 0.45) -> None:
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


def _transformar(H: np.ndarray | None, puntos) -> np.ndarray:
    p = np.asarray(puntos, dtype=np.float64).reshape(-1, 2)
    if H is None or len(p) == 0:
        return p
    return cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)


def dibujar_cajas(frame: np.ndarray, dets: pd.DataFrame) -> np.ndarray:
    out = frame.copy()
    for r in dets.itertuples(index=False):
        color = _color_id(int(r.track_id))
        cv2.rectangle(out, (r.x1, r.y1), (r.x2, r.y2), color, 2)
        etiqueta = f"{r.track_id} {r.clase_detalle}"
        cv2.putText(out, etiqueta, (r.x1, max(10, r.y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)  # fmt: skip
    return out


def _flecha_normal(img, a: np.ndarray, b: np.ndarray, H: np.ndarray | None, color) -> None:
    """Flecha desde el punto medio hacia el lado positivo (ver convención en geometry.py)."""
    largo = max(18.0, 0.25 * float(np.hypot(*(b - a))))
    n = np.array([-(b[1] - a[1]), b[0] - a[0]], dtype=np.float64)
    n /= np.linalg.norm(n)
    medio = (a + b) / 2
    p0, p1 = _transformar(H, [medio, medio + largo * n])
    cv2.arrowedLine(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                    color, 2, cv2.LINE_AA, tipLength=0.35)  # fmt: skip


def dibujar_geometria(
    frame: np.ndarray,
    roi: Sequence[tuple[int, int]],
    lineas: Sequence[Linea],
    pendiente: Sequence[tuple[int, int]] = (),
    roi_cerrada: bool = True,
    H: np.ndarray | None = None,
) -> np.ndarray:
    """Dibuja ROI, líneas (con id, sentido y flecha de cruce positivo) y la línea en curso.

    `H` lleva coordenadas de referencia al frame que se dibuja (None = son iguales).
    """
    out = frame.copy()
    if len(roi) >= 3 and roi_cerrada:
        poligono = np.round(_transformar(H, roi)).astype(np.int32)
        capa = out.copy()
        cv2.fillPoly(capa, [poligono], COLOR_ROI)
        out = cv2.addWeighted(capa, 0.18, out, 0.82, 0)
        cv2.polylines(out, [poligono], True, COLOR_ROI, 2, cv2.LINE_AA)
    elif roi:
        pts = np.round(_transformar(H, roi)).astype(np.int32)
        cv2.polylines(out, [pts], False, COLOR_ROI, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, tuple(p), 4, COLOR_ROI, -1)

    for linea in lineas:
        a, b = (np.asarray(p, dtype=np.float64) for p in linea.puntos)
        color = COLOR_LINEA[linea.sentido_positivo]
        pa, pb = np.round(_transformar(H, [a, b])).astype(int)
        cv2.line(out, tuple(pa), tuple(pb), color, 3, cv2.LINE_AA)
        cv2.circle(out, tuple(pa), 4, color, -1)
        _flecha_normal(out, a, b, H, color)
        _texto(out, f"{linea.id} ({linea.sentido_positivo})", (int(pa[0]), int(pa[1]) - 6), color)

    if pendiente:
        pts = [np.asarray(p, dtype=np.float64) for p in pendiente]
        for p in np.round(_transformar(H, pts)).astype(int):
            cv2.circle(out, tuple(p), 5, COLOR_PENDIENTE, -1)
        if len(pts) == 2:
            pa, pb = np.round(_transformar(H, pts)).astype(int)
            cv2.line(out, tuple(pa), tuple(pb), COLOR_PENDIENTE, 2, cv2.LINE_AA)
            _flecha_normal(out, pts[0], pts[1], H, COLOR_PENDIENTE)
    return out


def dibujar_contadores(
    frame: np.ndarray, lineas: Sequence[Linea], conteos: dict[tuple[str, str], int]
) -> np.ndarray:
    """Panel arriba a la izquierda: cruces acumulados por línea (entrantes / salientes)."""
    out = frame.copy()
    alto_fila = 18
    ancho = 210
    capa = out.copy()
    cv2.rectangle(capa, (4, 4), (4 + ancho, 10 + alto_fila * (len(lineas) + 1)), (0, 0, 0), -1)
    out = cv2.addWeighted(capa, 0.55, out, 0.45, 0)
    _texto(out, "linea      entra  sale", (10, 4 + alto_fila), (255, 255, 255), 0.42)
    for k, linea in enumerate(lineas, start=2):
        e = conteos.get((linea.id, "entrante"), 0)
        s = conteos.get((linea.id, "saliente"), 0)
        _texto(out, f"{linea.id[:10]:<10} {e:>5} {s:>5}", (10, 4 + alto_fila * k),
               COLOR_LINEA[linea.sentido_positivo], 0.42)  # fmt: skip
    return out


def homografias_por_frame(estabilizacion: pd.DataFrame) -> dict[int, np.ndarray]:
    """frame -> H (frame -> referencia), desde estabilizacion.parquet."""
    H = estabilizacion[COLUMNAS_H].to_numpy().reshape(-1, 3, 3)
    return dict(zip(estabilizacion["frame"].astype(int), H, strict=True))


def video_debug(
    video: Path,
    detecciones: pd.DataFrame,
    salida: Path,
    geometria: Geometria | None = None,
    estabilizacion: pd.DataFrame | None = None,
    cruces: pd.DataFrame | None = None,
) -> Path:
    """mp4 con cajas e IDs; si hay geometría, ROI y líneas siguiendo a la cámara.

    Con `cruces`, suma un contador acumulado por línea y sentido: sirve para el chequeo
    manual de la Fase 2 (pausar el video y comparar con lo que uno ve).
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ultimo = int(detecciones["frame"].max()) if not detecciones.empty else -1
    por_frame = dict(tuple(detecciones.groupby("frame")))
    hs = homografias_por_frame(estabilizacion) if estabilizacion is not None else {}
    eventos = (
        list(cruces[["frame", "aproximacion", "sentido"]].itertuples(index=False, name=None))
        if cruces is not None and not cruces.empty
        else []
    )
    eventos.sort()
    conteos: dict[tuple[str, str], int] = {}
    writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        vigentes = detecciones.iloc[0:0]
        H_inv = None
        for i in range(ultimo + 1):
            ok, frame = cap.read()
            if not ok:
                break
            # con paso_frames > 1 hay frames sin inferencia: se mantienen las últimas cajas
            vigentes = por_frame.get(i, vigentes)
            if i in hs:
                H_inv = np.linalg.inv(hs[i])  # referencia -> este frame
            if geometria is not None:
                frame = dibujar_geometria(frame, geometria.roi, geometria.lineas, H=H_inv)
            frame = dibujar_cajas(frame, vigentes)
            while eventos and eventos[0][0] <= i:
                _, linea_id, sentido = eventos.pop(0)
                conteos[(linea_id, sentido)] = conteos.get((linea_id, sentido), 0) + 1
            if geometria is not None and cruces is not None:
                frame = dibujar_contadores(frame, geometria.lineas, conteos)
            writer.write(frame)
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
    hs = homografias_por_frame(estabilizacion)
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
