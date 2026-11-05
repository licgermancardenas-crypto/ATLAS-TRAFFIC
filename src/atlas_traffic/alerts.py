"""Regla de alerta: circulación en sentido contrario -> eventos.parquet (brief §4d).

Una sola regla, a propósito. Cada línea declara `sentido_permitido` (entrante / saliente
/ ambos). Un track que cruza una línea en el sentido no permitido genera un evento.

Se apoya en los mismos cruces que el conteo (cruces_con_indices), con tres filtros
antifalso-positivo; preferimos perder un evento real antes que mostrar uno falso:
1. Historia mínima: tracks con menos de `min_frames_historia` detecciones se ignoran
   (ruido del tracker, IDs que nacen y mueren).
2. Desplazamiento mínimo: si el desplazamiento neto del track (primer a último punto,
   coordenadas de referencia) es menor a `min_desplazamiento_px`, se ignora (vehículo
   detenido con temblor de caja).
3. Cruce limpio: al menos `frames_limpios` detecciones consecutivas del lado de origen
   antes del cruce y del lado de destino después, todas fuera de la banda muerta.

Nota: `sentido_positivo` (convención geométrica para nombrar cruces) y
`sentido_permitido` (regla de tránsito) son cosas distintas; el brief usaba uno solo.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atlas_traffic.config import AlertasCfg, Geometria, Linea
from atlas_traffic.counting import OPUESTO, cruces_con_indices, distancia_a_linea

COLUMNAS_EVENTOS = [
    "clip_id", "track_id", "ts", "frame", "tipo", "aproximacion", "sentido", "clase",
    "clase_detalle", "thumb",
]  # fmt: skip


def cruce_limpio(dist: np.ndarray, j: int, i: int, n: int, margen_px: float) -> bool:
    """¿Hay `n` puntos seguidos del lado de origen hasta j, y `n` del de destino desde i?"""
    if j + 1 < n or len(dist) - i < n:
        return False
    antes, despues = dist[j - n + 1 : j + 1], dist[i : i + n]
    s_antes, s_despues = np.sign(dist[j]), np.sign(dist[i])
    return bool(
        np.all(np.abs(antes) > margen_px) and np.all(np.sign(antes) == s_antes)
        and np.all(np.abs(despues) > margen_px) and np.all(np.sign(despues) == s_despues)
    )  # fmt: skip


def detectar_sentido_contrario(
    det: pd.DataFrame, geo: Geometria, cfg: AlertasCfg, clip_id: str, margen_px: float = 4.0
) -> pd.DataFrame:
    """Eventos de sentido contrario (sin recorte todavía: ver guardar_evidencia)."""
    lineas = [ln for ln in geo.lineas if ln.sentido_permitido != "ambos"]
    if det.empty or not lineas:
        return pd.DataFrame(columns=COLUMNAS_EVENTOS)
    filas = []
    for tid, g in det.sort_values(["track_id", "frame"]).groupby("track_id"):
        if g["clase"].mode().iat[0] != "vehiculo":
            continue  # contramano es una regla de tránsito vehicular
        if len(g) < cfg.min_frames_historia:
            continue
        xy = g[["cx_ref", "cy_ref"]].to_numpy(dtype=np.float64)
        if np.hypot(*(xy[-1] - xy[0])) < cfg.min_desplazamiento_px:
            continue
        frames, ts = g["frame"].to_numpy(), g["ts"].to_numpy(dtype=np.float64)
        for linea in lineas:
            dist = distancia_a_linea(linea, xy)
            for frame, t, positivo, j, i in cruces_con_indices(frames, ts, xy, linea, margen_px):
                sentido = linea.sentido_positivo if positivo else OPUESTO[linea.sentido_positivo]
                if sentido == linea.sentido_permitido:
                    continue
                if not cruce_limpio(dist, j, i, cfg.frames_limpios, margen_px):
                    continue
                filas.append((
                    clip_id, int(tid), t, frame, "sentido_contrario", linea.id, sentido,
                    g["clase"].mode().iat[0], g["clase_detalle"].mode().iat[0], None,
                ))  # fmt: skip
    return pd.DataFrame(filas, columns=COLUMNAS_EVENTOS)


def guardar_evidencia(
    video: Path,
    eventos: pd.DataFrame,
    det: pd.DataFrame,
    geo: Geometria,
    carpeta: Path,
    homografias: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Recorte por evento: la caja del vehículo en el cruce, con la línea y su flecha.

    La línea está en coordenadas de referencia; se lleva al frame con la inversa de la
    homografía de estabilización. Devuelve `eventos` con la columna `thumb` completa.
    """
    from atlas_traffic.render import dibujar_geometria

    carpeta.mkdir(parents=True, exist_ok=True)
    for viejo in carpeta.glob("*.jpg"):
        viejo.unlink()
    if eventos.empty:
        return eventos
    eventos = eventos.copy()
    lineas = {ln.id: ln for ln in geo.lineas}
    cap = cv2.VideoCapture(str(video))
    try:
        for k, ev in eventos.iterrows():
            g = det[det["track_id"] == ev["track_id"]]
            d = g.iloc[(g["frame"] - ev["frame"]).abs().argsort().iloc[0]]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(d["frame"]))
            ok, frame = cap.read()
            if not ok:
                continue
            H = None
            if homografias:
                cercano = min(homografias, key=lambda f: abs(f - int(d["frame"])))
                H = np.linalg.inv(homografias[cercano])
            linea: Linea = lineas[ev["aproximacion"]]
            frame = dibujar_geometria(frame, [], [linea], H=H)
            x1, y1, x2, y2 = (int(d[c]) for c in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            w, h = x2 - x1, y2 - y1
            cx1, cy1 = max(0, x1 - int(1.5 * w)), max(0, y1 - int(1.5 * h))
            recorte = frame[cy1 : y2 + int(1.5 * h), cx1 : x2 + int(1.5 * w)]
            nombre = f"{ev['track_id']}_{ev['aproximacion']}.jpg"
            cv2.imwrite(str(carpeta / nombre), recorte)
            eventos.at[k, "thumb"] = f"{carpeta.name}/{nombre}"
    finally:
        cap.release()
    return eventos
