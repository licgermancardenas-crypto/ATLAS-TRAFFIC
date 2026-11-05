"""Línea virtual, detección de cruce, sentido y conteo único por aproximación (brief §3b).

Reglas:
- Todo se mide en coordenadas de referencia (cx_ref, cy_ref): la cámara puede moverse.
- Tracks cuyo centroide nunca entra a la ROI se descartan (tránsito del fondo).
- Lado de un punto respecto de una línea: signo de lado() (ver geometry.py). Hay una
  banda muerta de ±margen_px alrededor de la línea donde el lado no cambia: un vehículo
  detenido sobre la línea tiembla un par de píxeles y sin banda contaría cruces falsos.
- Cruce = el lado pasa de un signo al otro y el tramo recorrido corta el segmento A-B
  (no la recta infinita: un auto que pasa lejos del segmento no cruza esa aproximación).
- Sentido: de lado negativo a positivo = sentido_positivo de la línea; al revés, el opuesto.
- Conteo único: un track cuenta como mucho una vez por línea y sentido. Si cruza y
  vuelve cuenta en ambos sentidos. Si el ID se rompe y reaparece, el track nuevo puede
  volver a contar: ese es el error residual que mide el chequeo manual de la Fase 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas_traffic.config import Geometria, Linea
from atlas_traffic.geometry import dentro_roi, lado

OPUESTO = {"entrante": "saliente", "saliente": "entrante"}
COLUMNAS_CRUCES = [
    "track_id", "clase", "clase_detalle", "aproximacion", "sentido", "positivo",
    "frame", "ts_cruce",
]  # fmt: skip


def _segmentos_se_cortan(p0, p1, a, b, tol: float = 1e-9) -> bool:
    """¿El tramo p0-p1 corta al segmento a-b? (incluye tocar el borde)."""

    def orient(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1, d2 = orient(a, b, p0), orient(a, b, p1)
    d3, d4 = orient(p0, p1, a), orient(p0, p1, b)
    return (d1 * d2 <= tol) and (d3 * d4 <= tol)


def distancia_a_linea(linea: Linea, xy: np.ndarray) -> np.ndarray:
    """Distancia con signo (px) de cada punto a la recta de la línea (signo = lado)."""
    a = np.asarray(linea.puntos[0], dtype=np.float64)
    b = np.asarray(linea.puntos[1], dtype=np.float64)
    return lado(linea, xy) / np.hypot(*(b - a))


def cruces_de_track(
    frames: np.ndarray, ts: np.ndarray, xy: np.ndarray, linea: Linea, margen_px: float = 4.0
) -> list[tuple[int, float, bool]]:
    """Cruces de un track sobre una línea: lista de (frame, ts interpolado, positivo).

    `frames`, `ts` y `xy` (N, 2) ordenados por frame. Devuelve todos los cruces, sin
    aplicar todavía la regla de conteo único.
    """
    return [c[:3] for c in cruces_con_indices(frames, ts, xy, linea, margen_px)]


def cruces_con_indices(
    frames: np.ndarray, ts: np.ndarray, xy: np.ndarray, linea: Linea, margen_px: float = 4.0
) -> list[tuple[int, float, bool, int, int]]:
    """Como cruces_de_track, más (j, i): último punto antes y primer punto después."""
    if len(frames) < 2:
        return []
    a = np.asarray(linea.puntos[0], dtype=np.float64)
    b = np.asarray(linea.puntos[1], dtype=np.float64)
    dist = distancia_a_linea(linea, xy)
    cruces: list[tuple[int, float, bool, int, int]] = []
    estado, idx_estado = 0, -1  # lado vigente (fuera de la banda muerta) y su último punto
    # Un track que aparece ya dentro de la banda muerta (detectado tarde, justo sobre la
    # línea) toma como lado inicial el de su primer punto: si después sale por el otro
    # lado, cruzó. Sin esto esos vehículos no contaban.
    if dist[0] != 0:
        estado, idx_estado = (1 if dist[0] > 0 else -1), 0
    for i, d in enumerate(dist):
        signo = 1 if d > margen_px else -1 if d < -margen_px else 0
        if signo == 0:
            continue
        if estado != 0 and signo != estado:
            j = idx_estado
            if _segmentos_se_cortan(xy[j], xy[i], a, b):
                # instante del cruce: interpolado donde la distancia pasa por cero
                t = dist[j] / (dist[j] - dist[i])
                frame = int(round(frames[j] + t * (frames[i] - frames[j])))
                cruces.append((frame, float(ts[j] + t * (ts[i] - ts[j])), signo > 0, j, i))
        estado, idx_estado = signo, i
    return cruces


def contar(det: pd.DataFrame, geo: Geometria, margen_px: float = 4.0) -> pd.DataFrame:
    """Un registro por cruce contado (ya aplicada la regla de conteo único)."""
    if det.empty:
        return pd.DataFrame(columns=COLUMNAS_CRUCES)
    det = det.sort_values(["track_id", "frame"])
    xy_todo = det[["cx_ref", "cy_ref"]].to_numpy(dtype=np.float64)
    en_roi = pd.Series(dentro_roi(geo.roi, xy_todo), index=det.index)
    tracks_validos = en_roi.groupby(det["track_id"]).any()
    filas: list[tuple] = []
    for tid, g in det[det["track_id"].isin(tracks_validos[tracks_validos].index)].groupby(
        "track_id"
    ):
        frames = g["frame"].to_numpy()
        ts = g["ts"].to_numpy(dtype=np.float64)
        xy = g[["cx_ref", "cy_ref"]].to_numpy(dtype=np.float64)
        clase = g["clase"].mode().iat[0]
        detalle = g["clase_detalle"].mode().iat[0]
        for linea in geo.lineas:
            if clase not in linea.clases:
                continue  # ej.: un peatón que cruza una línea de calzada no es un vehículo
            vistos: set[bool] = set()
            for frame, t, positivo in cruces_de_track(frames, ts, xy, linea, margen_px):
                if positivo in vistos:
                    continue  # una sola vez por línea y sentido
                vistos.add(positivo)
                sentido = linea.sentido_positivo if positivo else OPUESTO[linea.sentido_positivo]
                filas.append((int(tid), clase, detalle, linea.id, sentido, positivo, frame, t))
    cruces = pd.DataFrame(filas, columns=COLUMNAS_CRUCES)
    return cruces.sort_values(["frame", "track_id"]).reset_index(drop=True)


def resumen_conteo(cruces: pd.DataFrame, geo: Geometria) -> pd.DataFrame:
    """Tabla aproximación x sentido x clase (incluye ceros para todas las líneas)."""
    indice = pd.MultiIndex.from_product(
        [[linea.id for linea in geo.lineas], ["entrante", "saliente"]],
        names=["aproximacion", "sentido"],
    )
    tabla = (
        cruces.groupby(["aproximacion", "sentido", "clase"]).size().unstack("clase", fill_value=0)
        if not cruces.empty
        else pd.DataFrame(index=indice)
    )
    tabla = tabla.reindex(indice, fill_value=0)
    for clase in ("vehiculo", "peaton"):
        if clase not in tabla.columns:
            tabla[clase] = 0
    return tabla[["vehiculo", "peaton"]].astype(int)


def anotar_tracks(tracks: pd.DataFrame, cruces: pd.DataFrame) -> pd.DataFrame:
    """Agrega a tracks las columnas del brief §3.2: cruzo_linea, aproximacion, sentido, ts_cruce.

    Un vehículo que atraviesa el cruce suele pasar dos líneas (entra por una, sale por
    otra). tracks guarda el primer cruce y `n_cruces`; el detalle completo está en
    cruces.parquet, que es la fuente de los conteos.
    """
    tracks = tracks.drop(
        columns=["cruzo_linea", "aproximacion", "sentido", "ts_cruce", "n_cruces"],
        errors="ignore",
    )
    if cruces.empty:
        primero = pd.DataFrame(columns=["track_id", "aproximacion", "sentido", "ts_cruce"])
        n = pd.Series(dtype=int, name="n_cruces")
    else:
        orden = cruces.sort_values("ts_cruce")
        primero = orden.groupby("track_id").first()[["aproximacion", "sentido", "ts_cruce"]]
        primero = primero.reset_index()
        n = orden.groupby("track_id").size().rename("n_cruces")
    out = tracks.merge(primero, on="track_id", how="left")
    out = out.merge(n, left_on="track_id", right_index=True, how="left")
    out["n_cruces"] = out["n_cruces"].fillna(0).astype(int)
    out["cruzo_linea"] = out["n_cruces"] > 0
    return out
