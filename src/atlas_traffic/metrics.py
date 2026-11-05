"""Densidad, velocidad y agregación por intervalo -> intervalos.parquet (brief §3.3).

Por cada ventana de `intervalo_s` segundos:
- flujo_veh_min / flujo_peat_min: objetos únicos que atravesaron el cruce (al menos un
  cruce de línea), asignados a la ventana de su primer cruce, normalizados a por minuto
  con la duración real de la ventana (la última suele ser parcial).
- densidad_media: promedio, sobre los frames procesados de la ventana, de vehículos
  simultáneos con el centroide dentro de la ROI.
- velocidad_media_px: promedio de la velocidad instantánea (px/frame, coordenadas de
  referencia) de los vehículos dentro de la ROI. La velocidad instantánea se mide como
  desplazamiento en una ventana centrada de ±ventana/2 frames, no paso a paso: así el
  temblor de caja y estabilización no infla la velocidad de un auto quieto.
- congestion: la completa la Fase 4 (queda vacía acá).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas_traffic.config import Geometria
from atlas_traffic.geometry import dentro_roi

COLUMNAS_INTERVALOS = [
    "clip_id", "bucket", "t_inicio_s", "t_fin_s", "duracion_s",
    "flujo_veh_min", "flujo_peat_min", "densidad_media", "velocidad_media_px",
    "n_velocidad", "congestion",
]  # fmt: skip


def velocidad_instantanea(det: pd.DataFrame, ventana_frames: int) -> pd.Series:
    """Velocidad (px/frame) de cada detección: |p(t+h) - p(t-h)| / (2h), h = ventana/2.

    p(t) se interpola sobre la trayectoria del track. Cerca de los extremos la ventana
    se recorta; si lo que queda es menor a media ventana, la velocidad es NaN (track
    demasiado corto para medir).
    """
    h = ventana_frames / 2
    valores = np.full(len(det), np.nan)
    # se trabaja por posición: no depende de que el índice de `det` sea único
    for posiciones in det.reset_index(drop=True).groupby("track_id").indices.values():
        g = det.iloc[posiciones]
        orden = np.argsort(g["frame"].to_numpy(), kind="stable")
        posiciones, g = posiciones[orden], g.iloc[orden]
        f = g["frame"].to_numpy(dtype=float)
        x, y = g["cx_ref"].to_numpy(dtype=float), g["cy_ref"].to_numpy(dtype=float)
        t0 = np.clip(f - h, f[0], f[-1])
        t1 = np.clip(f + h, f[0], f[-1])
        lapso = t1 - t0
        dx = np.interp(t1, f, x) - np.interp(t0, f, x)
        dy = np.interp(t1, f, y) - np.interp(t0, f, y)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(lapso >= h, np.hypot(dx, dy) / lapso, np.nan)
        valores[posiciones] = v
    return pd.Series(valores, index=det.index)


def vehiculos_en_roi(det: pd.DataFrame, geo: Geometria, ventana_frames: int) -> pd.DataFrame:
    """Detecciones de vehículos con el centroide dentro de la ROI, con su velocidad (`vel`)."""
    veh = det[det["clase"] == "vehiculo"]
    if veh.empty:
        return veh.assign(vel=pd.Series(dtype=float))
    veh = veh.assign(vel=velocidad_instantanea(veh, ventana_frames))
    return veh[dentro_roi(geo.roi, veh[["cx_ref", "cy_ref"]].to_numpy())]


def calcular_intervalos(
    det: pd.DataFrame,
    cruces: pd.DataFrame,
    geo: Geometria,
    clip_id: str,
    duracion_s: float,
    frames_procesados: np.ndarray,
    fps: float,
    intervalo_s: float = 30,
    ventana_frames: int = 20,
) -> pd.DataFrame:
    """Una fila por ventana de `intervalo_s` segundos sobre los `duracion_s` procesados."""
    n_buckets = max(1, int(np.ceil(duracion_s / intervalo_s - 1e-9)))
    bordes = np.arange(n_buckets + 1) * intervalo_s
    bordes[-1] = duracion_s

    def bucket_de(ts: pd.Series | np.ndarray) -> np.ndarray:
        return np.clip(np.floor(np.asarray(ts) / intervalo_s), 0, n_buckets - 1).astype(int)

    # flujo: objetos únicos con algún cruce, en la ventana de su primer cruce
    flujo = {"vehiculo": np.zeros(n_buckets), "peaton": np.zeros(n_buckets)}
    if not cruces.empty:
        primeros = cruces.sort_values("ts_cruce").groupby("track_id").first()
        tabla = primeros.assign(b=bucket_de(primeros["ts_cruce"])).groupby(["clase", "b"]).size()
        for (clase, b), n in tabla.items():
            if clase in flujo:
                flujo[clase][b] = n

    # densidad y velocidad: vehículos dentro de la ROI
    veh = vehiculos_en_roi(det, geo, ventana_frames)
    por_frame = veh.groupby("frame").size() if not veh.empty else pd.Series(dtype=float)
    conteo_frames = pd.Series(0.0, index=pd.Index(frames_procesados, name="frame"))
    conteo_frames.loc[por_frame.index.intersection(conteo_frames.index)] = por_frame
    b_frames = bucket_de(conteo_frames.index.to_numpy() / fps)
    densidad = conteo_frames.groupby(b_frames).mean().reindex(range(n_buckets))

    if not veh.empty:
        b_veh = bucket_de(veh["frame"].to_numpy() / fps)
        vel = veh.groupby(b_veh)["vel"].agg(["mean", "count"]).reindex(range(n_buckets))
    else:
        vel = pd.DataFrame({"mean": np.nan, "count": 0}, index=range(n_buckets))

    dur = np.diff(bordes)
    out = pd.DataFrame({
        "clip_id": clip_id,
        "bucket": np.arange(n_buckets),
        "t_inicio_s": bordes[:-1],
        "t_fin_s": bordes[1:],
        "duracion_s": dur,
        "flujo_veh_min": flujo["vehiculo"] / (dur / 60),
        "flujo_peat_min": flujo["peaton"] / (dur / 60),
        "densidad_media": densidad.to_numpy(),
        "velocidad_media_px": vel["mean"].to_numpy(),
        "n_velocidad": vel["count"].fillna(0).astype(int).to_numpy(),
        "congestion": pd.Series([None] * n_buckets, dtype="object"),
    })  # fmt: skip
    return out[COLUMNAS_INTERVALOS]
