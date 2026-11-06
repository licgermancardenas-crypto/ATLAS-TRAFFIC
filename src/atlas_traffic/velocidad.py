"""Velocidad de los vehículos en km/h, con la calibración métrica del suelo (suelo.py).

- Punto medido: el centro del borde inferior de la caja, donde el vehículo toca el
  suelo. El centro de la caja está a ~1 m de altura y, visto desde un balcón, proyectado
  sobre el suelo exagera la velocidad.
- Ese punto se lleva al cuadro de referencia con la estabilización del cuadro y de ahí a
  metros con la homografía del suelo.
- Velocidad instantánea: desplazamiento en una ventana centrada (por defecto 0,5 s); así
  el temblor de la caja no suma velocidad a un auto quieto.
- Por vehículo: mediana de la velocidad mientras se mueve (más de 3 km/h) y percentil 90
  como velocidad máxima robusta.
- Por intervalo: velocidad media de todos los vehículos en la ROI (incluye los detenidos:
  mide cuán fluido está el tránsito) y percentil 85 de las velocidades de los vehículos,
  el indicador estándar de ingeniería de tránsito.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas_traffic.config import Geometria
from atlas_traffic.geometry import dentro_roi
from atlas_traffic.metrics import velocidad_instantanea
from atlas_traffic.stabilize import transformar_puntos
from atlas_traffic.suelo import a_metros

__all__ = ["en_suelo", "velocidad_kmh", "velocidad_por_track", "velocidad_por_intervalo"]

EN_MOVIMIENTO_KMH = 3.0


def en_suelo(det: pd.DataFrame, homografias: dict[int, np.ndarray], H_suelo) -> pd.DataFrame:
    """Agrega x_m, y_m: posición en el suelo (m) del punto de apoyo de cada caja."""
    if det.empty:
        return det.assign(x_m=pd.Series(dtype=float), y_m=pd.Series(dtype=float))
    apoyo = np.c_[(det["x1"] + det["x2"]) / 2, det["y2"]].astype(float)
    ref = np.empty_like(apoyo)
    frames = det["frame"].to_numpy()
    for f in np.unique(frames):
        m = frames == f
        H = homografias.get(int(f))
        ref[m] = apoyo[m] if H is None else transformar_puntos(H, apoyo[m])
    m = a_metros(H_suelo, ref)
    return det.assign(x_m=m[:, 0], y_m=m[:, 1])


def velocidad_kmh(det: pd.DataFrame, fps: float, ventana_s: float = 0.5) -> pd.Series:
    """Velocidad instantánea (km/h) de cada detección, con x_m / y_m ya calculados."""
    if det.empty:
        return pd.Series(dtype=float)
    ventana = max(2, round(ventana_s * fps))
    por_frame = velocidad_instantanea(det.assign(cx_ref=det["x_m"], cy_ref=det["y_m"]), ventana)
    return por_frame * fps * 3.6


def velocidad_por_track(veh: pd.DataFrame) -> pd.DataFrame:
    """Por track: mediana mientras se mueve y percentil 90 (km/h), sobre `vel_kmh`."""
    filas = []
    for tid, g in veh.groupby("track_id"):
        v = g["vel_kmh"].dropna().to_numpy()
        mov = v[v > EN_MOVIMIENTO_KMH]
        filas.append((
            int(tid),
            float(np.median(mov)) if len(mov) else np.nan,
            float(np.percentile(v, 90)) if len(v) else np.nan,
        ))  # fmt: skip
    return pd.DataFrame(filas, columns=["track_id", "velocidad_kmh", "velocidad_max_kmh"])


def velocidad_por_intervalo(
    veh: pd.DataFrame, geo: Geometria, bordes_s: np.ndarray
) -> pd.DataFrame:
    """Por ventana: velocidad media en la ROI (km/h) y percentil 85 de los vehículos."""
    n = len(bordes_s) - 1
    media, p85 = np.full(n, np.nan), np.full(n, np.nan)
    if not veh.empty:
        en_roi = veh[dentro_roi(geo.roi, veh[["cx_ref", "cy_ref"]].to_numpy())]
        b = np.clip(np.searchsorted(bordes_s, en_roi["ts"], side="right") - 1, 0, n - 1)
        for k, g in en_roi.groupby(b):
            v = g["vel_kmh"].dropna()
            if not v.empty:
                media[k] = v.mean()
            por_track = velocidad_por_track(g)["velocidad_kmh"].dropna()
            if len(por_track) >= 3:
                p85[k] = np.percentile(por_track, 85)
    return pd.DataFrame({"velocidad_kmh_media": media, "velocidad_kmh_p85": p85})
