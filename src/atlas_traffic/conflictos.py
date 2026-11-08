"""Situaciones de riesgo entre vehículos y peatones (conflictos de tránsito).

Indicador: PET (post-encroachment time), el estándar de los estudios de seguridad vial
con video. Es el tiempo entre que un usuario deja un lugar y otro llega a ese mismo
lugar. No hace falta que choquen ni que estén juntos al mismo tiempo: un auto que pasa
por donde un peatón estaba 1 segundo antes estuvo a 1 segundo de atropellarlo.

Acá "el mismo lugar" es a menos de `radio_m` (2 m) en el suelo, con la calibración
métrica (suelo.py). Filtros, validados a ojo en los clips (ver docs/conflictos.md):
- el vehículo circula (más de 8 km/h) y se desplaza al menos 10 m en el clip: si no, es
  un objeto fijo de la vereda que el detector toma por vehículo;
- el peatón se siguió al menos 1 s (15 detecciones): una detección de un instante en
  medio de la calle suele ser un poste o un reflejo;
- el peatón está en la calzada: por ese punto pasan al menos 3 vehículos en el clip. Un
  peatón que espera en el cordón está a ~2 m de los autos, pero no en su camino.

Gravedad por PET: menos de 1 s grave, 1-2 s moderado, 2-3 s leve.

Sin OpenCV: corre también en el dashboard publicado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas_traffic.config import ConflictosCfg

__all__ = ["COLUMNAS_CONFLICTOS", "detectar_conflictos", "gravedad"]

COLUMNAS_CONFLICTOS = [
    "track_id",  # vehículo
    "track_id_peaton",
    "ts",  # cuando el vehículo pasa por el lugar
    "ts_peaton",  # cuando el peatón está en ese lugar
    "pet_s",
    "distancia_m",
    "vel_kmh",
    "x_m",
    "y_m",
    "primero",  # quién pasó primero: "peaton" o "vehiculo"
    "gravedad",
]


def gravedad(pet_s: float, cfg: ConflictosCfg) -> str:
    if pet_s < cfg.grave_s:
        return "grave"
    if pet_s < cfg.moderado_s:
        return "moderado"
    return "leve"


def _en_calzada(xy: np.ndarray, veh: pd.DataFrame, cfg: ConflictosCfg) -> np.ndarray:
    """Por punto: ¿pasan por ahí al menos N vehículos distintos circulando?"""
    v = veh[veh["vel_kmh"] > cfg.vel_min_kmh]
    if v.empty or len(xy) == 0:
        return np.zeros(len(xy), dtype=bool)
    vxy = v[["x_m", "y_m"]].to_numpy(float)
    ids = v["track_id"].to_numpy()
    res = np.zeros(len(xy), dtype=bool)
    for k, p in enumerate(xy):
        cerca = np.hypot(vxy[:, 0] - p[0], vxy[:, 1] - p[1]) < cfg.calzada_radio_m
        res[k] = len(np.unique(ids[cerca])) >= cfg.calzada_min_vehiculos
    return res


def detectar_conflictos(veh: pd.DataFrame, ped: pd.DataFrame, cfg: ConflictosCfg) -> pd.DataFrame:
    """Un registro por par vehículo-peatón con PET < pet_max_s (el momento más cercano).

    `veh`: track_id, ts, x_m, y_m, vel_kmh. `ped`: track_id, ts, x_m, y_m. Ambos ya
    filtrados a la zona de análisis (la calibración del suelo vale ahí).
    """
    if veh.empty or ped.empty:
        return pd.DataFrame(columns=COLUMNAS_CONFLICTOS)
    veh = veh.dropna(subset=["x_m", "y_m", "vel_kmh"])
    # objetos fijos: no se desplazan aunque la caja tiemble
    extremos = veh.sort_values("ts").groupby("track_id")[["x_m", "y_m"]].agg(["first", "last"])
    despl = np.hypot(
        extremos[("x_m", "last")] - extremos[("x_m", "first")],
        extremos[("y_m", "last")] - extremos[("y_m", "first")],
    )
    veh = veh[veh["track_id"].isin(despl[despl >= cfg.desplazamiento_min_m].index)]
    filas = []
    for pid, p in ped.sort_values("ts").groupby("track_id"):
        pt, pxy = p["ts"].to_numpy(float), p[["x_m", "y_m"]].to_numpy(float)
        if len(p) < cfg.peaton_min_detecciones or pt[-1] - pt[0] < cfg.peaton_min_s:
            continue
        cerca_t = veh[veh["ts"].between(pt[0] - cfg.pet_max_s, pt[-1] + cfg.pet_max_s)]
        calzada = None  # se calcula una vez por peatón, solo si hace falta
        for vid, v in cerca_t.groupby("track_id"):
            vt, vxy = v["ts"].to_numpy(float), v[["x_m", "y_m"]].to_numpy(float)
            vv = v["vel_kmh"].to_numpy(float)
            d = np.hypot(pxy[:, None, 0] - vxy[None, :, 0], pxy[:, None, 1] - vxy[None, :, 1])
            dt = np.abs(pt[:, None] - vt[None, :])
            ok = (d < cfg.radio_m) & (vv[None, :] > cfg.vel_min_kmh) & (dt < cfg.pet_max_s)
            if not ok.any():
                continue
            if calzada is None:
                calzada = _en_calzada(pxy, veh, cfg)
            ok &= calzada[:, None]
            if not ok.any():
                continue
            i, j = np.unravel_index(np.argmin(np.where(ok, dt, np.inf)), d.shape)
            pet = float(dt[i, j])
            filas.append((
                int(vid), int(pid), float(vt[j]), float(pt[i]), pet, float(d[i, j]),
                float(vv[j]), float(pxy[i, 0]), float(pxy[i, 1]),
                "peaton" if pt[i] <= vt[j] else "vehiculo", gravedad(pet, cfg),
            ))  # fmt: skip
    out = pd.DataFrame(filas, columns=COLUMNAS_CONFLICTOS)
    return out.sort_values("ts", ignore_index=True)
