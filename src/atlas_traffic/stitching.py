"""Unión de tracks cortados (re-identificación por movimiento).

Cuando algo tapa un objeto (un árbol, un colectivo) el tracker lo pierde, y al reaparecer
le da un número nuevo: el mismo auto o la misma persona queda partido en dos tracks y se
puede contar dos veces. Acá se unen esos pedazos.

Un track B continúa a un track A si:
- son de la misma clase (vehículo o peatón);
- B empieza después de que A termina, con un hueco de a lo sumo `hueco_max_s`;
- B aparece donde A tendría que estar si siguió moviéndose igual (posición final de A +
  su velocidad × el hueco), a menos de `factor_distancia` alturas de caja;
- las cajas tienen tamaño parecido.

Entre todos los pares posibles se eligen primero los más parecidos, y cada track continúa
a lo sumo a uno y es continuado por a lo sumo uno. Todo en coordenadas de referencia
(estabilizadas), así el movimiento de la cámara no confunde la predicción.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["resumen_extremos", "unir_tracks", "aplicar_uniones"]


def _velocidad(ts: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Velocidad (px/s) por ajuste lineal de los puntos; 0 si hay pocos."""
    if len(ts) < 3 or np.ptp(ts) <= 0:
        return np.zeros(2)
    return np.array([np.polyfit(ts, xy[:, k], 1)[0] for k in range(2)])


def resumen_extremos(det: pd.DataFrame, n_puntos: int = 10) -> pd.DataFrame:
    """Una fila por track: clase, inicio y fin (instante, posición, velocidad) y altura."""
    filas = []
    for tid, g in det.sort_values("frame").groupby("track_id"):
        ts = g["ts"].to_numpy(float)
        xy = g[["cx_ref", "cy_ref"]].to_numpy(float)
        alto = float(np.median(g["y2"].to_numpy(float) - g["y1"].to_numpy(float)))
        filas.append(dict(
            track_id=int(tid), clase=g["clase"].mode().iat[0],
            t_ini=ts[0], t_fin=ts[-1], f_ini=int(g["frame"].iat[0]),
            f_fin=int(g["frame"].iat[-1]),
            x_ini=xy[0, 0], y_ini=xy[0, 1], x_fin=xy[-1, 0], y_fin=xy[-1, 1],
            v_ini=_velocidad(ts[:n_puntos], xy[:n_puntos]),
            v_fin=_velocidad(ts[-n_puntos:], xy[-n_puntos:]),
            alto=max(alto, 1.0),
        ))  # fmt: skip
    return pd.DataFrame(filas)


def unir_tracks(
    det: pd.DataFrame,
    hueco_max_s: float = 2.0,
    factor_distancia: float = 1.0,
    distancia_min_px: float = 6.0,
    razon_tamano_max: float = 1.6,
    clases: tuple[str, ...] | None = None,
    velocidad_min_px_s: float = 0.0,
) -> dict[int, int]:
    """Devuelve {track_id original: track_id unido} solo para los tracks que cambian.

    El track unido conserva el número del primer pedazo.
    """
    if det.empty:
        return {}
    ext = resumen_extremos(det)
    if clases is not None:
        ext = ext[ext["clase"].isin(clases)]
    candidatos = []
    for a in ext.itertuples():
        for b in ext.itertuples():
            if a.track_id == b.track_id or a.clase != b.clase or b.f_ini <= a.f_fin:
                continue
            hueco = b.t_ini - a.t_fin
            if hueco > hueco_max_s:
                continue
            razon = max(a.alto, b.alto) / min(a.alto, b.alto)
            if razon > razon_tamano_max:
                continue
            # dónde debería estar A al reaparecer, y desde dónde debería venir B
            pred_a = np.array([a.x_fin, a.y_fin]) + a.v_fin * hueco
            pred_b = np.array([b.x_ini, b.y_ini]) - b.v_ini * hueco
            err = 0.5 * (
                np.hypot(*(pred_a - [b.x_ini, b.y_ini])) + np.hypot(*(pred_b - [a.x_fin, a.y_fin]))
            )
            limite = max(distancia_min_px, factor_distancia * 0.5 * (a.alto + b.alto))
            if err > limite:
                continue
            # si los dos se mueven, tienen que ir hacia el mismo lado
            na, nb = np.linalg.norm(a.v_fin), np.linalg.norm(b.v_ini)
            if min(na, nb) < velocidad_min_px_s:
                continue
            if na > 5 and nb > 5 and float(a.v_fin @ b.v_ini) / (na * nb) < 0.5:
                continue
            candidatos.append((err / limite + hueco / hueco_max_s, a.track_id, b.track_id))

    siguiente: dict[int, int] = {}
    anterior: dict[int, int] = {}
    for _, ta, tb in sorted(candidatos):
        if ta in siguiente or tb in anterior:
            continue
        siguiente[ta], anterior[tb] = tb, ta

    mapa: dict[int, int] = {}
    for tb in anterior:
        raiz = tb
        while raiz in anterior:
            raiz = anterior[raiz]
        mapa[tb] = raiz
    return mapa


def aplicar_uniones(det: pd.DataFrame, mapa: dict[int, int]) -> pd.DataFrame:
    """Renumera los tracks según el mapa de `unir_tracks`."""
    if not mapa or det.empty:
        return det
    det = det.copy()
    det["track_id"] = (
        det["track_id"].map(lambda t: mapa.get(int(t), int(t))).astype(det["track_id"].dtype)
    )
    return det
