"""Datos para "seguir un vehículo" en el dashboard: su ficha y su recorrido.

Sin OpenCV (el dashboard publicado corre sin él): solo pandas y numpy sobre los archivos
que deja el pipeline (trayectorias.parquet, giros, cruces, tracks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["Ficha", "ficha_vehiculo", "recorrido_suavizado", "EN_MOVIMIENTO_KMH"]

EN_MOVIMIENTO_KMH = 3.0
CALLES = {"av_cordoba": "Córdoba", "rodriguez_pena": "Rodríguez Peña"}


def _calle(linea: str) -> str:
    base = linea.rsplit("_", 1)[0]
    return CALLES.get(base, base.replace("_", " ").title())


@dataclass
class Ficha:
    track_id: int
    tipo: str | None = None
    color: str | None = None
    thumb: str | None = None
    aparece_s: float = np.nan
    sale_s: float = np.nan
    velocidad_kmh: float = np.nan  # mediana mientras se mueve
    velocidad_max_kmh: float = np.nan
    distancia_m: float = np.nan
    detenido_s: float = np.nan
    movimiento: str | None = None  # "Córdoba → sigue por Córdoba"
    cruces: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def en_escena_s(self) -> float:
        return self.sale_s - self.aparece_s


def recorrido_suavizado(tray: pd.DataFrame, paso_s: float = 0.25) -> pd.DataFrame:
    """Posición en el suelo cada `paso_s` segundos (mediana en cada paso): quita el temblor
    de la caja, que sumaría metros a la distancia recorrida."""
    if tray.empty or "x_m" not in tray:
        return pd.DataFrame(columns=["ts", "x_m", "y_m", "vel_kmh"])
    t = tray.sort_values("ts")
    paso = np.floor((t["ts"] - t["ts"].iat[0]) / paso_s)
    columnas = ["ts", "x_m", "y_m"] + (["vel_kmh"] if "vel_kmh" in t else [])
    return t.groupby(paso)[columnas].median().reset_index(drop=True)


def ficha_vehiculo(
    tid: int,
    tray: pd.DataFrame,
    tracks: pd.DataFrame,
    cruces: pd.DataFrame,
    giros: pd.DataFrame,
) -> Ficha:
    """Todo lo que se sabe de un vehículo de un clip (tablas ya filtradas por clip)."""
    f = Ficha(track_id=int(tid))
    t = tray[tray["track_id"] == tid].sort_values("ts")
    if not t.empty:
        f.aparece_s, f.sale_s = float(t["ts"].iat[0]), float(t["ts"].iat[-1])
    fila = tracks[tracks["track_id"] == tid]
    if not fila.empty:
        r = fila.iloc[0]
        f.tipo = r.get("clase_detalle")
        f.color = r.get("color") if isinstance(r.get("color"), str) else None
        f.thumb = r.get("thumb") if isinstance(r.get("thumb"), str) else None
        f.velocidad_kmh = float(r.get("velocidad_kmh", np.nan))
        f.velocidad_max_kmh = float(r.get("velocidad_max_kmh", np.nan))
    s = recorrido_suavizado(t)
    if len(s) >= 2:
        # suavizado por tramos, anclado en el primer y el último punto reales (si no, se
        # pierde medio tramo en cada punta)
        x = np.r_[t["x_m"].iat[0], s["x_m"], t["x_m"].iat[-1]]
        y = np.r_[t["y_m"].iat[0], s["y_m"], t["y_m"].iat[-1]]
        f.distancia_m = float(np.hypot(np.diff(x), np.diff(y)).sum())
    if "vel_kmh" in t and t["vel_kmh"].notna().any():
        dt = np.diff(t["ts"].to_numpy(), append=t["ts"].iat[-1])
        f.detenido_s = float(dt[(t["vel_kmh"] < EN_MOVIMIENTO_KMH).to_numpy()].sum())
    f.cruces = cruces[cruces["track_id"] == tid].sort_values("ts_cruce")
    if not giros.empty:
        mismo = giros["track_id"] == tid
        if "track_id_salida" in giros:  # recorrido reconstruido: tid puede ser la salida
            mismo |= giros["track_id_salida"] == tid
        g = giros[mismo]
        g = g.dropna(subset=["origen"])
        if not g.empty:
            o, d = g["origen"].iat[0], g["destino"].iat[0]
            if isinstance(d, str):
                verbo = "sigue por" if _calle(o) == _calle(d) else "dobla a"
                f.movimiento = f"{_calle(o)} → {verbo} {_calle(d)}"
            else:
                f.movimiento = f"entra por {_calle(o)} (no se lo vio salir)"
    return f
