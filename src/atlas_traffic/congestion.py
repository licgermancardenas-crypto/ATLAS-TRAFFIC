"""Calibración de umbrales y clasificación de congestión (brief §4).

Congestión = densidad + velocidad, no volumen: mucho flujo a buena velocidad es
tránsito fluido; mucha densidad con velocidad baja es congestión.

Paso 1 — Velocidad normalizada por encuadre: velocidad_media_px / p90 de las velocidades
instantáneas observadas en ese encuadre (todos los clips que comparten geometría). El
p90 representa el flujo libre de esa vista; así la escala en píxeles (zoom, distancia)
deja de importar.
SUPUESTO: en cada encuadre hay al menos un momento de flujo libre. Si un encuadre solo
tiene clips congestionados, su p90 es bajo, la velocidad relativa se infla y el encuadre
queda sesgado hacia LOW.

Paso 2 — Umbrales con los propios datos: densidad y velocidad normalizada se cortan en
los percentiles 33 y 66 de todos los intervalos. Se persisten en calibracion.yaml.

Paso 3 — Matriz explícita (un parámetro discutible, no una verdad):

    densidad \\ velocidad   baja       media      alta
    baja                    LOW        LOW        LOW
    media                   MODERATE   MODERATE   LOW
    alta                    HIGH       HIGH       MODERATE

Limitación: la densidad es "vehículos simultáneos dentro de la ROI" y no se normaliza.
Si los encuadres tienen ROIs de distinto tamaño físico, sus densidades no son del todo
comparables (el brief recomienda un único encuadre por eso).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from atlas_traffic.config import Calibracion, Cortes

NIVELES = ("baja", "media", "alta")
MATRIZ: dict[tuple[str, str], str] = {
    ("baja", "baja"): "LOW",
    ("baja", "media"): "LOW",
    ("baja", "alta"): "LOW",
    ("media", "baja"): "MODERATE",
    ("media", "media"): "MODERATE",
    ("media", "alta"): "LOW",
    ("alta", "baja"): "HIGH",
    ("alta", "media"): "HIGH",
    ("alta", "alta"): "MODERATE",
}


def nivel(valor: float, cortes: Cortes) -> str | None:
    """baja (< p33), media (< p66) o alta. None si el valor no existe."""
    if valor is None or not np.isfinite(valor):
        return None
    if valor < cortes.p33:
        return "baja"
    return "media" if valor < cortes.p66 else "alta"


def clasificar(densidad: float, velocidad_norm: float, cal: Calibracion) -> str | None:
    """Nivel de congestión según la matriz. Sin velocidad medible solo se decide si la
    densidad es baja (no hay a quién medirle velocidad = no hay congestión)."""
    d = nivel(densidad, cal.densidad)
    if d is None:
        return None
    if d == "baja":
        return "LOW"
    v = nivel(velocidad_norm, cal.velocidad_norm)
    return None if v is None else MATRIZ[(d, v)]


def calibrar(
    intervalos: pd.DataFrame, velocidades: dict[str, np.ndarray], clips: list[str] | None = None
) -> Calibracion:
    """Umbrales a partir de los intervalos de todos los clips.

    `intervalos` necesita las columnas encuadre, densidad_media y velocidad_media_px.
    `velocidades`: por encuadre, las velocidades instantáneas (px/frame) de los vehículos
    dentro de la ROI, de todos sus clips. De ahí sale el p90 de flujo libre.
    """
    p90 = {enc: float(np.nanpercentile(v, 90)) for enc, v in velocidades.items() if len(v)}
    faltan = set(intervalos["encuadre"]) - set(p90)
    if faltan:
        raise ValueError(f"no hay velocidades para los encuadres {sorted(faltan)}")
    vnorm = intervalos["velocidad_media_px"] / intervalos["encuadre"].map(p90)
    dens = intervalos["densidad_media"].dropna()
    vnorm = vnorm.dropna()
    if dens.empty or vnorm.empty:
        raise ValueError("no hay intervalos con densidad y velocidad para calibrar")
    return Calibracion(
        velocidad_p90_por_encuadre=p90,
        densidad=Cortes(p33=float(np.percentile(dens, 33)), p66=float(np.percentile(dens, 66))),
        velocidad_norm=Cortes(
            p33=float(np.percentile(vnorm, 33)), p66=float(np.percentile(vnorm, 66))
        ),
        generado=dt.datetime.now().replace(microsecond=0),
        clips=sorted(clips or []),
        n_intervalos=int(len(intervalos)),
    )


def aplicar(intervalos: pd.DataFrame, encuadre: str, cal: Calibracion) -> pd.DataFrame:
    """Completa velocidad_norm y congestion en los intervalos de un clip."""
    out = intervalos.copy()
    p90 = cal.velocidad_p90_por_encuadre.get(encuadre)
    if p90 is None or p90 <= 0:
        out["velocidad_norm"] = np.nan
        out["congestion"] = None
        return out
    out["velocidad_norm"] = out["velocidad_media_px"] / p90
    out["congestion"] = [
        clasificar(d, v, cal)
        for d, v in zip(out["densidad_media"], out["velocidad_norm"], strict=True)
    ]
    return out
