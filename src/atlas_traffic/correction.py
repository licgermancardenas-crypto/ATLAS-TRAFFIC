"""Corrección de conteos por tasa de detección, con intervalo de confianza.

Los conteos automáticos pierden vehículos (motos lejanas, autos oscuros de noche) y,
según la validación, prácticamente no inventan: el error es por omisión. Si de día la
línea del frente detecta 38 de 40 vehículos, un conteo de 19 se corrige a 19 / 0.95 = 20.

- La tasa de detección se mide con los conteos manuales de la validación
  (docs/validacion_conteo_registros.csv), agrupando por línea y por luz (día / noche),
  y se le calcula un intervalo de Wilson.
- El conteo corregido lleva intervalo: [c / tasa_alta, c / tasa_baja]. Ese intervalo
  refleja la incertidumbre de la tasa (cuántos vehículos se validaron); no incluye la
  variabilidad propia del tránsito.
- Una línea sin validación para esa luz no se corrige y queda marcada.

Flujo entrante: los vehículos que **entran** al cruce, cada uno contado en su línea de
entrada. Cada vehículo entra una sola vez (no hay doble conteo) y la entrada de Córdoba
es la línea más cercana a la cámara, donde los vehículos se ven más grandes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

LUCES = ("dia", "noche")


@dataclass(frozen=True)
class Tasa:
    linea: str
    luz: str
    detectados: int
    reales: int
    tasa: float
    ic_bajo: float
    ic_alto: float


def wilson(k: int, n: int, confianza: float = 0.95) -> tuple[float, float, float]:
    """Proporción k/n con su intervalo de Wilson (se porta bien con n chico)."""
    if n <= 0:
        raise ValueError("n tiene que ser positivo")
    z = NormalDist().inv_cdf(0.5 + confianza / 2)
    p = k / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margen = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return p, max(0.0, centro - margen), min(1.0, centro + margen)


def leer_registros(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta) if Path(ruta).exists() else pd.DataFrame()


def calcular_tasas(registros: pd.DataFrame, confianza: float = 0.95) -> dict[tuple[str, str], Tasa]:
    """Tasa de detección por (línea, luz), juntando todos los clips validados."""
    if registros.empty:
        return {}
    tasas = {}
    for (linea, luz), g in registros.groupby(["linea", "luz"]):
        n = int(g["manual"].sum())
        if n == 0:
            continue  # nada que medir (ej. ningún vehículo pasó por esa línea)
        k = int(np.minimum(g["automatico"], g["manual"]).sum())  # sin falsos positivos
        p, lo, hi = wilson(k, n, confianza)
        tasas[(linea, luz)] = Tasa(linea, luz, k, n, p, lo, hi)
    return tasas


def corregir(conteo: float, tasa: Tasa | None) -> tuple[float, float, float]:
    """(corregido, ic_bajo, ic_alto). Sin tasa: sin corrección."""
    if tasa is None or tasa.tasa == 0:
        return float(conteo), float("nan"), float("nan")
    alto = conteo / tasa.ic_bajo if tasa.ic_bajo > 0 else float("inf")
    return conteo / tasa.tasa, conteo / tasa.ic_alto, alto


def detectar_luz(frame: np.ndarray, umbral_rojo_azul: float = 1.4) -> str:
    """ "noche" si la calle (mitad inferior) está teñida de naranja por el alumbrado.

    El cociente rojo / azul de la calle separa bien: día 1.04-1.19, noche 1.56-1.86 en
    los clips de muestra. El brillo solo separa poco (el teléfono compensa la exposición).
    """
    calle = frame[frame.shape[0] // 2 :].reshape(-1, 3).astype(float)
    b, _, r = np.median(calle, axis=0)
    return "noche" if b > 0 and r / b > umbral_rojo_azul else "dia"


def corregir_conteo(
    conteo: pd.DataFrame, tasas: dict[tuple[str, str], Tasa], luz: str
) -> pd.DataFrame:
    """Agrega al conteo de vehículos por (aproximación, sentido) la versión corregida."""
    filas = []
    for (aprox, sentido), fila in conteo.iterrows():
        tasa = tasas.get((aprox, luz))
        c, lo, hi = corregir(fila["vehiculo"], tasa)
        filas.append({
            "aproximacion": aprox, "sentido": sentido, "vehiculo": int(fila["vehiculo"]),
            "tasa_deteccion": tasa.tasa if tasa else np.nan,
            "vehiculo_corregido": c, "ic_bajo": lo, "ic_alto": hi,
            "validada": tasa is not None,
        })  # fmt: skip
    return pd.DataFrame(filas).set_index(["aproximacion", "sentido"])


def flujo_entrante(
    cruces: pd.DataFrame,
    tasas: dict[tuple[str, str], Tasa],
    luz: str,
    bordes_s: np.ndarray,
) -> pd.DataFrame:
    """Flujo de vehículos que entran al cruce por ventana (veh/min): crudo, corregido e IC.

    `bordes_s`: límites de las ventanas en segundos (n + 1 valores).
    """
    n = len(bordes_s) - 1
    dur_min = np.diff(bordes_s) / 60
    crudo, corr, lo, hi = (np.zeros(n) for _ in range(4))
    validado = np.ones(n, dtype=bool)
    if not cruces.empty:
        e = cruces[(cruces["sentido"] == "entrante") & (cruces["clase"] == "vehiculo")]
        b = np.clip(np.searchsorted(bordes_s, e["ts_cruce"], side="right") - 1, 0, n - 1)
        for (bucket, linea), cantidad in e.groupby([b, e["aproximacion"]]).size().items():
            c, l_, h_ = corregir(cantidad, tasas.get((linea, luz)))
            crudo[bucket] += cantidad
            corr[bucket] += c
            if np.isnan(l_):  # línea sin validar: entra sin corrección y sin margen
                l_ = h_ = c
                validado[bucket] = False
            lo[bucket] += l_
            hi[bucket] += h_
    return pd.DataFrame({
        "flujo_entrante_min": crudo / dur_min,
        "flujo_entrante_corr_min": corr / dur_min,
        "flujo_entrante_ic_bajo": lo / dur_min,
        "flujo_entrante_ic_alto": hi / dur_min,
        "flujo_entrante_validado": validado,
    })  # fmt: skip
