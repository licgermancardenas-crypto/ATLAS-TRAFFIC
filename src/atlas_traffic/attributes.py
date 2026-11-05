"""Extracción de color por votación y recortes de evidencia (brief §4b).

1. Muestreo: hasta `max_frames` detecciones por track, repartidas a lo largo de su
   permanencia. Se descartan las cajas que tocan el borde del frame (objeto cortado) o
   que se solapan con otra caja más de `solape_max` (proporción del área propia).
2. Región: el 50 % central de la caja. En vehículos se sesga hacia la mitad superior,
   que es carrocería y no sombra bajo el chasis.
3. Clasificación por píxel en HSV contra una paleta de 8 colores:
   - poco brillo -> negro (el matiz de un píxel oscuro no es confiable);
   - poca saturación -> blanco / gris según el brillo;
   - si no, por matiz -> rojo / amarillo / verde / azul (los matices intermedios, como
     violeta o magenta, no votan).
   El color del frame es el más frecuente entre los píxeles de la región.
4. Votación: el color del track es la moda de los frames; color_confianza es la
   proporción de votos del ganador. Por debajo de `confianza_min` -> "indefinido".
5. Evidencia: recorte del frame donde la caja es más grande y menos ocluida, en
   thumbs/<track_id>.jpg.

Limitación conocida (va al README): blanco, plata y gris claro se confunden según la
luz; un vehículo blanco en sombra mide gris. Los saturados son mucho más confiables.
Solo se clasifica el color de vehículos; los peatones quedan sin color.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from atlas_traffic.config import ColorCfg

COLORES = ("blanco", "negro", "gris", "rojo", "azul", "verde", "amarillo")
INDEFINIDO = "indefinido"
COLUMNAS_ATRIBUTOS = ["track_id", "color", "color_confianza", "thumb"]


# --------------------------------------------------------------------------- muestreo


def solapes(det: pd.DataFrame) -> pd.Series:
    """Para cada detección: máxima intersección con otra caja del mismo frame / área propia."""
    valores = np.zeros(len(det))
    cajas = det[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    for posiciones in det.reset_index(drop=True).groupby("frame").indices.values():
        if len(posiciones) < 2:
            continue
        b = cajas[posiciones]
        ix = np.clip(
            np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]),
            0,
            None,
        )
        iy = np.clip(
            np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]),
            0,
            None,
        )
        inter = ix * iy
        np.fill_diagonal(inter, 0)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        valores[posiciones] = inter.max(axis=1) / np.maximum(area, 1)
    return pd.Series(valores, index=det.index)


def muestrear(
    det: pd.DataFrame, ancho: int, alto: int, cfg: ColorCfg, margen_borde: int = 2
) -> pd.DataFrame:
    """Detecciones elegidas para votar color (`votar`) y para el recorte (`es_thumb`)."""
    if det.empty:
        return det.assign(solape=[], votar=[], es_thumb=[])
    det = det.assign(solape=solapes(det))
    en_borde = (
        (det["x1"] <= margen_borde) | (det["y1"] <= margen_borde)
        | (det["x2"] >= ancho - margen_borde) | (det["y2"] >= alto - margen_borde)
    )  # fmt: skip
    valida = ~en_borde & (det["solape"] <= cfg.solape_max)
    area = (det["x2"] - det["x1"]) * (det["y2"] - det["y1"])
    votar = pd.Series(False, index=det.index)
    thumb = pd.Series(False, index=det.index)
    for _, g in det.groupby("track_id"):
        g = g.sort_values("frame")
        validas = g.index[valida.loc[g.index].to_numpy()]
        if len(validas):
            elegidas = validas[
                np.unique(np.linspace(0, len(validas) - 1, cfg.max_frames).round().astype(int))
            ]
            votar.loc[elegidas] = True
        candidatas = validas if len(validas) else g.index
        puntaje = area.loc[candidatas] * (1 - det.loc[candidatas, "solape"])
        thumb.loc[puntaje.idxmax()] = True
    return det.assign(votar=votar, es_thumb=thumb)


# --------------------------------------------------------------------------- color


def region_color(bbox: tuple[int, int, int, int], clase: str) -> tuple[int, int, int, int]:
    """50 % central de la caja; en vehículos, desplazado hacia la mitad superior."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    rx1, rx2 = x1 + 0.25 * w, x2 - 0.25 * w
    if clase == "vehiculo":
        ry1, ry2 = y1 + 0.10 * h, y1 + 0.60 * h  # 50 % de alto, corrido hacia arriba
    else:
        ry1, ry2 = y1 + 0.25 * h, y2 - 0.25 * h
    return int(rx1), int(ry1), max(int(rx2), int(rx1) + 1), max(int(ry2), int(ry1) + 1)


def brillo_fondo(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    cajas: np.ndarray,
    expansion: float = 0.5,
) -> float | None:
    """Brillo (V mediano) del entorno de la caja: un anillo alrededor, sin otras cajas.

    Es casi siempre asfalto, con la misma luz que el vehículo (sol, sombra, noche).
    Comparar contra él hace que "negro" y "blanco" no dependan de la exposición.
    """
    alto, ancho = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * expansion), int((y2 - y1) * expansion)
    ex1, ey1 = max(0, x1 - mx), max(0, y1 - my)
    ex2, ey2 = min(ancho, x2 + mx), min(alto, y2 + my)
    mascara = np.ones((ey2 - ey1, ex2 - ex1), dtype=bool)
    for cx1, cy1, cx2, cy2 in cajas:  # incluye la propia caja
        mascara[max(0, cy1 - ey1) : max(0, cy2 - ey1), max(0, cx1 - ex1) : max(0, cx2 - ex1)] = (
            False
        )
    if mascara.sum() < 20:
        return None
    v = cv2.cvtColor(frame[ey1:ey2, ex1:ex2], cv2.COLOR_BGR2HSV)[..., 2]
    return float(np.median(v[mascara]))


def umbrales_brillo(cfg: ColorCfg, v_fondo: float | None) -> tuple[float, float]:
    """(límite de negro, límite de blanco) en V. Relativos al fondo si hay fondo medido."""
    if cfg.brillo_relativo and v_fondo:
        return cfg.r_negro * v_fondo, min(255.0, cfg.r_blanco * v_fondo)
    return float(cfg.v_negro), float(cfg.v_blanco)


def clasificar_pixeles(hsv: np.ndarray, cfg: ColorCfg, v_fondo: float | None = None) -> np.ndarray:
    """Etiqueta por píxel (índice en COLORES, -1 = no vota). hsv en escala OpenCV."""
    h = hsv[..., 0].astype(int)
    s = hsv[..., 1].astype(int)
    v = hsv[..., 2].astype(int)
    v_negro, v_blanco = umbrales_brillo(cfg, v_fondo)
    out = np.full(h.shape, -1, dtype=int)
    oscuro = v < v_negro
    acromatico = ~oscuro & (s < cfg.sat_min)
    out[oscuro] = COLORES.index("negro")
    out[acromatico & (v >= v_blanco)] = COLORES.index("blanco")
    out[acromatico & (v < v_blanco)] = COLORES.index("gris")
    cromatico = ~oscuro & ~acromatico
    for nombre, (lo, hi) in {
        "rojo": (0, 12), "amarillo": (12, 35), "verde": (35, 85), "azul": (85, 130),
    }.items():  # fmt: skip
        out[cromatico & (h >= lo) & (h < hi)] = COLORES.index(nombre)
    out[cromatico & (h >= 160)] = COLORES.index("rojo")  # el rojo da la vuelta en 180
    return out


def color_de_region(bgr: np.ndarray, cfg: ColorCfg, v_fondo: float | None = None) -> str | None:
    """Color más frecuente entre los píxeles de la región; None si ninguno vota."""
    if bgr.size == 0:
        return None
    etiquetas = clasificar_pixeles(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), cfg, v_fondo).ravel()
    etiquetas = etiquetas[etiquetas >= 0]
    if etiquetas.size == 0:
        return None
    return COLORES[int(np.bincount(etiquetas, minlength=len(COLORES)).argmax())]


def votar_color(votos: list[str], confianza_min: float) -> tuple[str, float]:
    """Moda de los votos y proporción del ganador; debajo del mínimo -> indefinido."""
    if not votos:
        return INDEFINIDO, 0.0
    valores, cuentas = np.unique(votos, return_counts=True)
    i = int(cuentas.argmax())
    confianza = float(cuentas[i] / len(votos))
    return (str(valores[i]) if confianza >= confianza_min else INDEFINIDO), confianza


# --------------------------------------------------------------------------- pasada sobre el video


def extraer_atributos(
    video: Path, det: pd.DataFrame, cfg: ColorCfg, carpeta_thumbs: Path, margen_thumb: float = 0.15
) -> pd.DataFrame:
    """Una fila por track: color, color_confianza y ruta del recorte (relativa a la corrida)."""
    if det.empty:
        return pd.DataFrame(columns=COLUMNAS_ATRIBUTOS)
    cap = cv2.VideoCapture(str(video))
    ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    m = muestrear(det, ancho, alto, cfg)
    usadas = m[m["votar"] | m["es_thumb"]]
    por_frame = dict(tuple(usadas.groupby("frame")))
    cajas_por_frame = {
        f: g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=int)
        for f, g in det[det["frame"].isin(por_frame)].groupby("frame")
    }
    carpeta_thumbs.mkdir(parents=True, exist_ok=True)
    for viejo in carpeta_thumbs.glob("*.jpg"):
        viejo.unlink()  # no mezclar recortes de una corrida anterior
    votos: dict[int, list[str]] = {}
    thumbs: dict[int, str] = {}
    try:
        ultimo = max(por_frame) if por_frame else -1
        for i in range(ultimo + 1):
            if i not in por_frame:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            for r in por_frame[i].itertuples(index=False):
                tid = int(r.track_id)
                if r.votar and r.clase == "vehiculo":
                    caja = (r.x1, r.y1, r.x2, r.y2)
                    fondo = brillo_fondo(frame, caja, cajas_por_frame[i])
                    x1, y1, x2, y2 = region_color(caja, r.clase)
                    color = color_de_region(frame[y1:y2, x1:x2], cfg, fondo)
                    if color is not None:
                        votos.setdefault(tid, []).append(color)
                if r.es_thumb:
                    mx = int((r.x2 - r.x1) * margen_thumb)
                    my = int((r.y2 - r.y1) * margen_thumb)
                    recorte = frame[max(0, r.y1 - my) : r.y2 + my, max(0, r.x1 - mx) : r.x2 + mx]
                    nombre = f"{tid}.jpg"
                    cv2.imwrite(str(carpeta_thumbs / nombre), recorte)
                    thumbs[tid] = f"{carpeta_thumbs.name}/{nombre}"
    finally:
        cap.release()
    filas = []
    for tid, g in det.groupby("track_id"):
        if g["clase"].mode().iat[0] == "vehiculo":
            color, conf = votar_color(votos.get(int(tid), []), cfg.confianza_min)
        else:
            color, conf = None, np.nan
        filas.append((int(tid), color, conf, thumbs.get(int(tid))))
    return pd.DataFrame(filas, columns=COLUMNAS_ATRIBUTOS)
