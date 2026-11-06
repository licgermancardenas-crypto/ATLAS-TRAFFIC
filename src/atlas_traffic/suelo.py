"""Calibración métrica del suelo: de píxeles (coordenadas de referencia) a metros.

Para medir velocidades en km/h hace falta saber cuántos metros hay entre dos puntos de
la imagen. Con una cámara en un balcón, esa escala cambia en toda la imagen (lo lejano se
ve chico): hace falta una homografía del plano de la calle.

Referencia: las sendas peatonales. El Manual de Señalamiento Horizontal de Vialidad
Nacional (Res. 2501/2012, punto IV.3) fija franjas de 0,40 a 0,50 m de ancho con un
espacio igual a su ancho: la distancia entre franjas consecutivas (período) es de 0,80 a
1,00 m. Se usa 1,00 m (franjas de 0,50 m, lo habitual en CABA) y se verifica con el ancho
de las franjas medido sobre la calibración.

Método:
1. Fondo limpio: mediana de muchos cuadros estabilizados (borra los vehículos).
2. A lo largo de cada senda se muestrea el brillo y se detecta el centro de cada franja.
3. Modelo del suelo: las sendas que cruzan Córdoba son filas de franjas equiespaciadas en
   una dirección (X); las que cruzan Rodríguez Peña, en la perpendicular (Y). Con dos
   sendas por dirección quedan fijados el horizonte, la escala en las dos direcciones y
   el ángulo recto.
4. Se ajusta la homografía suelo -> imagen que mejor reproduce los centros detectados
   (mínimos cuadrados, Levenberg-Marquardt con numpy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field

__all__ = [
    "Senda",
    "CalibracionSuelo",
    "fondo_estabilizado",
    "centros_franjas",
    "ajustar_suelo",
    "transferir_suelo",
    "a_metros",
    "guardar_calibracion_suelo",
    "cargar_calibracion_suelo",
]


@dataclass
class Senda:
    """Una senda peatonal: segmento que la recorre por el medio, atravesando las franjas.

    direccion: "x" si las franjas se suceden en la dirección X del suelo (sendas que cruzan
    la avenida) o "y" (sendas que cruzan la calle transversal).
    """

    id: str
    direccion: Literal["x", "y"]
    p0: tuple[float, float]
    p1: tuple[float, float]
    centros: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    anchos_px: np.ndarray = field(default_factory=lambda: np.zeros(0))


class CalibracionSuelo(BaseModel):
    encuadre: str
    homografia: list[list[float]]  # coordenadas de referencia (px) -> suelo (m)
    periodo_m: float = Field(gt=0)
    residuo_px: float  # error medio de reproyección de los centros de franja
    n_franjas: int
    verificaciones: dict[str, float] = {}  # cantidades derivadas para controlar a ojo
    # posición en el suelo de cada senda (para calibrar otros encuadres de la misma
    # esquina): dirección, coordenada fija q y posición s0 de su primera franja (m)
    sendas: dict[str, dict] = {}
    notas: str = ""


def fondo_estabilizado(
    video: Path, homografias: dict[int, np.ndarray], tam: tuple[int, int], n: int = 41
) -> np.ndarray:
    """Mediana de `n` cuadros llevados al cuadro de referencia: la calle sin vehículos."""
    cuadros = sorted(homografias)
    elegidos = set(cuadros[:: max(1, len(cuadros) // n)])
    cap = cv2.VideoCapture(str(video))
    pila = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in elegidos:
            pila.append(cv2.warpPerspective(frame, homografias[i], tam))
        i += 1
    cap.release()
    if not pila:
        raise ValueError("no se pudo leer ningún cuadro para el fondo")
    return np.median(np.stack(pila), axis=0).astype(np.uint8)


def _perfil(gris: np.ndarray, p0, p1, ancho: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Brillo a lo largo de p0-p1 (1 muestra por px), promediado en una franja de ancho."""
    a, b = np.asarray(p0, float), np.asarray(p1, float)
    largo = int(np.hypot(*(b - a)))
    t = np.linspace(0, 1, largo)
    d = (b - a) / largo
    n = np.array([-d[1], d[0]])
    pts = a + np.outer(t, b - a)
    muestras = []
    for k in range(-(ancho // 2), ancho // 2 + 1):
        q = (pts + k * n).astype(np.float32)
        muestras.append(cv2.remap(gris, q[:, 0], q[:, 1], cv2.INTER_LINEAR).ravel())
    return pts, np.mean(muestras, axis=0)


def centros_franjas(fondo: np.ndarray, senda: Senda) -> Senda:
    """Centros (px de referencia) de las franjas blancas a lo largo de la senda.

    Umbral local (media móvil): el brillo cambia a lo largo de la senda (sombras, faroles).
    Se descartan las franjas cortadas en los extremos y las muy angostas (ruido).
    """
    gris = cv2.cvtColor(fondo, cv2.COLOR_BGR2GRAY) if fondo.ndim == 3 else fondo
    pts, v = _perfil(gris.astype(np.float32), senda.p0, senda.p1)
    v = np.convolve(v, np.ones(3) / 3, mode="same")
    ventana = max(9, len(v) // 6) | 1
    local = np.convolve(np.pad(v, ventana // 2, mode="edge"), np.ones(ventana) / ventana, "valid")
    blanco = v > local
    # tramos blancos
    bordes = np.flatnonzero(np.diff(np.r_[0, blanco.astype(int), 0]))
    tramos = list(zip(bordes[::2], bordes[1::2], strict=True))
    tramos = [(i, j) for i, j in tramos if i > 0 and j < len(v)]  # no cortados
    if not tramos:
        raise ValueError(f"senda {senda.id}: no se detectaron franjas")
    # franjas de ancho muy distinto al típico: ruido, una franja cortada o la ciclovía
    largos = np.array([j - i for i, j in tramos])
    tramos = [t for t, n in zip(tramos, largos, strict=True)
              if 0.6 * np.median(largos) <= n <= 1.6 * np.median(largos)]  # fmt: skip
    centros = np.array([pts[(i + j - 1) // 2] for i, j in tramos])
    anchos = np.array([j - i for i, j in tramos], dtype=float)
    # índice de cada franja: si falta una (tapada), el salto es de dos períodos
    paso = np.diff([(i + j) / 2 for i, j in tramos])
    tipico = np.median(paso) if len(paso) else 1.0
    saltos = np.maximum(1, np.round(paso / tipico)).astype(int) if len(paso) else []
    senda.centros = centros
    senda.indices = np.r_[0, np.cumsum(saltos)].astype(int)
    senda.anchos_px = anchos
    return senda


def _aplicar(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.c_[np.asarray(xy, float), np.ones(len(xy))] @ H.T
    return p[:, :2] / p[:, 2:3]


def a_metros(H: np.ndarray | list, xy: np.ndarray) -> np.ndarray:
    """Coordenadas de referencia (px, N x 2) -> suelo (m, N x 2)."""
    return _aplicar(np.asarray(H, float), np.asarray(xy, float).reshape(-1, 2))


def _modelo(params: np.ndarray, sendas: list[Senda], periodo: float) -> np.ndarray:
    """Centros predichos en la imagen para los parámetros dados."""
    G = np.r_[params[:8], 1.0].reshape(3, 3)  # suelo -> imagen
    pred = []
    for k, s in enumerate(sendas):
        q, s0 = (0.0, 0.0) if k == 0 else params[8 + 2 * (k - 1) : 10 + 2 * (k - 1)]
        a = s0 + s.indices * periodo
        suelo = np.c_[a, np.full_like(a, q)] if s.direccion == "x" else np.c_[np.full_like(a, q), a]
        pred.append(_aplicar(G, suelo))
    return np.vstack(pred)


def _levenberg_marquardt(f, x0: np.ndarray, iteraciones: int = 300) -> np.ndarray:
    x, lam = x0.astype(float), 1e-3
    r = f(x)
    costo = float(r @ r)
    for _ in range(iteraciones):
        eps = 1e-6 * np.maximum(1.0, np.abs(x))
        J = np.column_stack([(f(x + e) - r) / e[i] for i, e in enumerate(np.diag(eps))])
        A, g = J.T @ J, J.T @ r
        while True:
            paso = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r_nuevo = f(x + paso)
            c_nuevo = float(r_nuevo @ r_nuevo)
            if c_nuevo < costo:
                x, r, lam = x + paso, r_nuevo, lam / 3
                mejora = costo - c_nuevo
                costo = c_nuevo
                break
            lam *= 4
            if lam > 1e12:
                return x
        if mejora < 1e-10 * max(costo, 1e-12):
            break
    return x


def _suelo_de(s: Senda, q: float, s0: float, periodo: float, desfase: int = 0) -> np.ndarray:
    a = s0 + (s.indices + desfase) * periodo
    return np.c_[a, np.full_like(a, q)] if s.direccion == "x" else np.c_[np.full_like(a, q), a]


def transferir_suelo(
    sendas: list[Senda], referencia: dict[str, dict], periodo: float, rango: int = 6
) -> tuple[np.ndarray, float, dict[str, int]]:
    """Calibra otro encuadre de la misma esquina con las posiciones conocidas de las sendas.

    No se sabe a qué franja de la referencia corresponde la primera franja detectada acá
    (puede estar tapada o fuera de cuadro): se prueban los desfases posibles y se queda
    con el que mejor encaja. Devuelve (homografía imagen -> suelo, error medio en px,
    desfase elegido por senda).
    """
    import itertools

    obs = np.vstack([s.centros for s in sendas])
    mejor = (np.inf, None, None)
    for desfases in itertools.product(range(-rango, rango + 1), repeat=len(sendas)):
        suelo = np.vstack([
            _suelo_de(s, referencia[s.id]["q"], referencia[s.id]["s0"], periodo, d)
            for s, d in zip(sendas, desfases, strict=True)
        ])  # fmt: skip
        G, _ = cv2.findHomography(suelo, obs)
        if G is None:
            continue
        err = float(np.mean(np.hypot(*(_aplicar(G, suelo) - obs).T)))
        if err < mejor[0]:
            mejor = (err, G, desfases)
    err, G, desfases = mejor
    if G is None:
        raise ValueError("no se pudo calibrar el encuadre con las sendas dadas")
    H = np.linalg.inv(G)
    return H / H[2, 2], err, {s.id: int(d) for s, d in zip(sendas, desfases, strict=True)}


def ajustar_suelo(
    sendas: list[Senda], periodo: float, guia: dict[str, tuple[float, float]]
) -> tuple[np.ndarray, float, dict[str, dict]]:
    """Homografía imagen (px de referencia) -> suelo (m) y error medio de reproyección (px).

    `guia`: posición aproximada en metros (q, s0) de cada senda salvo la primera, que es el
    origen. Solo sirve de punto de partida: el ajuste la corrige.
    """
    obs = np.vstack([s.centros for s in sendas])
    # punto de partida: homografía directa con la posición aproximada de cada senda
    suelo0 = []
    for k, s in enumerate(sendas):
        q, s0 = (0.0, 0.0) if k == 0 else guia[s.id]
        a = s0 + s.indices * periodo
        suelo0.append(np.c_[a, np.full_like(a, q)] if s.direccion == "x"
                      else np.c_[np.full_like(a, q), a])  # fmt: skip
    G0, _ = cv2.findHomography(np.vstack(suelo0), obs)
    G0 = G0 / G0[2, 2]
    x0 = np.r_[G0.ravel()[:8], np.ravel([guia[s.id] for s in sendas[1:]])]
    x = _levenberg_marquardt(lambda p: (_modelo(p, sendas, periodo) - obs).ravel(), x0)
    residuo = float(np.mean(np.hypot(*(_modelo(x, sendas, periodo) - obs).T)))
    G = np.r_[x[:8], 1.0].reshape(3, 3)
    H = np.linalg.inv(G)
    posiciones = {}
    for k, s in enumerate(sendas):
        q, s0 = (0.0, 0.0) if k == 0 else x[8 + 2 * (k - 1) : 10 + 2 * (k - 1)]
        posiciones[s.id] = {"direccion": s.direccion, "q": float(q), "s0": float(s0)}
    return H / H[2, 2], residuo, posiciones


def guardar_calibracion_suelo(cal: CalibracionSuelo, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = cal.model_dump()
    texto = (
        f"# Calibración métrica del suelo del encuadre {cal.encuadre}.\n"
        "# Generado con scripts/calibrar_suelo.py (ver src/atlas_traffic/suelo.py).\n"
    )
    ruta.write_text(texto + yaml.safe_dump(datos, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")  # fmt: skip


def cargar_calibracion_suelo(ruta: Path) -> CalibracionSuelo | None:
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    return CalibracionSuelo.model_validate(yaml.safe_load(ruta.read_text(encoding="utf-8")))
