"""Compensación de movimiento de cámara (cámara en mano o que se corre).

Para cada frame se estima la homografía H que lo lleva al sistema de coordenadas de un
frame de referencia. La detección corre sobre el frame original; lo que se transforma
son los centroides. Conteo, ROI y velocidad usan esas coordenadas de referencia, y la
geometría del encuadre se marca sobre el frame de referencia.

Por qué una homografía alcanza: el movimiento de un teléfono en mano en un balcón es
casi pura rotación (y zoom), y una rotación de cámara induce una homografía exacta sobre
toda la escena, sin importar la profundidad. La traslación de la mano es mínima frente a
la distancia a la calle.

Puntos de interés: ORB, excluyendo las cajas de los objetos detectados (se mueven por
cuenta propia y ensucian la estimación). RANSAC descarta el resto de los outliers.
Si un frame no da una estimación confiable se reutiliza la última buena y se marca.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

IDENTIDAD = np.eye(3, dtype=np.float64)


@dataclass(frozen=True)
class Estimacion:
    H: np.ndarray  # 3x3, frame actual -> referencia
    inliers: int
    ok: bool  # False = no se pudo estimar y se reutilizó la última H buena


def _gris(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def mascara_sin_cajas(
    forma: tuple[int, int], cajas: Iterable[tuple[int, int, int, int]], margen: int = 4
) -> np.ndarray:
    """Máscara 255 donde se buscan puntos; 0 sobre las cajas de objetos en movimiento."""
    alto, ancho = forma
    m = np.full((alto, ancho), 255, np.uint8)
    for x1, y1, x2, y2 in cajas:
        m[
            max(0, y1 - margen) : min(alto, y2 + margen),
            max(0, x1 - margen) : min(ancho, x2 + margen),
        ] = 0
    return m


def homografia_plausible(
    H: np.ndarray, max_escala: float = 2.0, max_perspectiva: float = 2e-3
) -> bool:
    """Descarta homografías degeneradas: espejos, escalas extremas o perspectiva absurda."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    det = np.linalg.det(H[:2, :2])
    if det <= 0 or not (1 / max_escala**2 <= det <= max_escala**2):
        return False
    return bool(np.abs(H[2, :2]).max() <= max_perspectiva)


class Estabilizador:
    def __init__(
        self,
        referencia: np.ndarray,
        max_puntos: int = 1500,
        min_inliers: int = 40,
        umbral_ransac_px: float = 3.0,
    ) -> None:
        self._orb = cv2.ORB_create(nfeatures=max_puntos)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._min_inliers = min_inliers
        self._umbral = umbral_ransac_px
        self._kp_ref, self._des_ref = self._orb.detectAndCompute(_gris(referencia), None)
        if self._des_ref is None or len(self._kp_ref) < min_inliers:
            raise ValueError("el frame de referencia no tiene textura suficiente para estabilizar")
        self._ultima = IDENTIDAD.copy()

    def estimar(self, frame: np.ndarray, mascara: np.ndarray | None = None) -> Estimacion:
        # Ultralytics fija OpenCV en 1 hilo al importarse; ORB + matching rinden ~2.5x con todos.
        cv2.setNumThreads(os.cpu_count() or 1)
        kp, des = self._orb.detectAndCompute(_gris(frame), mascara)
        H, inliers = None, 0
        if des is not None and len(kp) >= self._min_inliers:
            pares = self._matcher.knnMatch(des, self._des_ref, k=2)
            buenos = [m for m, *resto in pares if resto and m.distance < 0.75 * resto[0].distance]
            if len(buenos) >= self._min_inliers:
                src = np.float32([kp[m.queryIdx].pt for m in buenos])
                dst = np.float32([self._kp_ref[m.trainIdx].pt for m in buenos])
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self._umbral)
                inliers = int(mask.sum()) if mask is not None else 0
        if inliers >= self._min_inliers and homografia_plausible(H):
            self._ultima = H
            return Estimacion(H=H, inliers=inliers, ok=True)
        return Estimacion(H=self._ultima.copy(), inliers=inliers, ok=False)


def transformar_puntos(H: np.ndarray, puntos: np.ndarray) -> np.ndarray:
    """Aplica H a un array (N, 2) de puntos."""
    if len(puntos) == 0:
        return np.empty((0, 2), dtype=np.float64)
    p = np.asarray(puntos, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def suavizar_homografias(H: np.ndarray, ventana: int) -> np.ndarray:
    """Media móvil centrada de una serie de homografías (N, 3, 3), normalizadas a h22 = 1.

    Promediar elemento a elemento es válido porque frames vecinos tienen H casi iguales.
    La ventana es simétrica y se achica cerca de los bordes (radio = min(ventana // 2,
    distancia al borde)): así un movimiento uniforme de cámara no queda sesgado al
    principio ni al final del clip.
    """
    n = len(H)
    if ventana <= 1 or n < 2:
        return H.copy()
    plano = (H / H[:, 2:3, 2:3]).reshape(n, 9)
    acum = np.vstack([np.zeros((1, 9)), np.cumsum(plano, axis=0)])
    idx = np.arange(n)
    radio = np.minimum(ventana // 2, np.minimum(idx, n - 1 - idx))
    ini, fin = idx - radio, idx + radio + 1
    suave = (acum[fin] - acum[ini]) / (fin - ini)[:, None]
    return suave.reshape(n, 3, 3)
