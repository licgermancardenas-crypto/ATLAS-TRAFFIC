"""Contrato de detección (brief §4c).

El resto del pipeline solo conoce `Deteccion` y `Detector`. Ningún módulo aguas abajo
importa el motor de inferencia ni conoce su vocabulario de clases: el mapeo a las
etiquetas del proyecto (vehiculo / peaton) vive en cada adaptador.

Los motores se eligen por nombre (`detector:` en configs/default.yaml) a través de un
registro simple. Agregar un motor nuevo = registrar una fábrica, sin tocar counting.py,
metrics.py ni attributes.py.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

Clase = Literal["vehiculo", "peaton"]
CLASES: tuple[str, ...] = ("vehiculo", "peaton")


@dataclass(frozen=True)
class Deteccion:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    clase: str  # etiqueta normalizada del proyecto: vehiculo / peaton
    confianza: float
    track_id: int | None
    clase_detalle: str = ""  # car, truck, bus, motorcycle, person

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox inválida: {self.bbox}")
        if self.clase not in CLASES:
            raise ValueError(f"clase fuera del vocabulario del proyecto: {self.clase!r}")
        if not 0.0 <= self.confianza <= 1.0:
            raise ValueError(f"confianza fuera de [0, 1]: {self.confianza}")

    @property
    def centroide(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@runtime_checkable
class Detector(Protocol):
    def track(self, frame: np.ndarray) -> list[Deteccion]: ...

    def reset(self) -> None: ...  # limpia estado del tracker entre clips


# --------------------------------------------------------------------------- registro

DetectorFactory = Callable[..., Detector]
_REGISTRO: dict[str, DetectorFactory] = {}


def registrar(nombre: str) -> Callable[[DetectorFactory], DetectorFactory]:
    def deco(factory: DetectorFactory) -> DetectorFactory:
        if nombre in _REGISTRO:
            raise ValueError(f"detector ya registrado: {nombre}")
        _REGISTRO[nombre] = factory
        return factory

    return deco


def disponibles() -> list[str]:
    return sorted(_REGISTRO)


def crear_detector(nombre: str, **kwargs: object) -> Detector:
    try:
        factory = _REGISTRO[nombre]
    except KeyError:
        raise KeyError(f"detector desconocido {nombre!r}. Disponibles: {disponibles()}") from None
    return factory(**kwargs)


# --------------------------------------------------------------------------- FakeDetector


@registrar("fake")
class FakeDetector:
    """Devuelve detecciones fijas, frame a frame. Para tests y corridas en seco.

    `guion[i]` son las detecciones del frame i. Pasado el final del guion devuelve listas
    vacías. `reset()` vuelve al frame 0, como un tracker que arranca un clip nuevo.
    """

    def __init__(self, guion: Iterable[Sequence[Deteccion]] = ()) -> None:
        self._guion: list[list[Deteccion]] = [list(f) for f in guion]
        self._i = 0

    def track(self, frame: np.ndarray) -> list[Deteccion]:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame debe ser un np.ndarray")
        dets = self._guion[self._i] if self._i < len(self._guion) else []
        self._i += 1
        return list(dets)

    def reset(self) -> None:
        self._i = 0
