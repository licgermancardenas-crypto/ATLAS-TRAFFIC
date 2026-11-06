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

    def __init__(self, guion: Iterable[Sequence[Deteccion]] = (), **_motor: object) -> None:
        # **_motor: ignora los parámetros de motor de la config (modelo, imgsz...), para
        # poder elegir `detector: fake` desde default.yaml sin tocar nada más.
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


# --------------------------------------------------------------------------- Ultralytics

# Vocabulario del modelo -> vocabulario del proyecto. Vive acá y en ningún otro lado.
# Se traduce por NOMBRE de clase, no por número: el modelo de COCO (80 clases) y un
# modelo ajustado para esta esquina (solo estas 5) numeran distinto pero usan los mismos
# nombres, así que el mismo adaptador sirve para los dos.
NOMBRE_A_PROYECTO: dict[str, str] = {
    "person": "peaton",
    "car": "vehiculo",
    "motorcycle": "vehiculo",
    "bus": "vehiculo",
    "truck": "vehiculo",
}


@registrar("ultralytics")
class UltralyticsDetector:
    """YOLO + ByteTrack vía Ultralytics (AGPL-3.0, extra `detect`).

    El import es perezoso: el paquete solo carga Ultralytics si se instancia esta clase.
    """

    def __init__(
        self,
        modelo: str = "yolov8n.pt",
        dispositivo: str = "cpu",
        confianza_min: float = 0.3,
        iou: float = 0.5,
        tracker: str = "bytetrack.yaml",
        imgsz: int = 640,
        nms_agnostico: bool = True,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - depende del entorno
            raise ImportError(
                "Falta Ultralytics. Instalalo con: uv pip install -e '.[detect]'"
            ) from e
        self._yolo_cls = YOLO
        self._modelo = modelo
        self._kw = dict(
            persist=True,
            tracker=tracker,
            conf=confianza_min,
            iou=iou,
            imgsz=imgsz,
            agnostic_nms=nms_agnostico,
            device=dispositivo,
            verbose=False,
        )
        self._model = YOLO(modelo)
        self._kw["classes"] = self._clases_utiles()

    def ajustar(self, imgsz: int) -> None:
        """Cambia la resolución de inferencia (se ajusta por clip, ver Clip.imgsz)."""
        self._kw["imgsz"] = imgsz

    def _clases_utiles(self) -> list[int]:
        """Números de clase del modelo cargado que corresponden a las clases del proyecto."""
        return sorted(i for i, n in self._model.names.items() if n in NOMBRE_A_PROYECTO)

    def track(self, frame: np.ndarray) -> list[Deteccion]:
        res = self._model.track(frame, **self._kw)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().round().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(cls)
        h, w = frame.shape[:2]
        salida: list[Deteccion] = []
        nombres = self._model.names
        for (x1, y1, x2, y2), c, p, tid in zip(xyxy, cls, conf, ids, strict=True):
            detalle = nombres.get(int(c))
            if detalle not in NOMBRE_A_PROYECTO:
                continue
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            clase = NOMBRE_A_PROYECTO[detalle]
            salida.append(
                Deteccion(
                    bbox=(x1, y1, x2, y2),
                    clase=clase,
                    confianza=float(p),
                    track_id=None if tid is None else int(tid),
                    clase_detalle=detalle,
                )
            )
        return salida

    def reset(self) -> None:
        # Un modelo nuevo trae un tracker limpio: los IDs vuelven a arrancar en 1.
        self._model = self._yolo_cls(self._modelo)
        self._kw["classes"] = self._clases_utiles()
