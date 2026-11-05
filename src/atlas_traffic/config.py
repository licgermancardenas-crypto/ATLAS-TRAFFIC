"""Modelos Pydantic de configuración y carga de YAML.

Cuatro archivos de configuración, cuatro modelos:
- configs/default.yaml      -> AppConfig       (parámetros del pipeline)
- configs/clips.yaml        -> ClipsFile       (registro de clips, ver clips.py)
- configs/geometria/*.yaml  -> Geometria       (ROI y líneas de conteo por encuadre)
- configs/calibracion.yaml  -> Calibracion     (umbrales de congestión, generado)

Todos usan extra="forbid": un typo en el YAML es un error, no un campo ignorado.
"""

from __future__ import annotations

import datetime as dt
import unicodedata
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

Punto = tuple[int, int]
Sentido = Literal["entrante", "saliente"]

DIAS_SEMANA = ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo")


class _Modelo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- default.yaml


class DeteccionCfg(_Modelo):
    detector: str = "ultralytics"  # nombre en el registro de detect.py
    modelo: str = "yolov8n.pt"  # yolov8n para iterar, yolov8s para la corrida final
    dispositivo: str = "cpu"
    confianza_min: float = Field(
        0.1, ge=0.0, le=1.0
    )  # bajo a propósito: ByteTrack re-asocia con las débiles
    iou: float = Field(0.5, ge=0.0, le=1.0)
    tracker: str = "configs/bytetrack.yaml"
    imgsz: int = Field(640, ge=32, multiple_of=32)  # lado de inferencia; subir ayuda a lo lejano

    def kwargs_motor(self) -> dict[str, object]:
        """Argumentos para la fábrica del detector (todo menos el nombre)."""
        return self.model_dump(exclude={"detector"})


class VideoCfg(_Modelo):
    paso_frames: int = Field(2, ge=1)  # procesar 1 de cada N frames (60 fps -> 2 da 30 fps)


class EstabilizacionCfg(_Modelo):
    """Compensación de movimiento de cámara (ver stabilize.py)."""

    activa: bool = True  # con cámara fija la H estimada es ~identidad: no hace daño
    frame_referencia: int = Field(0, ge=0)  # la geometría del encuadre se marca sobre este
    max_puntos: int = Field(1500, ge=100)
    min_inliers: int = Field(40, ge=8)
    umbral_ransac_px: float = Field(3.0, gt=0)
    ventana_suavizado: int = Field(9, ge=1)  # frames procesados; 1 = sin suavizar


class MetricasCfg(_Modelo):
    intervalo_s: int = Field(30, gt=0)
    ventana_velocidad_s: float = Field(0.33, gt=0)  # ventana para medir desplazamiento


class ColorCfg(_Modelo):
    max_frames: int = Field(10, gt=0)
    solape_max: float = Field(0.4, ge=0.0, le=1.0)
    confianza_min: float = Field(0.5, ge=0.0, le=1.0)
    sat_min: int = Field(40, ge=0, le=255)  # HSV 0-255; debajo: acromático (blanco/gris)
    # Brillo relativo al asfalto de alrededor (ver attributes.brillo_fondo):
    # negro = V < r_negro * fondo; blanco = acromático con V >= r_blanco * fondo.
    brillo_relativo: bool = True
    r_negro: float = Field(0.6, gt=0)
    r_blanco: float = Field(1.3, gt=0)
    # umbrales absolutos: se usan si brillo_relativo=False o no se pudo medir el fondo
    v_negro: int = Field(60, ge=0, le=255)
    v_blanco: int = Field(170, ge=0, le=255)


class ConteoCfg(_Modelo):
    # banda muerta alrededor de cada línea: dentro de ±margen el lado no cambia
    margen_px: float = Field(4.0, ge=0.0)


class AlertasCfg(_Modelo):
    min_frames_historia: int = Field(15, ge=1)
    min_desplazamiento_px: float = Field(40.0, ge=0.0)
    frames_limpios: int = Field(3, ge=1)


class RutasCfg(_Modelo):
    clips: Path = Path("configs/clips.yaml")
    calibracion: Path = Path("configs/calibracion.yaml")
    outputs: Path = Path("outputs")


class AppConfig(_Modelo):
    deteccion: DeteccionCfg = DeteccionCfg()
    video: VideoCfg = VideoCfg()
    estabilizacion: EstabilizacionCfg = EstabilizacionCfg()
    metricas: MetricasCfg = MetricasCfg()
    color: ColorCfg = ColorCfg()
    conteo: ConteoCfg = ConteoCfg()
    alertas: AlertasCfg = AlertasCfg()
    rutas: RutasCfg = RutasCfg()


# --------------------------------------------------------------------------- clips.yaml


def _sin_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


class Clip(_Modelo):
    id: str = Field(min_length=1)
    archivo: Path
    fecha: dt.date
    hora_inicio: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    dia_semana: str
    tipo_dia: Literal["habil", "finde"]
    clima: str
    geometria: Path
    notas: str = ""

    @field_validator("dia_semana")
    @classmethod
    def _normalizar_dia(cls, v: str) -> str:
        dia = _sin_acentos(v.strip().lower())
        if dia not in DIAS_SEMANA:
            raise ValueError(f"dia_semana inválido: {v!r}")
        return dia

    @model_validator(mode="after")
    def _dia_coincide_con_fecha(self) -> Clip:
        esperado = DIAS_SEMANA[self.fecha.weekday()]
        if self.dia_semana != esperado:
            raise ValueError(
                f"clip {self.id}: fecha {self.fecha} es {esperado}, no {self.dia_semana}"
            )
        return self


class ClipsFile(_Modelo):
    clips: list[Clip] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_unicos(self) -> ClipsFile:
        ids = [c.id for c in self.clips]
        repetidos = sorted({i for i in ids if ids.count(i) > 1})
        if repetidos:
            raise ValueError(f"ids de clip repetidos: {repetidos}")
        return self


# --------------------------------------------------------------------------- geometria/*.yaml


class Linea(_Modelo):
    id: str = Field(min_length=1)
    puntos: tuple[Punto, Punto]
    sentido_positivo: Sentido  # qué significa cruzar de A hacia B
    # Sentido de circulación legal en esa aproximación (alerta de contramano, §4d).
    # "ambos" = calle doble mano o sin dato: no genera alertas.
    sentido_permitido: Literal["entrante", "saliente", "ambos"] = "ambos"

    @model_validator(mode="after")
    def _segmento_no_degenerado(self) -> Linea:
        if self.puntos[0] == self.puntos[1]:
            raise ValueError(f"línea {self.id}: los dos puntos son iguales")
        return self


class Geometria(_Modelo):
    encuadre: str = Field(min_length=1)
    resolucion: tuple[int, int]  # ancho, alto
    roi: list[Punto] = Field(min_length=3)
    lineas: list[Linea] = Field(min_length=1)

    @model_validator(mode="after")
    def _validar(self) -> Geometria:
        ancho, alto = self.resolucion
        puntos = list(self.roi) + [p for linea in self.lineas for p in linea.puntos]
        fuera = [p for p in puntos if not (0 <= p[0] <= ancho and 0 <= p[1] <= alto)]
        if fuera:
            raise ValueError(f"puntos fuera de la resolución {ancho}x{alto}: {fuera}")
        ids = [linea.id for linea in self.lineas]
        if len(ids) != len(set(ids)):
            raise ValueError(f"ids de línea repetidos en {self.encuadre}")
        return self


# --------------------------------------------------------------------------- calibracion.yaml


class Cortes(_Modelo):
    p33: float
    p66: float

    @model_validator(mode="after")
    def _ordenados(self) -> Cortes:
        if self.p33 > self.p66:
            raise ValueError("p33 no puede ser mayor que p66")
        return self


class Calibracion(_Modelo):
    """Umbrales de congestión calculados con los propios datos (brief §4)."""

    velocidad_p90_por_encuadre: dict[str, float]  # flujo libre de cada vista, en px/frame
    densidad: Cortes
    velocidad_norm: Cortes
    generado: dt.datetime | None = None


# --------------------------------------------------------------------------- carga


def read_yaml(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no existe el archivo de configuración: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_app_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    return AppConfig.model_validate(read_yaml(path))


def load_geometria(path: Path) -> Geometria:
    return Geometria.model_validate(read_yaml(path))


def load_calibracion(path: Path) -> Calibracion:
    return Calibracion.model_validate(read_yaml(path))


def save_calibracion(cal: Calibracion, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cal.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)
