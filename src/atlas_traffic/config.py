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
    modelo: str = "modelos/atlas_esquina_yolov8s.pt"  # yolov8s ajustado a esta esquina
    dispositivo: str = "cpu"
    confianza_min: float = Field(
        0.1, ge=0.0, le=1.0
    )  # bajo a propósito: ByteTrack re-asocia con las débiles
    iou: float = Field(0.5, ge=0.0, le=1.0)
    tracker: str = "configs/bytetrack.yaml"
    imgsz: int = Field(640, ge=32, multiple_of=32)
    # Filtro de cajas repetidas sin importar la clase: si el modelo dibuja "auto" y
    # "camión" sobre la misma camioneta, queda una sola caja (si no, se cuenta doble).
    nms_agnostico: bool = True  # lado de inferencia; subir ayuda a lo lejano

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
    ventana_velocidad_kmh_s: float = Field(0.5, gt=0)  # idem, para la velocidad en km/h


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
    # Balance de blancos con el asfalto como referencia neutra (ver attributes.balancear).
    balance_blancos: bool = True
    tinte_min: float = Field(1.15, ge=1.0)  # día: ~1.03 (p95 1.13); noche: ~1.35 (p5 1.14)
    intensidad_balance: float = Field(0.75, ge=0.0, le=1.0)


class ConteoCfg(_Modelo):
    # banda muerta alrededor de cada línea: dentro de ±margen el lado no cambia
    margen_px: float = Field(4.0, ge=0.0)
    # cruzar y volver a cruzar la misma línea en menos de esto es ruido, no un vehículo
    ida_vuelta_min_s: float = Field(1.0, ge=0.0)


class UnionCfg(_Modelo):
    """Unión de tracks cortados por una oclusión (ver stitching.py)."""

    activa: bool = True
    # a esta resolución, con huecos más largos el movimiento ya no distingue a un objeto
    # de su vecino (validado a ojo en muestra_3): mejor no unir que unir mal
    hueco_max_s: float = Field(0.5, ge=0.0)
    factor_distancia: float = Field(0.5, gt=0.0)
    # segunda pasada, solo personas en movimiento: caminando en grupo el tracker las pierde
    # 1-1.5 s justo al cruzar la línea. Para contar no hace falta saber quién es quién:
    # alcanza con emparejar a quien desaparece caminando hacia la línea con quien aparece
    # del otro lado siguiendo igual.
    peaton_hueco_max_s: float = Field(1.5, ge=0.0)
    peaton_factor_distancia: float = Field(1.5, gt=0.0)
    peaton_velocidad_min_px_s: float = Field(4.0, ge=0.0)


class AlertasCfg(_Modelo):
    min_frames_historia: int = Field(15, ge=1)
    min_desplazamiento_px: float = Field(40.0, ge=0.0)
    frames_limpios: int = Field(3, ge=1)


class CorreccionCfg(_Modelo):
    """Corrección de conteos por tasa de detección (ver correction.py)."""

    activa: bool = True
    registros: Path = Path("docs/validacion_conteo_registros.csv")  # conteos manuales
    confianza: float = Field(0.95, gt=0, lt=1)


class RenderCfg(_Modelo):
    escala: float = Field(1.5, gt=0)  # agranda el video anotado para que el texto se lea
    ancho_max: int = Field(1080, ge=64)  # tope: un video 1080p no se agranda más
    gif_inicio_s: float = Field(0.0, ge=0)
    gif_duracion_s: float = Field(6.0, gt=0)
    gif_ancho: int = Field(320, ge=64)  # chico a propósito: no deja leer patentes
    gif_fps: float = Field(8.0, gt=0)
    gif_colores: int = Field(64, ge=2, le=256)


class RutasCfg(_Modelo):
    clips: Path = Path("configs/clips.yaml")
    calibracion: Path = Path("configs/calibracion.yaml")
    outputs: Path = Path("outputs")
    suelo: Path = Path("configs/suelo")  # calibración métrica por encuadre (km/h)


class AppConfig(_Modelo):
    deteccion: DeteccionCfg = DeteccionCfg()
    video: VideoCfg = VideoCfg()
    estabilizacion: EstabilizacionCfg = EstabilizacionCfg()
    metricas: MetricasCfg = MetricasCfg()
    color: ColorCfg = ColorCfg()
    conteo: ConteoCfg = ConteoCfg()
    union: UnionCfg = UnionCfg()
    alertas: AlertasCfg = AlertasCfg()
    correccion: CorreccionCfg = CorreccionCfg()
    render: RenderCfg = RenderCfg()
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
    # día / noche: elige la tasa de detección para corregir conteos. Si se omite, se
    # detecta por el tinte de la calle (ver correction.detectar_luz).
    luz: Literal["dia", "noche"] | None = None
    # Ajustes opcionales de este clip (si se omiten, rige configs/default.yaml):
    # - frame_referencia: cuadro sobre el que se marca la geometría y contra el que se
    #   estabiliza. Útil si la cámara se acomoda al principio de la grabación.
    # - imgsz: resolución de inferencia. Con video original (no WhatsApp) 1280 detecta
    #   bastante más gente; en video comprimido no suma y es 4 veces más lento.
    # - correccion: corregir los conteos con las tasas de detección de la validación. Esas
    #   tasas se midieron con los videos de WhatsApp: no aplican a un video de otra calidad.
    correccion: bool = True
    frame_referencia: int | None = Field(None, ge=0)
    imgsz: int | None = Field(None, ge=32)

    def estabilizacion(self, base: EstabilizacionCfg) -> EstabilizacionCfg:
        if self.frame_referencia is None:
            return base
        return base.model_copy(update={"frame_referencia": self.frame_referencia})

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
    # Qué cuenta la línea. Las de calzada cuentan vehículos; las de senda peatonal,
    # personas. Por defecto, ambas (compatibilidad con geometrías viejas).
    clases: list[Literal["vehiculo", "peaton"]] = Field(
        default_factory=lambda: ["vehiculo", "peaton"], min_length=1
    )

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
    # Zonas de exclusión: objetos fijos de la escena que el detector confunde con
    # vehículos (una marca pintada, el techo de una parada). Ver geometry.filtrar_exclusiones.
    exclusiones: list[list[Punto]] = Field(default_factory=list)
    # Grupo de normalización de velocidad (congestión). Por defecto cada encuadre se
    # normaliza solo (brief §4). Encuadres distintos con la misma escala en píxeles (misma
    # cámara y posición, apenas corrida) pueden compartir grupo: así un encuadre que solo
    # tiene clips congestionados toma el flujo libre de los demás.
    grupo_normalizacion: str | None = None

    @model_validator(mode="after")
    def _validar(self) -> Geometria:
        ancho, alto = self.resolucion
        if any(len(z) < 3 for z in self.exclusiones):
            raise ValueError("cada zona de exclusión necesita al menos 3 puntos")
        puntos = list(self.roi) + [p for linea in self.lineas for p in linea.puntos]
        puntos += [p for zona in self.exclusiones for p in zona]
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
    grupos: dict[str, str] = Field(default_factory=dict)  # encuadre -> grupo de normalización
    densidad: Cortes
    velocidad_norm: Cortes
    generado: dt.datetime | None = None
    # trazabilidad: con qué datos se calibró
    clips: list[str] = Field(default_factory=list)
    n_intervalos: int = 0


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
