"""ROI, líneas de conteo y editor interactivo de puntos (brief §3b).

Convención de lado y sentido (la usan el editor, el render y el conteo):

    lado(linea, P) = cross(B - A, P - A)

Positivo de un lado de la línea, negativo del otro. Un cruce positivo es pasar de lado
negativo a lado positivo; `sentido_positivo` dice qué significa eso (entrante/saliente).
En pantalla, la flecha perpendicular que dibuja el render apunta hacia el lado positivo:
cruzar en el sentido de la flecha = sentido_positivo.

Las coordenadas viven en el sistema del frame de referencia de la estabilización.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from atlas_traffic.config import Geometria, Linea, load_geometria

__all__ = [
    "EditorGeometria",
    "Geometria",
    "dentro_roi",
    "guardar_geometria",
    "lado",
    "load_geometria",
    "normal_positiva",
]


# --------------------------------------------------------------------------- funciones puras


def lado(linea: Linea, punto: tuple[float, float] | np.ndarray) -> np.ndarray | float:
    """Producto cruzado (B - A) x (P - A). Acepta un punto o un array (N, 2)."""
    (ax, ay), (bx, by) = linea.puntos
    p = np.asarray(punto, dtype=np.float64)
    return (bx - ax) * (p[..., 1] - ay) - (by - ay) * (p[..., 0] - ax)


def normal_positiva(linea: Linea) -> np.ndarray:
    """Vector unitario perpendicular a la línea, apuntando al lado positivo."""
    (ax, ay), (bx, by) = linea.puntos
    n = np.array([-(by - ay), bx - ax], dtype=np.float64)
    return n / np.linalg.norm(n)


def dentro_roi(roi: list[tuple[int, int]], puntos: np.ndarray) -> np.ndarray:
    """Máscara booleana: qué puntos (N, 2) caen dentro (o sobre el borde) del polígono."""
    poligono = np.asarray(roi, dtype=np.float32).reshape(-1, 1, 2)
    pts = np.asarray(puntos, dtype=np.float64).reshape(-1, 2)
    return np.array(
        [cv2.pointPolygonTest(poligono, (float(x), float(y)), False) >= 0 for x, y in pts],
        dtype=bool,
    )


def guardar_geometria(geo: Geometria, ruta: Path) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = geo.model_dump(mode="json")
    with ruta.open("w", encoding="utf-8") as f:
        f.write(
            f"# Geometría del encuadre {geo.encuadre}. Generado con `atlas-traffic geometry`.\n"
        )
        f.write("# Coordenadas en el frame de referencia de la estabilización.\n")
        f.write("# Cruce positivo = en el sentido de la flecha que dibuja el render.\n")
        yaml.safe_dump(datos, f, sort_keys=False, allow_unicode=True, default_flow_style=None)


# --------------------------------------------------------------------------- editor (lógica)


@dataclass
class EditorGeometria:
    """Estado del editor, sin GUI: se puede testear simulando clics.

    Etapa "roi": cada clic agrega un vértice; `cerrar_roi()` pasa a la etapa "lineas".
    Etapa "lineas": dos clics definen A y B; `confirmar_linea(id, sentido)` la guarda.
    """

    encuadre: str
    resolucion: tuple[int, int]
    roi: list[tuple[int, int]] = field(default_factory=list)
    lineas: list[Linea] = field(default_factory=list)
    etapa: str = "roi"
    pendiente: list[tuple[int, int]] = field(default_factory=list)  # puntos de la línea en curso

    @classmethod
    def desde_geometria(cls, geo: Geometria) -> EditorGeometria:
        return cls(geo.encuadre, geo.resolucion, list(geo.roi), list(geo.lineas), "lineas")

    def clic(self, x: int, y: int) -> None:
        ancho, alto = self.resolucion
        p = (int(min(max(x, 0), ancho)), int(min(max(y, 0), alto)))
        if self.etapa == "roi":
            self.roi.append(p)
        elif len(self.pendiente) < 2 and (not self.pendiente or self.pendiente[0] != p):
            self.pendiente.append(p)

    @property
    def linea_lista(self) -> bool:
        return self.etapa == "lineas" and len(self.pendiente) == 2

    def cerrar_roi(self) -> None:
        if len(self.roi) < 3:
            raise ValueError("la ROI necesita al menos 3 puntos")
        self.etapa = "lineas"

    def confirmar_linea(self, id_: str, sentido_positivo: str) -> Linea:
        if not self.linea_lista:
            raise ValueError("marcá los dos extremos de la línea antes de confirmarla")
        if any(linea.id == id_ for linea in self.lineas):
            raise ValueError(f"ya existe una línea con id {id_!r}")
        linea = Linea(id=id_, puntos=tuple(self.pendiente), sentido_positivo=sentido_positivo)
        self.lineas.append(linea)
        self.pendiente = []
        return linea

    def deshacer(self) -> None:
        """Quita lo último: punto de línea en curso, línea, o vértice de la ROI."""
        if self.pendiente:
            self.pendiente.pop()
        elif self.etapa == "lineas" and self.lineas:
            self.lineas.pop()
        elif self.etapa == "lineas":
            self.etapa = "roi"
        elif self.roi:
            self.roi.pop()

    def reiniciar(self) -> None:
        self.roi, self.lineas, self.pendiente, self.etapa = [], [], [], "roi"

    def geometria(self) -> Geometria:
        return Geometria(
            encuadre=self.encuadre,
            resolucion=self.resolucion,
            roi=self.roi,
            lineas=self.lineas,
        )


# --------------------------------------------------------------------------- editor (GUI)

AYUDA = [
    "ROI: clic = vertice | ENTER = cerrar ROI",
    "LINEAS: 2 clics (A, B) -> nombre y sentido en la terminal",
    "z = deshacer | r = reiniciar | s = guardar | q/ESC = salir",
]


def editar_interactivo(
    frame: np.ndarray, editor: EditorGeometria, ruta: Path, alto_max: int = 900
) -> bool:
    """Abre una ventana OpenCV para marcar ROI y líneas. Devuelve True si se guardó."""
    from atlas_traffic.render import dibujar_geometria

    escala = min(1.0, alto_max / frame.shape[0])
    if frame.shape[0] < 600:  # frames chicos (ej. 464x832): agrandar para marcar con precisión
        escala = min(alto_max / frame.shape[0], 1.6)
    ventana = f"ATLAS Traffic - geometria {editor.encuadre}"
    cv2.namedWindow(ventana, cv2.WINDOW_AUTOSIZE)

    def on_mouse(evento: int, x: int, y: int, *_: object) -> None:
        if evento == cv2.EVENT_LBUTTONDOWN:
            editor.clic(round(x / escala), round(y / escala))

    cv2.setMouseCallback(ventana, on_mouse)
    print("\n".join(AYUDA))
    guardado = False
    ultimo_guardado: Geometria | None = None

    def autoguardar() -> None:
        # Guarda cada vez que la geometría es válida y cambió: el foco del teclado suele
        # quedar en la terminal después de nombrar una línea, y una "s" tecleada ahí no
        # llega a la ventana. Así cerrar la ventana nunca pierde trabajo.
        nonlocal guardado, ultimo_guardado
        try:
            geo = editor.geometria()
        except ValueError:
            return
        if geo != ultimo_guardado:
            guardar_geometria(geo, ruta)
            ultimo_guardado, guardado = geo, True
            print(f"  (guardado automático en {ruta})")

    try:
        while True:
            vista = dibujar_geometria(
                frame, editor.roi, editor.lineas, pendiente=editor.pendiente,
                roi_cerrada=editor.etapa == "lineas",
            )  # fmt: skip
            vista = cv2.resize(vista, None, fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)
            etapa = "ROI" if editor.etapa == "roi" else f"LINEAS ({len(editor.lineas)})"
            cv2.putText(vista, etapa, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(vista, etapa, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(ventana, vista)
            tecla = cv2.waitKey(30) & 0xFF
            if editor.linea_lista:
                _pedir_datos_linea(editor)
                autoguardar()
                print("  Volvé a hacer clic en la ventana para seguir marcando.")
            elif tecla in (13, 10):
                try:
                    editor.cerrar_roi()
                    print("ROI cerrada. Ahora marcá las líneas de conteo (A y B).")
                except ValueError as e:
                    print(e)
            elif tecla == ord("z"):
                editor.deshacer()
                autoguardar()
            elif tecla == ord("r"):
                editor.reiniciar()
            elif tecla == ord("s"):
                try:
                    editor.geometria()
                    autoguardar()
                    print(f"Guardado: {ruta} ({len(editor.lineas)} líneas)")
                    return True
                except ValueError as e:
                    print(f"No se puede guardar todavía: {e}")
            elif tecla in (ord("q"), 27):
                return guardado
            if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                return guardado
    finally:
        cv2.destroyWindow(ventana)


def _pedir_datos_linea(editor: EditorGeometria) -> None:
    print("\nLínea marcada. La flecha muestra el cruce positivo.")
    while True:
        id_ = input("  id de la línea (ej. av_norte; vacío = descartar): ").strip()
        if not id_:
            editor.pendiente = []
            return
        resp = input("  cruzar en el sentido de la flecha es [e]ntrante o [s]aliente? ").strip()
        sentido = {"e": "entrante", "s": "saliente"}.get(resp[:1].lower())
        if sentido is None:
            print("  respondé e o s")
            continue
        try:
            editor.confirmar_linea(id_, sentido)
            print(f"  OK: {id_} ({sentido})")
            return
        except ValueError as e:
            print(f"  {e}")
