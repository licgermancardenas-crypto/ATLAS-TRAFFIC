"""Geometrías de los tres encuadres (Av. Córdoba 1680 y Rodríguez Peña, CABA).

Coordenadas marcadas sobre el frame de referencia (frame 0) de cada clip, guiadas por
las trayectorias observadas. El sentido se calcula: cruzar hacia el centro del cruce
= entrante.
"""

from pathlib import Path

from atlas_traffic.config import Geometria, Linea
from atlas_traffic.geometry import guardar_geometria, lado

RAIZ = Path(__file__).resolve().parents[1]

ENCUADRES = {
    "encuadre_m1": dict(
        centro=(250, 445),
        roi=[
            (55, 380),
            (300, 355),
            (464, 370),
            (464, 820),
            (330, 820),
            (240, 590),
            (120, 525),
            (0, 525),
            (0, 420),
        ],
        lineas={
            "av_cordoba_frente": ((250, 600), (464, 552)),
            "av_cordoba_fondo": ((62, 392), (265, 365)),
            "rodriguez_pena_der": ((445, 388), (445, 475)),
            "rodriguez_pena_izq": ((60, 428), (60, 515)),
        },
    ),
    "encuadre_m2": dict(
        centro=(165, 450),
        roi=[
            (0, 330),
            (120, 335),
            (250, 365),
            (330, 375),
            (300, 450),
            (464, 520),
            (464, 810),
            (330, 810),
            (140, 580),
            (0, 565),
        ],
        lineas={
            "av_cordoba_frente": ((176, 641), (400, 523)),
            "av_cordoba_fondo": ((8, 468), (65, 335)),
            "rodriguez_pena_der": ((215, 395), (275, 430)),
        },
    ),
    "encuadre_m3": dict(
        centro=(260, 470),
        roi=[
            (95, 375),
            (290, 365),
            (464, 400),
            (464, 810),
            (330, 810),
            (240, 600),
            (150, 540),
            (0, 540),
            (0, 440),
        ],
        lineas={
            "av_cordoba_frente": ((255, 612), (464, 555)),
            "av_cordoba_fondo": ((115, 395), (290, 382)),
            "rodriguez_pena_der": ((430, 412), (430, 492)),
            "rodriguez_pena_izq": ((80, 440), (80, 532)),
        },
    ),
}

# Sentido de circulación legal (alerta de contramano). Av. Córdoba: mano única hacia el
# fondo (confirmado con las trayectorias). Rodríguez Peña: de derecha a izquierda según lo
# observado en muestra_3; PENDIENTE de confirmar.
PERMITIDO = {
    "av_cordoba_frente": "entrante",
    "av_cordoba_fondo": "saliente",
    "rodriguez_pena_der": "entrante",
    "rodriguez_pena_izq": "saliente",
}

for encuadre, spec in ENCUADRES.items():
    lineas = []
    for id_, puntos in spec["lineas"].items():
        provisoria = Linea(id=id_, puntos=puntos, sentido_positivo="entrante")
        sentido = "entrante" if lado(provisoria, spec["centro"]) > 0 else "saliente"
        lineas.append(Linea(id=id_, puntos=puntos, sentido_positivo=sentido,
                            sentido_permitido=PERMITIDO.get(id_, "ambos")))  # fmt: skip
    geo = Geometria(encuadre=encuadre, resolucion=(464, 832), roi=spec["roi"], lineas=lineas)
    guardar_geometria(geo, RAIZ / "configs" / "geometria" / f"{encuadre}.yaml")
    print(encuadre, [(ln.id, ln.sentido_positivo) for ln in lineas])
