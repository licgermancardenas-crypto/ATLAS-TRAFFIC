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
# fondo (confirmado con las trayectorias). Rodríguez Peña: de derecha a izquierda mirando
# desde el balcón (observado en muestra_3 y confirmado).
PERMITIDO = {
    "av_cordoba_frente": "entrante",
    "av_cordoba_fondo": "saliente",
    "rodriguez_pena_der": "entrante",
    "rodriguez_pena_izq": "saliente",
}

# Líneas de peatones: atraviesan por la mitad la senda peatonal que cruza Córdoba del lado
# de la cámara (corte paralelo a la avenida): toda persona que cruza la avenida la corta.
SENDAS = {
    "encuadre_m1": {"senda_cordoba": ((266, 480), (314, 530))},
    "encuadre_m2": {
        "senda_cordoba": ((138, 482), (192, 528)),
        "senda_rodriguez_pena": ((259, 382), (241, 422)),  # cruza R. Peña (esquina derecha)
    },
    "encuadre_m3": {
        "senda_cordoba": ((281, 511), (319, 569)),
        "senda_cordoba_fondo": ((188, 404), (212, 436)),  # cruza Córdoba del otro lado
        # vereda de Córdoba del lado de la cámara, donde se ve caminar a la gente
        "vereda_cordoba_edificios": ((5, 545), (95, 545)),
        # ciclovía de Córdoba junto a la parada: el modelo ve a los ciclistas como personas
        # (no hay clase bicicleta), así que esta línea cuenta personas en bici
        "ciclovia_cordoba": ((185, 560), (245, 560)),
    },
}

# Los tres encuadres son la misma cámara en el mismo balcón, apenas corrida: misma escala
# en píxeles. Comparten la referencia de flujo libre para la congestión (ver
# docs/validacion_congestion.md); si no, el encuadre de la cola (sin flujo libre) queda
# sesgado hacia LOW.
GRUPO_NORMALIZACION = "balcon_cordoba_1680"

# Zonas de exclusión: objetos fijos de la escena que el detector toma por vehículos.
# Cajas medidas con las detecciones de yolov8s (centro +/- 1.15 x semiancho/semialto).
EXCLUSIONES: dict[str, list[list[tuple[int, int]]]] = {
    "encuadre_m1": [[(268, 335), (357, 335), (357, 411), (268, 411)]],  # vidriera iluminada
    "encuadre_m2": [[(200, 407), (233, 407), (233, 438), (200, 438)]],  # valla metálica blanca
    "encuadre_m3": [[(95, 550), (243, 550), (243, 701), (95, 701)]],  # techo de la parada
}

for encuadre, spec in ENCUADRES.items():
    lineas = []
    for id_, puntos in spec["lineas"].items():
        provisoria = Linea(id=id_, puntos=puntos, sentido_positivo="entrante")
        sentido = "entrante" if lado(provisoria, spec["centro"]) > 0 else "saliente"
        lineas.append(Linea(id=id_, puntos=puntos, sentido_positivo=sentido,
                            sentido_permitido=PERMITIDO.get(id_, "ambos"),
                            clases=["vehiculo"]))  # fmt: skip
    for id_, puntos in SENDAS.get(encuadre, {}).items():
        provisoria = Linea(id=id_, puntos=puntos, sentido_positivo="entrante")
        sentido = "entrante" if lado(provisoria, spec["centro"]) > 0 else "saliente"
        lineas.append(Linea(id=id_, puntos=puntos, sentido_positivo=sentido, clases=["peaton"]))
    geo = Geometria(encuadre=encuadre, resolucion=(464, 832), roi=spec["roi"], lineas=lineas,
                    exclusiones=EXCLUSIONES.get(encuadre, []),
                    grupo_normalizacion=GRUPO_NORMALIZACION)  # fmt: skip
    guardar_geometria(geo, RAIZ / "configs" / "geometria" / f"{encuadre}.yaml")
    print(encuadre, [(ln.id, ln.sentido_positivo) for ln in lineas])
