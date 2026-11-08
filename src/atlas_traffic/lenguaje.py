"""Búsqueda en lenguaje natural: "auto rojo por Córdoba alrededor de las 7:30 pm".

Un intérprete de reglas, en español rioplatense, que convierte una frase en un filtro de
la consulta (consulta.Filtro) y explica qué entendió. No usa servicios externos: es
gratis, funciona sin internet y la misma frase da siempre el mismo resultado.

Qué entiende:
- tipo: auto, taxi, camioneta, camión, colectivo, moto, bicicleta, persona;
- color: rojo, blanco, negro, gris, azul, amarillo (y sinónimos: plateado, bordó...);
- lugar: calle (Córdoba, Rodríguez Peña) y barrio (los de configs/camaras.yaml);
- hora: "a las 19:30", "7:30 pm", "alrededor de las 7", "entre las 19 y las 20",
  "después de las 19", "antes de las 8", "de noche", "a la tarde";
- fecha: "el 4/11", "4 de noviembre";
- velocidad: "a más de 40 km/h", "rápido" (40 km/h o más).

Lo que no puede saber lo dice en vez de ignorarlo: marca, patente, ropa, conductas
("sospechoso"). Las bicicletas no son una clase del detector: se buscan como personas que
pasaron por la ciclovía.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from dataclasses import dataclass, field

from atlas_traffic.consulta import Filtro, encuadres_de_barrios

__all__ = ["Interpretacion", "interpretar", "EJEMPLOS"]

EJEMPLOS = [
    "auto rojo por Córdoba alrededor de las 7:30 pm",
    "busco sospechoso en bicicleta por avenida Córdoba",
    "motos en San Nicolás",
    "taxis entre las 12 y las 12:01",
    "autos blancos a más de 35 km/h",
    "colectivos de noche",
]

TIPOS = {
    "car": ["auto", "autos", "coche", "coches", "carro", "carros", "vehiculo particular",
            "camioneta", "camionetas", "taxi", "taxis", "remis", "utilitario"],
    "truck": ["camion", "camiones"],
    "bus": ["colectivo", "colectivos", "bondi", "bondis", "bus", "buses", "omnibus"],
    "motorcycle": ["moto", "motos", "motocicleta", "motocicletas", "motoquero", "motoqueros",
                   "motociclista"],
    "bici": ["bici", "bicis", "bicicleta", "bicicletas", "ciclista", "ciclistas"],
    "person": ["persona", "personas", "peaton", "peatones", "gente", "hombre", "mujer",
               "chico", "chica", "nene", "nena", "senor", "senora"],
}  # fmt: skip
NOMBRE_TIPO = {"car": "auto", "truck": "camión", "bus": "colectivo", "motorcycle": "moto",
               "bici": "bicicleta", "person": "persona"}  # fmt: skip
COLORES = {
    "rojo": ["rojo", "roja", "rojos", "rojas", "bordo", "colorado"],
    "blanco": ["blanco", "blanca", "blancos", "blancas"],
    "negro": ["negro", "negra", "negros", "negras", "oscuro", "oscura"],
    "gris": ["gris", "grises", "plateado", "plateada", "plata"],
    "azul": ["azul", "azules", "celeste", "celestes"],
    "amarillo": ["amarillo", "amarilla", "amarillos", "amarillas"],
    "verde": ["verde", "verdes"],
}
CALLES = {
    "av_cordoba": ["avenida cordoba", "av. cordoba", "av cordoba", "cordoba"],
    "rodriguez_pena": ["rodriguez pena", "rodriguez peña", "r. pena", "rodriguez"],
}
# en las líneas de conteo, la calle aparece así (av_cordoba_frente, senda_cordoba,
# ciclovia_cordoba, senda_rodriguez_pena_der...)
TOKEN_CALLE = {"av_cordoba": "cordoba", "rodriguez_pena": "rodriguez_pena"}
NOMBRE_CALLE = {"av_cordoba": "Av. Córdoba", "rodriguez_pena": "Rodríguez Peña"}
NO_SE_PUEDE = {
    "marca": ["marca", "ford", "fiat", "toyota", "peugeot", "renault", "chevrolet",
              "volkswagen", "vw", "honda", "citroen", "nissan", "audi", "bmw", "mercedes"],
    "patente": ["patente", "dominio", "chapa"],
    "ropa": ["remera", "campera", "buzo", "gorra", "gorro", "capucha", "pantalon", "mochila",
             "casco", "vestido", "jean", "zapatillas"],
    "conducta": ["sospechoso", "sospechosa", "sospechosos", "sospechosas", "ladron",
                 "robo", "delincuente", "arrebato"],
}  # fmt: skip
AVISO_NO_SE_PUEDE = {
    "marca": "La marca no se puede reconocer con la resolución de estos videos: la ignoro.",
    "patente": "Las patentes no se pueden leer con estos videos (ni se intenta): la ignoro.",
    "ropa": "El sistema no describe la ropa de las personas: lo ignoro.",
    "conducta": "El sistema no evalúa conductas (\"sospechoso\"): busca por tipo, color, "
                "lugar y hora. Los resultados son vehículos o personas que coinciden con la "
                "descripción, no sospechosos.",
}  # fmt: skip
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]  # fmt: skip
PARTES_DEL_DIA = {
    "noche": ((19, 0), (5, 59)), "madrugada": ((0, 0), (5, 59)),
    "manana": ((6, 0), (11, 59)), "mediodia": ((11, 30), (14, 0)), "tarde": ((13, 0), (19, 59)),
}  # fmt: skip
MARGEN_EXACTO_MIN = 15  # "a las 19:30" -> 19:15 a 19:45
MARGEN_APROX_MIN = 30  # "alrededor de las 19:30" -> 19:00 a 20:00
VELOCIDAD_RAPIDO = 40.0


@dataclass
class Interpretacion:
    filtro: Filtro
    entendido: list[str] = field(default_factory=list)  # qué se entendió, para mostrar
    avisos: list[str] = field(default_factory=list)  # qué no se pudo o cómo se aproximó
    # para guardar como vehículo de interés (None si no se puede: p. ej. bicicletas)
    criterios: dict | None = None


def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower().replace("ñ", "n"))
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"\bp\.?\s?m\.?(?=\W|$)", " pm", t)
    t = re.sub(r"\ba\.?\s?m\.?(?=\W|$)", " am", t)
    return " " + re.sub(r"\s+", " ", t).strip() + " "


def _hay(t: str, palabras: list[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", t) for p in palabras)


def _quitar(t: str, palabras: list[str]) -> str:
    for p in sorted(palabras, key=len, reverse=True):
        t = re.sub(rf"(?<!\w){re.escape(p)}(?!\w)", " ", t)
    return t


def _hhmm(h: int, m: int) -> dt.time:
    return dt.time(h % 24, m)


def _sumar(t: dt.time, minutos: int) -> dt.time:
    base = dt.datetime.combine(dt.date(2000, 1, 2), t) + dt.timedelta(minutes=minutos)
    return base.time()


_HORA = re.compile(
    r"(?<![\d/])(\d{1,2})(?:[:.](\d{2}))?\s*(?:hs|h|horas)?"
    r"(?:\s*(am|pm|de la manana|de la tarde|de la noche|de la madrugada))?(?![\d/])"
)


def _horas(t: str, avisos: list[str]) -> tuple[tuple[dt.time, dt.time] | None, str | None]:
    """Franja horaria pedida (desde, hasta) y su descripción."""
    # solo números que son horas: después de "a las", "entre las", etc., o con hs / : / pm
    marcadores = r"(a las|a la|las|la|a|y|eso de las|tipo|entre las|entre|de|desde las|desde|" \
                 r"despues de las|despues de|antes de las|antes de|hasta las|hasta|" \
                 r"alrededor de las|cerca de las|aprox(?:imadamente)?)"  # fmt: skip
    encontradas = []
    for m in _HORA.finditer(t):
        antes = t[: m.start()]
        es_hora = bool(m.group(2) or m.group(3)
                       or re.search(r"(hs|h|horas)\s*$", m.group(0).strip())
                       or re.search(marcadores + r"\s*$", antes))  # fmt: skip
        h = int(m.group(1))
        if not es_hora or h > 24:
            continue
        mi = int(m.group(2) or 0)
        sufijo = m.group(3) or ""
        if sufijo in ("pm", "de la tarde", "de la noche") and h < 12:
            h += 12
        elif sufijo in ("am", "de la manana", "de la madrugada") and h == 12:
            h = 0
        if not sufijo and 1 <= h <= 11:
            avisos.append(f"Tomé las {h}:{mi:02d} como de la mañana. Si era de la tarde o la "
                          f"noche, escribí {h + 12}:{mi:02d} o {h}:{mi:02d} pm.")  # fmt: skip
        encontradas.append((m.start(), _hhmm(h, mi), antes))
    if len(encontradas) >= 2:
        a, b = encontradas[0][1], encontradas[1][1]
        return (a, b), f"entre las {a:%H:%M} y las {b:%H:%M}"
    if len(encontradas) == 1:
        _, h, antes = encontradas[0]
        if re.search(r"(despues de las|despues de|desde las|desde|a partir de las)\s*$", antes):
            return (h, dt.time(23, 59)), f"después de las {h:%H:%M}"
        if re.search(r"(antes de las|antes de|hasta las|hasta)\s*$", antes):
            return (dt.time(0, 0), h), f"antes de las {h:%H:%M}"
        aprox = re.search(r"(alrededor|rededor|cerca|tipo|eso de|aprox|mas o menos)", antes[-30:])
        margen = MARGEN_APROX_MIN if aprox else MARGEN_EXACTO_MIN
        return ((_sumar(h, -margen), _sumar(h, margen)),
                f"alrededor de las {h:%H:%M} (±{margen} min)")  # fmt: skip
    for parte, ((h0, m0), (h1, m1)) in PARTES_DEL_DIA.items():
        if re.search(rf"(de|a la|al|por la|en la) {parte}\b", t) or f" {parte} " in t:
            a, b = dt.time(h0, m0), dt.time(h1, m1)
            nombre = {"manana": "mañana", "mediodia": "mediodía"}.get(parte, parte)
            return (a, b), f"a la {nombre} ({a:%H:%M} a {b:%H:%M})"
    return None, None


def _fecha(t: str, anio: int) -> tuple[tuple[dt.date, dt.date] | None, str | None]:
    m = re.search(r"(?<!\d)(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?!\d)", t)
    if m:
        a = int(m.group(3)) if m.group(3) else anio
        a = a + 2000 if a < 100 else a
        try:
            d = dt.date(a, int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None, None
        return (d, d), f"el {d:%d/%m/%Y}"
    m = re.search(r"(?<!\d)(\d{1,2}) de (" + "|".join(MESES) + r")", t)
    if m:
        try:
            d = dt.date(anio, MESES.index(m.group(2)) + 1, int(m.group(1)))
        except ValueError:
            return None, None
        return (d, d), f"el {d:%d/%m/%Y}"
    return None, None


def interpretar(
    texto: str,
    camaras=None,
    lineas_disponibles: list[str] | None = None,
    colores_disponibles: list[str] | None = None,
    anio: int | None = None,
) -> Interpretacion:
    """Frase -> filtro de la consulta, con lo que se entendió y los avisos."""
    t = _normalizar(texto)
    entendido: list[str] = []
    avisos: list[str] = []
    anio = anio or dt.date.today().year

    for clave, palabras in NO_SE_PUEDE.items():
        if _hay(t, palabras):
            avisos.append(AVISO_NO_SE_PUEDE[clave])

    # velocidad (antes que las horas: "40 km/h" no es una hora)
    vel = 0.0
    m = re.search(r"(?:a )?(?:mas de|arriba de|mayor a|por encima de|superior a|>)\s*(\d{1,3})"
                  r"\s*(?:km/?h|kmh|kilometros(?: por hora)?)?", t)  # fmt: skip
    if m:
        vel = float(m.group(1))
        t = t.replace(m.group(0), " ")
    elif _hay(t, ["rapido", "rapida", "a toda velocidad", "a alta velocidad", "picando"]):
        vel = VELOCIDAD_RAPIDO
    t = re.sub(r"\d{1,3}\s*(km/?h|kmh)", " ", t)
    if vel:
        entendido.append(f"a más de {vel:.0f} km/h")

    # tipo
    tipos = [k for k, pal in TIPOS.items() if _hay(t, pal)]
    taxi = _hay(t, ["taxi", "taxis"])
    if _hay(t, ["camioneta", "camionetas", "utilitario"]):
        avisos.append("El detector no separa camionetas de autos: busco autos.")
    bici = "bici" in tipos
    if bici:
        tipos = [x for x in tipos if x != "bici"]
        if "person" not in tipos:
            tipos.append("person")
        avisos.append("El detector no reconoce bicicletas: un ciclista aparece como persona. "
                      "Busco personas que pasaron por la ciclovía.")  # fmt: skip
    t = _quitar(t, [p for pal in TIPOS.values() for p in pal])
    if tipos:
        nombres = ["bicicleta" if bici and x == "person" else NOMBRE_TIPO[x] for x in tipos]
        entendido.append("tipo: " + " o ".join(nombres))

    # color
    colores = [k for k, pal in COLORES.items() if _hay(t, pal)]
    if taxi and not colores:
        colores = ["amarillo"]
        entendido.append("taxi: auto amarillo (los taxis de CABA son negros y amarillos)")
    if colores:
        if "person" in tipos and len(tipos) == 1:
            avisos.append("El color se mide solo en vehículos, no en personas: lo ignoro.")
            colores = []
        else:
            entendido.append("color: " + " o ".join(colores))
            sin_datos = [c for c in colores if colores_disponibles is not None
                         and c not in colores_disponibles]  # fmt: skip
            if sin_datos:
                avisos.append(f"No hay vehículos de color {', '.join(sin_datos)} en lo "
                              "procesado.")  # fmt: skip
    t = _quitar(t, [p for pal in COLORES.values() for p in pal])

    # lugar: calles y barrios
    calles = [k for k, pal in CALLES.items() if _hay(t, pal)]
    t = _quitar(t, [p for pal in CALLES.values() for p in pal])
    if calles:
        entendido.append("calle: " + " o ".join(NOMBRE_CALLE[c] for c in calles))
    barrios = []
    if camaras is not None:
        todos = sorted({b for c in camaras.camaras for b in c.barrios})
        barrios = [b for b in todos if _normalizar(b).strip() in t]
    if barrios:
        entendido.append("barrio: " + " o ".join(barrios))

    # fecha y hora
    fechas, desc_fecha = _fecha(t, anio)
    if desc_fecha:
        entendido.append("fecha: " + desc_fecha)
        t = re.sub(r"(?<!\d)\d{1,2}/\d{1,2}(?:/\d{2,4})?(?!\d)", " ", t)
        t = re.sub(r"(?<!\d)\d{1,2} de (" + "|".join(MESES) + r")", " ", t)
    franja, desc_hora = _horas(t, avisos)
    if desc_hora:
        entendido.append("hora: " + desc_hora)

    # líneas: las personas y las bicicletas se cuentan en otras líneas (sendas, veredas,
    # ciclovía), así que la calle se busca en el nombre de la línea
    aproximaciones: list[str] = []
    filtro_calles = calles
    if lineas_disponibles is not None and (calles or bici):
        lineas = list(lineas_disponibles)
        if bici:
            lineas = [x for x in lineas if "ciclovia" in x]
            if not lineas:
                avisos.append("No hay una ciclovía marcada en estas cámaras.")
                lineas = ["(ninguna)"]
        if calles:
            lineas = [x for x in lineas if any(TOKEN_CALLE[c] in x for c in calles)] or [
                "(ninguna)"
            ]
        aproximaciones, filtro_calles = lineas, []

    if not entendido:
        avisos.append("No reconocí ningún criterio: muestro todos los vehículos. Probá con "
                      "tipo, color, calle, barrio u hora.")  # fmt: skip
    filtro = Filtro(
        clases=tipos, colores=colores, calles=filtro_calles, aproximaciones=aproximaciones,
        encuadres=encuadres_de_barrios(camaras, barrios) if camaras is not None else [],
        franja=franja, fechas=fechas, velocidad_min_kmh=vel,
    )  # fmt: skip
    criterios = None
    if not bici and "person" not in tipos:
        criterios = {
            "tipos": tipos, "colores": colores, "calles": calles, "barrios": barrios,
            "hora_desde": f"{franja[0]:%H:%M}" if franja else None,
            "hora_hasta": f"{franja[1]:%H:%M}" if franja else None,
            "velocidad_min_kmh": vel,
        }  # fmt: skip
    return Interpretacion(filtro, entendido, avisos, criterios)
