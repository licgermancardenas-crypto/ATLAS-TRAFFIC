"""Vehículos de interés: búsquedas guardadas que generan alertas."""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from atlas_traffic.config import (
    CamarasFile,
    InteresFile,
    VehiculoInteres,
    load_camaras,
    load_interes,
    save_interes,
)
from atlas_traffic.consulta import Datos, alertas, coincidencias, describir_interes

RAIZ = Path(__file__).resolve().parents[1]
CAMARAS = load_camaras(RAIZ / "configs" / "camaras.yaml")


def _cruces():
    filas = [
        # clip, track, hora, línea, clase, color, km/h, encuadre
        ("a", 1, "2026-11-04 19:30:05", "av_cordoba_frente", "car", "rojo", 30.0, "encuadre_m1"),
        ("a", 2, "2026-11-04 12:00:05", "av_cordoba_frente", "car", "rojo", 30.0, "encuadre_m1"),
        ("a", 3, "2026-11-04 19:40:00", "rodriguez_pena_izq", "car", "rojo", 30.0, "encuadre_m1"),
        ("b", 4, "2026-11-04 19:45:00", "av_cordoba_fondo", "motorcycle", "negro", 45.0, "otro"),
    ]
    columnas = ["clip_id", "track_id", "hora", "aproximacion", "clase_detalle", "color",
                "velocidad_kmh", "encuadre"]  # fmt: skip
    df = pd.DataFrame(filas, columns=columnas)
    df["hora"] = pd.to_datetime(df["hora"])
    df["ts_cruce"] = df["hora"].dt.second.astype(float)
    df["sentido"] = "entrante"
    df["clase"] = "vehiculo"
    df["fecha"] = pd.Timestamp("2026-11-04")
    df["thumb"] = None
    return df


ROJO_NOCHE = VehiculoInteres(id="rojo", nombre="Auto rojo de noche por Córdoba", tipos=["car"],
                             colores=["rojo"], calles=["av_cordoba"], hora_desde="19:00",
                             hora_hasta="23:59", prioridad="alta")  # fmt: skip


def test_coincide_por_tipo_color_calle_y_horario():
    m = coincidencias(_cruces(), InteresFile(vehiculos=[ROJO_NOCHE]), CAMARAS)
    # el 2 es de día y el 3 va por Rodríguez Peña
    assert list(m["track_id"]) == [1] and m["interes_nombre"].iat[0] == ROJO_NOCHE.nombre


def test_barrio_y_velocidad_y_pausa():
    rapido_en_recoleta = VehiculoInteres(id="r", nombre="x", barrios=["Recoleta"],
                                         velocidad_min_kmh=40)  # fmt: skip
    m = coincidencias(_cruces(), InteresFile(vehiculos=[rapido_en_recoleta]), CAMARAS)
    assert m.empty  # la moto va a 45 km/h, pero su encuadre no es de esta cámara
    lento = rapido_en_recoleta.model_copy(update={"velocidad_min_kmh": 0.0})
    assert len(coincidencias(_cruces(), InteresFile(vehiculos=[lento]), CAMARAS)) == 3
    pausada = ROJO_NOCHE.model_copy(update={"activa": False})
    assert coincidencias(_cruces(), InteresFile(vehiculos=[pausada]), CAMARAS).empty


def test_genera_alertas_con_la_prioridad():
    m = coincidencias(_cruces(), InteresFile(vehiculos=[ROJO_NOCHE]), CAMARAS)
    datos = Datos(summary=pd.DataFrame(), tracks=pd.DataFrame(), cruces=_cruces(),
                  eventos=pd.DataFrame(), outputs=Path("."))  # fmt: skip
    al = alertas(datos, interes=m)
    assert list(al["tipo"]) == ["Vehículo de interés"] and al["gravedad"].iat[0] == "grave"
    assert "Auto rojo de noche" in al["detalle"].iat[0]


def test_guardar_y_leer(tmp_path):
    ruta = tmp_path / "configs" / "vehiculos_interes.yaml"
    lista = InteresFile(vehiculos=[ROJO_NOCHE.model_copy(
        update={"creado": dt.datetime(2026, 11, 8, 10, 0)})])  # fmt: skip
    save_interes(lista, ruta)
    assert ruta.read_text(encoding="utf-8").startswith("# Vehículos de interés")
    assert load_interes(ruta) == lista
    assert load_interes(tmp_path / "no_existe.yaml").vehiculos == []


def test_validaciones():
    with pytest.raises(ValidationError):
        VehiculoInteres(id="x", nombre="x", hora_desde="25:00")
    with pytest.raises(ValidationError):
        InteresFile(vehiculos=[ROJO_NOCHE, ROJO_NOCHE])


def test_descripcion():
    texto = describir_interes(ROJO_NOCHE, {"av_cordoba": "Av. Córdoba"})
    assert texto == "auto · rojo · por Av. Córdoba · de 19:00 a 23:59"
    assert describir_interes(VehiculoInteres(id="x", nombre="x")) == "cualquier vehículo"


def test_sin_camaras_el_barrio_no_encuentra_nada():
    v = VehiculoInteres(id="x", nombre="x", barrios=["Palermo"])
    assert coincidencias(_cruces(), InteresFile(vehiculos=[v]), CamarasFile()).empty
