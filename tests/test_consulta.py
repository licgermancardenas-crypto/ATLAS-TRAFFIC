"""Fase 7: lógica de la consulta y del panorama, y el dashboard renderizando de verdad."""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
import yaml

from atlas_traffic.consulta import (
    Filtro,
    cargar,
    congestion_por_franja,
    consultar,
    flujo_por_tipo_dia,
    para_csv,
    tabla_clips,
)


def _cruces():
    filas = [
        # clip, track, ts, aproximación, sentido, clase, color, confianza
        ("a", 1, 5.0, "frente", "entrante", "car", "blanco", 0.9),
        ("a", 1, 9.0, "fondo", "saliente", "car", "blanco", 0.9),  # mismo auto, 2 cruces
        ("a", 2, 20.0, "frente", "entrante", "car", "rojo", 1.0),
        ("a", 3, 40.0, "frente", "entrante", "bus", "blanco", 0.6),
        ("b", 1, 12.0, "frente", "entrante", "car", "blanco", 0.4),  # otro clip, mismo id
    ]
    cols = ["clip_id", "track_id", "ts_cruce", "aproximacion", "sentido", "clase_detalle",
            "color", "color_confianza"]  # fmt: skip
    df = pd.DataFrame(filas, columns=cols)
    df["fecha"] = pd.Timestamp("2026-11-04")
    df.loc[df.clip_id == "b", "fecha"] = pd.Timestamp("2026-11-08")
    df["thumb"] = "runs/x.jpg"
    return df


def test_consulta_cuenta_vehiculos_unicos():
    assert len(consultar(_cruces(), Filtro())) == 4  # (a,1) cuenta una vez; (b,1) es otro


def test_consulta_criterio_del_brief():
    # "autos blancos que entraron por el frente entre los segundos 2 y 30"
    f = Filtro(clases=["car"], colores=["blanco"], aproximaciones=["frente"],
               sentidos=["entrante"], segundos=(2, 30))  # fmt: skip
    r = consultar(_cruces(), f)
    assert sorted(zip(r.clip_id, r.track_id, strict=True)) == [("a", 1), ("b", 1)]


def test_consulta_filtros_sueltos():
    c = _cruces()
    assert set(consultar(c, Filtro(clips=["b"])).clip_id) == {"b"}
    assert len(consultar(c, Filtro(confianza_color_min=0.8))) == 2
    assert len(consultar(c, Filtro(fechas=(dt.date(2026, 11, 8), dt.date(2026, 11, 8))))) == 1
    # el auto 1 cruzó el fondo saliendo: aparece aunque su primer cruce sea el frente
    r = consultar(c, Filtro(aproximaciones=["fondo"]))
    assert list(r.track_id) == [1] and r.ts_cruce.iat[0] == 9.0
    assert consultar(pd.DataFrame(), Filtro()).empty


def test_movimientos():
    from atlas_traffic.consulta import movimientos_disponibles, parsear_movimiento

    c = _cruces()
    etiquetas = movimientos_disponibles(c)
    assert etiquetas == ["fondo · saliente", "frente · entrante"]
    r = consultar(c, Filtro(movimientos=[parsear_movimiento("fondo · saliente")]))
    assert list(r.track_id) == [1]
    # el par se respeta: "frente saliente" no existe aunque haya frente y haya saliente
    assert consultar(c, Filtro(movimientos=[("frente", "saliente")])).empty


def test_csv():
    texto = para_csv(consultar(_cruces(), Filtro())).decode("utf-8")
    assert texto.splitlines()[0].startswith("clip_id,track_id")
    assert len(texto.splitlines()) == 5


def _summary():
    return pd.DataFrame({
        "clip_id": ["a", "a", "b"], "tipo_dia": ["habil", "habil", "finde"],
        "hora_del_dia": [8, 8, 7], "congestion": ["HIGH", "MODERATE", "LOW"],
        "flujo_veh_min": [60.0, 40.0, 10.0], "densidad_media": [12.0, 8.0, 1.0],
        "velocidad_norm": [0.2, 0.4, 0.9], "duracion_s": [30.0, 20.0, 30.0],
        "fecha": pd.Timestamp("2026-11-04"), "hora_inicio": ["08:00", "08:00", "07:00"],
        "dia_semana": ["miercoles", "miercoles", "domingo"], "clima": "x",
    })  # fmt: skip


def test_panorama():
    t = flujo_por_tipo_dia(_summary()).set_index("tipo_dia")
    assert t.loc["habil", "flujo_veh_min"] == 50.0 and t.loc["finde", "clips"] == 1
    f = congestion_por_franja(_summary())
    assert f[f.hora_del_dia == 8]["proporcion"].sum() == pytest.approx(1.0)
    c = tabla_clips(_summary()).set_index("clip_id")
    assert c.loc["a", "duracion_s"] == 50.0 and c.loc["b", "congestion"] == "LOW"


def test_buscador_hora_calle_y_velocidad():
    c = _cruces().assign(
        hora=pd.to_datetime(["2026-11-04 08:00:05", "2026-11-04 08:00:09",
                             "2026-11-04 19:30:00", "2026-11-04 23:50:00",
                             "2026-11-08 00:10:00"]),
        aproximacion=["av_cordoba_frente", "av_cordoba_fondo", "rodriguez_pena_izq",
                      "av_cordoba_frente", "av_cordoba_frente"],
        velocidad_kmh=[30.0, 30.0, 55.0, 20.0, None],
        encuadre=["m1", "m1", "m1", "m1", "n1"],
    )  # fmt: skip
    franja = consultar(c, Filtro(franja=(dt.time(19, 0), dt.time(20, 0))))
    assert list(franja.track_id) == [2]
    # franja que cruza la medianoche
    noche = consultar(c, Filtro(franja=(dt.time(23, 0), dt.time(1, 0))))
    assert sorted(zip(noche.clip_id, noche.track_id, strict=True)) == [("a", 3), ("b", 1)]
    assert list(consultar(c, Filtro(calles=["rodriguez_pena"])).track_id) == [2]
    assert list(consultar(c, Filtro(velocidad_min_kmh=50)).track_id) == [2]
    assert list(consultar(c, Filtro(encuadres=["n1"])).clip_id) == ["b"]


def test_alertas_unificadas():
    from atlas_traffic.consulta import Datos, alertas

    hora = pd.Timestamp("2026-11-04 08:00")
    eventos = pd.DataFrame({"clip_id": ["a"], "track_id": [3], "ts": [40.0],
                            "aproximacion": ["frente"], "sentido": ["saliente"],
                            "clase_detalle": ["bus"], "thumb": [None],
                            "hora": [hora + pd.Timedelta(seconds=40)]})  # fmt: skip
    conflictos = pd.DataFrame({"clip_id": ["a"], "track_id": [2], "track_id_peaton": [9],
                               "ts": [20.0], "pet_s": [0.6], "vel_kmh": [30.0],
                               "primero": ["peaton"], "gravedad": ["grave"],
                               "hora": [hora + pd.Timedelta(seconds=20)]})  # fmt: skip
    cruces = _cruces().assign(clase="vehiculo", hora=hora,
                              aproximacion="av_cordoba_frente")  # fmt: skip
    tracks = pd.DataFrame({"clip_id": ["a", "a"], "track_id": [1, 2],
                           "velocidad_max_kmh": [75.0, 40.0], "encuadre": ["m1", "m1"],
                           "thumb": [None, None]})  # fmt: skip
    datos = Datos(summary=pd.DataFrame(), tracks=tracks, cruces=cruces, eventos=eventos,
                  outputs=Path("."), conflictos=conflictos)  # fmt: skip
    al = alertas(datos, {"m1": {"av_cordoba": 60.0}})
    assert sorted(al["tipo"]) == ["Exceso de velocidad", "Riesgo peatonal", "Sentido contrario"]
    exceso = al[al["tipo"] == "Exceso de velocidad"].iloc[0]
    assert exceso["track_id"] == 1 and exceso["gravedad"] == "moderado"  # 15 km/h de más
    assert al["ts"].iat[0] == 40.0  # la más reciente primero
    assert len(alertas(datos)) == 2  # sin límites no hay alertas de velocidad


def test_cargar_sin_datos(tmp_path):
    assert cargar(tmp_path).vacio


# --------------------------------------------------------------------------- dashboard real

PESTANAS = ["Centro de control", "Alertas", "Buscador", "Vehículos de interés",
            "Seguir un vehículo", "Cámaras", "Estadísticas"]  # fmt: skip


def _sin_contador(etiquetas):
    # "Alertas (3)" -> "Alertas"
    return [t.split(" (")[0] for t in etiquetas]


RAIZ = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("con_datos", [True, False])
def test_dashboard_renderiza(tmp_path, monkeypatch, con_datos):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/default.yaml").write_text(
        yaml.safe_dump({"rutas": {"outputs": "out"}}), encoding="utf-8")  # fmt: skip
    out = tmp_path / "out"
    out.mkdir()
    if con_datos:
        s = _summary().assign(t_inicio_s=[0, 30, 0], inicio=pd.Timestamp("2026-11-04 08:00"))
        s.to_parquet(out / "summary.parquet")
        _cruces().assign(hora=pd.Timestamp("2026-11-04 08:00")).to_parquet(
            out / "cruces_todos.parquet")  # fmt: skip
        pd.DataFrame({"clip_id": ["a"], "track_id": [1]}).to_parquet(out / "tracks_todos.parquet")
        pd.DataFrame({"clip_id": ["a"], "track_id": [3], "ts": [40.0], "aproximacion": ["frente"],
                      "sentido": ["saliente"], "clase_detalle": ["bus"], "thumb": [None],
                      "hora": [pd.Timestamp("2026-11-04 08:00:40")]}).to_parquet(
            out / "eventos_todos.parquet")  # fmt: skip
    at = AppTest.from_file(str(RAIZ / "dashboard/app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    if con_datos:
        assert _sin_contador(t.label for t in at.tabs) == PESTANAS
        assert "Alertas (1)" in [t.label for t in at.tabs]  # el evento de contramano
        assert any(m.value == "4" for m in at.metric)  # vehículos de la consulta sin filtros
        assert not any("demostración" in i.value for i in at.info)
    else:
        # sin datos locales: usa los publicados en demo/ (como en Streamlit Cloud)
        assert any("demostración" in i.value for i in at.info)
        assert _sin_contador(t.label for t in at.tabs) == PESTANAS


def test_dashboard_sin_datos_ni_demo(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    import shutil

    from streamlit.testing.v1 import AppTest

    # copia del dashboard fuera del repo: no hay carpeta demo/ al lado
    (tmp_path / "dashboard").mkdir()
    shutil.copy(RAIZ / "dashboard/app.py", tmp_path / "dashboard/app.py")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/default.yaml").write_text(
        yaml.safe_dump({"rutas": {"outputs": "out"}}), encoding="utf-8")  # fmt: skip
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(tmp_path / "dashboard/app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("batch" in w.value for w in at.warning)


def test_dashboard_vista_directa(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/default.yaml").write_text(
        yaml.safe_dump({"rutas": {"outputs": "out"}}), encoding="utf-8")  # fmt: skip
    out = tmp_path / "out"
    out.mkdir()
    s = _summary().assign(t_inicio_s=[0, 30, 0], inicio=pd.Timestamp("2026-11-04 08:00"))
    s.to_parquet(out / "summary.parquet")
    _cruces().assign(hora=pd.Timestamp("2026-11-04 08:00")).to_parquet(out / "cruces_todos.parquet")
    at = AppTest.from_file(str(RAIZ / "dashboard/app.py"), default_timeout=60)
    at.query_params["vista"] = "consulta"
    at.query_params["color"] = "blanco"
    at.query_params["movimiento"] = "frente · entrante"
    at.query_params["desde"] = "2"
    at.query_params["hasta"] = "30"
    at.run()
    assert not at.exception, at.exception
    # "consulta" es el nombre viejo del buscador: los links compartidos siguen andando
    assert len(at.tabs) == 0 and at.header[0].value == "Buscador"
    # la consulta del criterio, armada solo desde la URL: (a,1) y (b,1)
    assert not at.error, [e.value for e in at.error]
    assert [m.value for m in at.metric if "filtro" in m.label] == ["2"]
