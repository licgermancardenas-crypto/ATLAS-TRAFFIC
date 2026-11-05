"""Dashboard de ATLAS Traffic (Fase 7).

Uso:  atlas-traffic dashboard          (o: streamlit run dashboard/app.py -- --config X)

Lee los consolidados que genera `atlas-traffic batch` en outputs/:
summary.parquet (Panorama), eventos_todos.parquet (Eventos) y cruces_todos.parquet
(Consulta). La lógica de datos vive en atlas_traffic.consulta; acá solo hay interfaz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from atlas_traffic.config import DEFAULT_CONFIG_PATH, load_app_config
from atlas_traffic.consulta import (
    ORDEN_CONGESTION,
    Filtro,
    cargar,
    congestion_por_franja,
    consultar,
    flujo_por_tipo_dia,
    movimientos_disponibles,
    para_csv,
    parsear_movimiento,
    tabla_clips,
)

COLOR_CONGESTION = alt.Scale(domain=ORDEN_CONGESTION, range=["#2a9d8f", "#e9a23b", "#d1495b"])
COLUMNAS_GRILLA = 6


def _config_path() -> Path:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args, _ = p.parse_known_args(sys.argv[1:])
    return args.config


@st.cache_data(show_spinner=False)
def _cargar(outputs: str):
    return cargar(Path(outputs))


def _imagen(datos, thumb) -> str | None:
    if not isinstance(thumb, str):
        return None
    ruta = datos.outputs / thumb
    return str(ruta) if ruta.exists() else None


# --------------------------------------------------------------------------- pestañas


def panorama(datos) -> None:
    s = datos.summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clips", s["clip_id"].nunique())
    c2.metric("Intervalos", len(s))
    c3.metric("Vehículos que cruzaron", datos.cruces.drop_duplicates(["clip_id", "track_id"])
              .shape[0] if not datos.cruces.empty else 0)  # fmt: skip
    c4.metric("Eventos", len(datos.eventos))

    st.subheader("Flujo por intervalo")
    nombres = {
        "flujo_veh_min": "Flujo (veh/min)",
        "densidad_media": "Densidad (veh en zona)",
        "velocidad_norm": "Velocidad normalizada",
    }
    metrica = st.radio("Métrica", list(nombres), horizontal=True, format_func=nombres.get)
    base = alt.Chart(s).encode(
        x=alt.X("t_inicio_s:Q", title="segundo del clip"),
        y=alt.Y(f"{metrica}:Q", title=None),
        color=alt.Color("clip_id:N", title="clip"),
        tooltip=["clip_id", "inicio", "t_inicio_s", "flujo_veh_min", "densidad_media",
                 "velocidad_norm", "congestion"],
    )  # fmt: skip
    st.altair_chart(base.mark_line() + base.mark_point(filled=True, size=80),
                    width="stretch")  # fmt: skip

    izq, der = st.columns(2)
    with izq:
        st.subheader("Hábil vs fin de semana")
        t = flujo_por_tipo_dia(s)
        if t["tipo_dia"].nunique() < 2:
            st.caption("Por ahora solo hay clips de un tipo de día: falta grabar el otro.")
        st.altair_chart(alt.Chart(t).mark_bar().encode(
            x=alt.X("tipo_dia:N", title=None), y=alt.Y("flujo_veh_min:Q", title="veh/min (media)"),
            tooltip=list(t.columns)), width="stretch")  # fmt: skip
    with der:
        st.subheader("Congestión por franja horaria")
        f = congestion_por_franja(s)
        if f.empty:
            st.info("Sin congestión calibrada: corré `atlas-traffic batch`.")
        else:
            st.altair_chart(alt.Chart(f).mark_bar().encode(
                x=alt.X("hora_del_dia:O", title="hora"),
                y=alt.Y("proporcion:Q", title="proporción", axis=alt.Axis(format="%")),
                color=alt.Color("congestion:N", scale=COLOR_CONGESTION, title="nivel"),
                order=alt.Order("congestion:N"),
                tooltip=["hora_del_dia", "congestion", "intervalos"],
            ), width="stretch")  # fmt: skip

    st.subheader("Clips")
    tabla = tabla_clips(s).round({"flujo_veh_min": 1, "densidad_media": 1})
    st.dataframe(tabla, hide_index=True, width="stretch")


def eventos(datos) -> None:
    e = datos.eventos
    st.caption("Circulación en sentido contrario. Cada evento se verifica mirando el recorte.")
    if e.empty:
        st.success("No hay eventos de sentido contrario en los clips procesados.")
        return
    c1, c2 = st.columns(2)
    clips = c1.multiselect("Clip", sorted(e["clip_id"].unique()))
    aprox = c2.multiselect("Aproximación", sorted(e["aproximacion"].unique()))
    if clips:
        e = e[e["clip_id"].isin(clips)]
    if aprox:
        e = e[e["aproximacion"].isin(aprox)]
    st.write(f"**{len(e)}** eventos")
    for r in e.sort_values(["clip_id", "ts"]).itertuples():
        img, info = st.columns([1, 3])
        ruta = _imagen(datos, r.thumb)
        if ruta:
            img.image(ruta, width="stretch")
        info.markdown(
            f"**{r.clip_id}** · track {r.track_id} · {r.clase_detalle}  \n"
            f"{r.aproximacion} → {r.sentido} (no permitido)  \n"
            f"segundo {r.ts:.1f} · {pd.Timestamp(r.hora):%d/%m %H:%M:%S}"
        )
        st.divider()


def filtros_consulta(datos) -> Filtro:
    c = datos.cruces
    sb = st.sidebar
    sb.header("Consulta")
    clips = sb.multiselect("Clip", sorted(c["clip_id"].unique()), help="Vacío = todos")
    fechas = pd.to_datetime(c["fecha"]).dt.date
    rango_fechas = None
    if fechas.nunique() > 1:
        rango_fechas = sb.date_input("Rango de fechas", (fechas.min(), fechas.max()))
        if not isinstance(rango_fechas, tuple) or len(rango_fechas) != 2:
            rango_fechas = None
    t_max = float(c["ts_cruce"].max()) if not c.empty else 60.0
    tope = float(max(1, round(t_max + 1)))  # float: el slider no acepta tipos mezclados
    segundos = sb.slider("Ventana dentro del video (s)", 0.0, tope, (0.0, tope), step=1.0)
    clases = sb.multiselect("Tipo de vehículo", sorted(c["clase_detalle"].dropna().unique()))
    colores = sb.multiselect("Color", sorted(c["color"].dropna().unique()))
    movimientos = sb.multiselect(
        "Movimiento", movimientos_disponibles(c),
        help="Aproximación y sentido del cruce. Ej.: av_cordoba_frente · entrante",
    )  # fmt: skip
    conf = sb.slider("Confianza mínima de color", 0.0, 1.0, 0.0, 0.05)
    return Filtro(clips=clips, fechas=rango_fechas, segundos=segundos, clases=clases,
                  colores=colores, movimientos=[parsear_movimiento(m) for m in movimientos],
                  confianza_color_min=conf)  # fmt: skip


def consulta(datos, filtro: Filtro) -> None:
    if datos.cruces.empty:
        st.info("No hay cruces consolidados: corré `atlas-traffic batch`.")
        return
    res = consultar(datos.cruces, filtro)
    c1, c2 = st.columns([3, 1])
    c1.metric("Vehículos que cumplen el filtro", len(res))
    c2.download_button("Descargar CSV", para_csv(res), "consulta_atlas_traffic.csv",
                       "text/csv", width="stretch")  # fmt: skip
    if res.empty:
        st.info("Ningún vehículo cumple el filtro.")
        return
    st.caption("Recortes de evidencia (filtros en la barra lateral). Cada uno es un vehículo.")
    filas = [res.iloc[i : i + COLUMNAS_GRILLA] for i in range(0, len(res), COLUMNAS_GRILLA)]
    for fila in filas[:20]:
        cols = st.columns(COLUMNAS_GRILLA)
        for col, r in zip(cols, fila.itertuples(), strict=False):
            ruta = _imagen(datos, r.thumb)
            if ruta:
                col.image(ruta, width="stretch")
            col.caption(f"{r.clip_id} · {r.track_id}  \n{r.color} · {r.ts_cruce:.1f} s  \n"
                        f"{r.aproximacion} {r.sentido}")  # fmt: skip
    if len(filas) > 20:
        st.caption(f"Mostrando 120 de {len(res)}. El CSV tiene todos.")
    with st.expander("Tabla"):
        st.dataframe(res.drop(columns=["thumb"], errors="ignore"), hide_index=True)


# --------------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="ATLAS Traffic", layout="wide")
    st.title("ATLAS Traffic")
    cfg = load_app_config(_config_path())
    datos = _cargar(str(cfg.rutas.outputs.resolve()))
    if datos.vacio:
        st.warning("No hay datos consolidados en outputs/. Corré `atlas-traffic batch` primero.")
        st.stop()
    st.caption("Av. Córdoba y Rodríguez Peña, CABA — conteo por aproximación, congestión "
               "por densidad + velocidad, consulta por vehículo.")  # fmt: skip
    filtro = filtros_consulta(datos)
    a, b, c = st.tabs(["Panorama", "Eventos", "Consulta"])
    with a:
        panorama(datos)
    with b:
        eventos(datos)
    with c:
        consulta(datos, filtro)


main()
