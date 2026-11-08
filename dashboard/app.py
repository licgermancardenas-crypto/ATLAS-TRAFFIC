"""ATLAS · Centro de Monitoreo Vial: dashboard de seguridad, control y monitoreo.

Uso:  atlas-traffic dashboard          (o: streamlit run dashboard/app.py -- --config X)

Pestañas: Centro de control (estado, mapa, últimas alertas), Alertas (con evidencia),
Buscador (vehículos por tipo, color, calle, barrio y hora), Seguir un vehículo, Cámaras
y Estadísticas. Lee los consolidados que genera `atlas-traffic batch` en outputs/; la
lógica de datos vive en atlas_traffic.consulta y acá solo hay interfaz.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
DEMO = RAIZ / "demo" / "outputs"  # datos publicados (scripts/publicar_demo.py)
try:
    import atlas_traffic  # noqa: F401
except ImportError:  # Streamlit Community Cloud: el paquete no se instala, se usa src/
    sys.path.insert(0, str(RAIZ / "src"))

from atlas_traffic.config import (
    DEFAULT_CONFIG_PATH,
    EXTENSION_PLANO,
    CamarasFile,
    load_app_config,
    load_camaras,
)
from atlas_traffic.consulta import (
    ORDEN_CONGESTION,
    ORDEN_GRAVEDAD,
    Filtro,
    alertas,
    calle_de_linea,
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


@st.cache_data(show_spinner=False, ttl=30)  # 30 s: los cambios de batch se ven solos
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
        "flujo_entrante_corr_min": "Entran al cruce, corregido (veh/min)",
        "flujo_veh_min": "Atraviesan el cruce, crudo (veh/min)",
        "densidad_media": "Densidad (veh en zona)",
        "velocidad_kmh_media": "Velocidad media (km/h)",
        "velocidad_kmh_p85": "Velocidad p85 (km/h)",
        "velocidad_norm": "Velocidad normalizada",
    }
    disponibles = [m for m in nombres if m in s.columns]
    metrica = st.radio("Métrica", disponibles, horizontal=True, format_func=nombres.get)
    tooltip = [c for c in ["clip_id", "luz", "inicio", "t_inicio_s", "flujo_entrante_min",
               "flujo_entrante_corr_min", "flujo_entrante_ic_bajo", "flujo_entrante_ic_alto",
               "flujo_veh_min", "densidad_media", "velocidad_kmh_media", "velocidad_kmh_p85",
               "velocidad_norm", "congestion"]
               if c in s.columns]  # fmt: skip
    base = alt.Chart(s).encode(
        x=alt.X("t_inicio_s:Q", title="segundo del clip"),
        y=alt.Y(f"{metrica}:Q", title=None),
        color=alt.Color("clip_id:N", title="clip"),
        tooltip=tooltip,
    )
    capas = base.mark_line() + base.mark_point(filled=True, size=80)
    if metrica == "flujo_entrante_corr_min":
        # banda: intervalo de confianza del 95 % de la corrección por tasa de detección
        banda = (
            alt.Chart(s)
            .mark_errorbar(thickness=2, ticks=True)
            .encode(
                x="t_inicio_s:Q",
                y=alt.Y("flujo_entrante_ic_bajo:Q", title=None),
                y2="flujo_entrante_ic_alto:Q",
                color="clip_id:N",
            )
        )
        capas = banda + capas
        st.caption(
            "Vehículos que entran al cruce, corregidos por la tasa de detección medida en la "
            "validación (de día la línea del frente detecta ~95 %, de noche ~64 %). Las barras "
            "son el intervalo de confianza del 95 %: más anchas de noche, donde se pierde más."
        )
    st.altair_chart(capas, width="stretch")

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

    if not datos.conteos.empty and "vehiculo_corregido" in datos.conteos:
        st.subheader("Conteo por aproximación")
        st.caption("Vehículos que cruzaron cada línea: contados y corregidos por la tasa de "
                   "detección, con intervalo del 95 %. 'validada' = hay conteos manuales para "
                   "esa línea y esa luz.")  # fmt: skip
        k = datos.conteos[datos.conteos["vehiculo"] > 0]
        cols = ["clip_id", "aproximacion", "sentido", "vehiculo", "vehiculo_corregido",
                "ic_bajo", "ic_alto", "tasa_deteccion", "validada"]  # fmt: skip
        st.dataframe(k[cols].round(1), hide_index=True, width="stretch")

    velocidades(datos.tracks)

    if not datos.giros.empty:
        giros(datos.giros)

    st.subheader("Clips")
    tabla = tabla_clips(s).round({"flujo_veh_min": 1, "densidad_media": 1})
    st.dataframe(tabla, hide_index=True, width="stretch")


def velocidades(tracks: pd.DataFrame) -> None:
    if tracks.empty or "velocidad_kmh" not in tracks:
        return
    v = tracks[(tracks["clase"] == "vehiculo") & tracks["cruzo_linea"].fillna(False).astype(bool)]
    v = v.dropna(subset=["velocidad_kmh"])
    if v.empty:
        return
    st.subheader("Velocidad por vehículo")
    st.caption(
        "Velocidad de cada vehículo que cruzó el cruce, en km/h (mediana mientras se mueve). "
        "La escala en metros sale de las franjas de las sendas peatonales: 0,50 m de ancho "
        "con 0,50 m de separación, según el Manual de Señalamiento Horizontal de Vialidad "
        "Nacional. Margen estimado: ±5 %."
    )
    caja = (
        alt.Chart(v)
        .mark_boxplot(extent="min-max", size=40)
        .encode(
            x=alt.X("clip_id:N", title=None),
            y=alt.Y("velocidad_kmh:Q", title="km/h"),
            color=alt.Color("clip_id:N", legend=None),
        )
    )
    puntos = alt.Chart(v).mark_circle(size=30, opacity=0.5, color="black").encode(
        x="clip_id:N", y="velocidad_kmh:Q",
        tooltip=["clip_id", "track_id", "clase_detalle", "color",
                 alt.Tooltip("velocidad_kmh:Q", format=".0f"),
                 alt.Tooltip("velocidad_max_kmh:Q", format=".0f")],
    )  # fmt: skip
    st.altair_chart(caja + puntos, width="stretch")


CALLES = {"av_cordoba": "Córdoba", "rodriguez_pena": "Rodríguez Peña"}


def _calle(linea: str) -> str:
    base = linea.rsplit("_", 1)[0]
    return CALLES.get(base, base.replace("_", " ").title())


def giros(g: pd.DataFrame) -> None:
    from atlas_traffic.turning import RECORRIDOS_COMPLETOS, reparto_giros

    st.subheader("Giros: por dónde entra y por dónde sale cada vehículo")
    r = reparto_giros(g)
    if r.empty:
        st.info("Todavía no hay recorridos completos para armar la matriz de giros.")
        return
    completos = g["estado"].isin(RECORRIDOS_COMPLETOS).sum()
    con_origen = g["origen"].notna() & (g["estado"] != "dudoso")
    st.caption(
        f"Se ve el recorrido completo de {completos} de {int(con_origen.sum())} vehículos que "
        "entraron. El resto se pierde de vista dentro del cruce o el clip termina antes de que "
        "salgan. El porcentaje de cada giro sale de los recorridos completos (con intervalo "
        "del 95 %), y el volumen estimado aplica ese porcentaje a todos los que entraron por "
        "ese origen. Algunos recorridos se reconstruyen: el vehículo aparece ya pasada la "
        "entrada, o un objeto lo tapa y reaparece."
    )
    r = r.assign(
        movimiento=[
            f"{_calle(o)} → {'sigue por' if _calle(o) == _calle(d) else 'dobla a'} {_calle(d)}"
            for o, d in zip(r["origen"], r["destino"], strict=True)
        ],
        porcentaje=[f"{p:.0%} ({lo:.0%}–{hi:.0%})"
                    for p, lo, hi in zip(r["pct"], r["pct_inf"], r["pct_sup"], strict=True)],
    )  # fmt: skip
    tabla = r[["movimiento", "observados", "porcentaje", "entradas_origen", "volumen_estimado"]]
    st.dataframe(
        tabla.round({"volumen_estimado": 0}).rename(columns={
            "observados": "recorridos completos", "porcentaje": "% del origen (IC 95 %)",
            "entradas_origen": "entraron por ese origen", "volumen_estimado": "volumen estimado",
        }),
        hide_index=True, width="stretch",
    )  # fmt: skip


# --------------------------------------------------------------- seguir un vehículo


@st.cache_data(show_spinner=False, ttl=30)
def _parquet(ruta: str) -> pd.DataFrame:
    return pd.read_parquet(ruta)


@st.cache_data(show_spinner=False, ttl=300)
def _data_uri(ruta: str, recorte_y: tuple[float, float] | None = None) -> str:
    """Imagen como data URI. `recorte_y`: franja vertical a conservar (fracciones 0-1)."""
    import io

    from PIL import Image

    img = Image.open(ruta)
    if recorte_y is not None:
        alto = img.height
        img = img.crop((0, round(recorte_y[0] * alto), img.width, round(recorte_y[1] * alto)))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _fondo(uri: str, ex: alt.Scale, ey: alt.Scale, ancho_px: int, alto_px: int) -> alt.Chart:
    """Imagen que llena exactamente el área del gráfico (ya recortada a sus límites)."""
    cx = (ex.domain[0] + ex.domain[1]) / 2
    cy = (ey.domain[0] + ey.domain[1]) / 2
    return (
        alt.Chart(pd.DataFrame({"url": [uri], "x": [cx], "y": [cy]}))
        .mark_image(
            width=ancho_px, height=alto_px, aspect=False, align="center", baseline="middle"
        )  # fmt: skip
        .encode(
            x=alt.X("x:Q", scale=ex, axis=None), y=alt.Y("y:Q", scale=ey, axis=None), url="url:N"
        )  # fmt: skip
    )


def _elegir_vehiculo(
    tray: pd.DataFrame, carpeta: Path, clave: str, elegido: int | None
) -> int | None:
    """Recorridos de todos los vehículos sobre la imagen de la esquina: click para elegir."""
    meta = json.loads((carpeta / "referencia.json").read_text(encoding="utf-8"))
    ancho, alto = meta["ancho"], meta["alto"]
    # se muestra la zona donde hay recorridos (no el cielo ni los edificios)
    y0 = max(0.0, float(tray["cy_ref"].min()) - 0.05 * alto)
    y1 = min(float(alto), float(tray["cy_ref"].max()) + 0.05 * alto)
    escala_x = alt.Scale(domain=[0, ancho], nice=False, zero=False)
    escala_y = alt.Scale(domain=[y1, y0], nice=False, zero=False)
    pts = tray.sort_values("ts").groupby("track_id", group_keys=False).nth(slice(None, None, 3))
    sel = alt.selection_point(
        name="vehiculo", fields=["track_id"], on="click", nearest=True, empty=False
    )
    # resaltado: el vehículo clickeado o el elegido desde la lista
    activo = sel | (alt.datum.track_id == (elegido if elegido is not None else -1))

    def segun(si, no):
        return alt.when(activo).then(alt.value(si)).otherwise(alt.value(no))

    color = segun("#ff2d55", "#00d0ff")
    base = alt.Chart(pts).encode(
        x=alt.X("cx_ref:Q", scale=escala_x, axis=None),
        y=alt.Y("cy_ref:Q", scale=escala_y, axis=None),
    )
    lineas = base.mark_line().encode(
        detail="track_id:N",
        order="ts:Q",
        color=color,
        opacity=segun(1.0, 0.45),
        strokeWidth=segun(5, 2),
    )
    puntos = (
        base.mark_circle(size=40)
        .encode(
            color=color,
            opacity=segun(1.0, 0.25),
            tooltip=[
                alt.Tooltip("track_id:N", title="vehículo"),
                alt.Tooltip("ts:Q", title="segundo", format=".1f"),
            ],
        )
        .add_params(sel)
    )
    alto_px = 520
    ancho_px = int(alto_px * ancho / (y1 - y0))
    uri = _data_uri(str(carpeta / "referencia.jpg"), (y0 / alto, y1 / alto))
    fondo = _fondo(uri, escala_x, escala_y, ancho_px, alto_px)
    grafico = (fondo + lineas + puntos).properties(width=ancho_px, height=alto_px)
    evento = st.altair_chart(
        grafico, on_select="rerun", selection_mode="vehiculo", key=clave, width="content"
    )
    elegidos = (evento.selection or {}).get("vehiculo") or []
    return int(elegidos[0]["track_id"]) if elegidos else None


def _plano(t: pd.DataFrame, carpeta: Path) -> alt.Chart | None:
    """Recorrido del vehículo visto desde arriba, en metros, coloreado por velocidad."""
    from atlas_traffic.seguimiento import recorrido_suavizado

    if "x_m" not in t or not (carpeta / "plano.jpg").exists():
        return None
    (x0, x1), (y0, y1) = EXTENSION_PLANO["x"], EXTENSION_PLANO["y"]
    s = recorrido_suavizado(t)
    s = s[s["x_m"].between(x0, x1) & s["y_m"].between(y0, y1)]
    ex = alt.Scale(domain=[x0, x1], nice=False, zero=False)
    ey = alt.Scale(domain=[y0, y1], nice=False, zero=False)
    ancho_px = 380
    alto_px = int(ancho_px * (y1 - y0) / (x1 - x0))
    fondo = _fondo(_data_uri(str(carpeta / "plano.jpg")), ex, ey, ancho_px, alto_px)
    base = alt.Chart(s).encode(
        x=alt.X("x_m:Q", scale=ex, title="metros"), y=alt.Y("y_m:Q", scale=ey, title="metros")
    )
    camino = base.mark_line(color="white", strokeWidth=5, opacity=0.6).encode(order="ts:Q")
    puntos = base.mark_circle(size=55, opacity=1).encode(
        color=alt.Color(
            "vel_kmh:Q",
            title="km/h",
            scale=alt.Scale(scheme="turbo", domain=[0, 50]),
            legend=alt.Legend(orient="bottom", gradientLength=200),
        ),
        tooltip=[
            alt.Tooltip("ts:Q", title="segundo", format=".1f"),
            alt.Tooltip("vel_kmh:Q", title="km/h", format=".0f"),
        ],
    )
    return (fondo + camino + puntos).properties(width=ancho_px, height=alto_px)


def _velocidad_en_el_tiempo(t: pd.DataFrame, cruces: pd.DataFrame) -> alt.Chart | None:
    if "vel_kmh" not in t or t["vel_kmh"].isna().all():
        return None
    linea = (
        alt.Chart(t.dropna(subset=["vel_kmh"]))
        .mark_line(color="#ff2d55")
        .encode(
            x=alt.X("ts:Q", title="segundo del clip"),
            y=alt.Y("vel_kmh:Q", title="km/h"),
            tooltip=[alt.Tooltip("ts:Q", format=".1f"), alt.Tooltip("vel_kmh:Q", format=".0f")],
        )
    )
    if cruces.empty:
        return linea.properties(height=260)
    marcas = cruces.assign(linea=cruces["aproximacion"] + " (" + cruces["sentido"] + ")")
    reglas = alt.Chart(marcas).mark_rule(strokeDash=[4, 3], color="gray").encode(x="ts_cruce:Q")
    textos = (
        alt.Chart(marcas)
        .mark_text(angle=270, align="left", dx=4, dy=-4, color="gray")
        .encode(x="ts_cruce:Q", y=alt.value(250), text="linea:N")
    )
    return (linea + reglas + textos).properties(height=260)


def _etiqueta_vehiculo(tracks: pd.DataFrame, tid: int) -> str:
    r = tracks[tracks["track_id"] == tid]
    if r.empty:
        return f"#{tid}"
    r = r.iloc[0]
    v = r.get("velocidad_kmh")
    vel = f" · {v:.0f} km/h" if pd.notna(v) else ""
    color = r.get("color") if isinstance(r.get("color"), str) else "?"
    return f"#{tid} · {r.get('clase_detalle', '')} · {color}{vel}"


def seguir(datos) -> None:
    from atlas_traffic.seguimiento import ficha_vehiculo

    runs = datos.outputs / "runs"
    clips = [
        p.parent.name
        for p in sorted(runs.glob("*/trayectorias.parquet"))
        if (p.parent / "referencia.json").exists()
    ]
    if not clips:
        st.info("Todavía no hay recorridos: corré `atlas-traffic batch --render`.")
        return
    st.caption(
        "Hacé click sobre el recorrido de un vehículo para aislarlo: el video lo sigue solo a "
        "él, y abajo están su recorrido en metros y su velocidad."
    )
    pedido = st.query_params.get("clip")
    clip = st.selectbox(
        "Clip", clips, index=clips.index(pedido) if pedido in clips else 0, key="seguir_clip"
    )
    carpeta = runs / clip
    vacio = pd.DataFrame(columns=["track_id", "clase", "clip_id"])
    cruces = datos.cruces[datos.cruces["clip_id"] == clip] if not datos.cruces.empty else vacio
    con_cruce = set(cruces.loc[cruces["clase"] == "vehiculo", "track_id"].astype(int))
    tray = _parquet(str(carpeta / "trayectorias.parquet"))
    tray = tray[tray["track_id"].isin(con_cruce)]
    if tray.empty:
        st.info("Este clip no tiene vehículos que hayan cruzado una línea.")
        return
    tracks = datos.tracks[datos.tracks["clip_id"] == clip] if not datos.tracks.empty else vacio
    giros_clip = (
        datos.giros[datos.giros["clip_id"] == clip] if not datos.giros.empty else (pd.DataFrame())
    )

    clave = f"veh_{clip}"
    pedido_v = st.query_params.get("vehiculo")
    if clave not in st.session_state and pedido_v and pedido_v.isdigit():
        if int(pedido_v) in con_cruce:
            st.session_state[clave] = int(pedido_v)
    izq, der = st.columns([3, 2])
    with izq:
        clicado = _elegir_vehiculo(
            tray, carpeta, clave=f"mapa_{clip}", elegido=st.session_state.get(clave)
        )
    # un click nuevo manda; el mismo click en la siguiente recarga no pisa la lista
    if (
        clicado is not None
        and clicado in con_cruce
        and clicado != st.session_state.get(f"click_{clip}")
    ):
        st.session_state[clave] = clicado
    st.session_state[f"click_{clip}"] = clicado
    with der:
        tid = st.selectbox(
            "…o elegilo de la lista",
            sorted(con_cruce),
            key=clave,
            placeholder="Elegí un vehículo",
            **({} if clave in st.session_state else {"index": None}),
            format_func=lambda t: _etiqueta_vehiculo(tracks, t),
        )
        if tid is None:
            st.info("Hacé click sobre un recorrido de la imagen.")
            return
        st.query_params.update({"clip": clip, "vehiculo": str(tid)})
        foco = carpeta / "foco" / f"{tid}.mp4"
        if foco.exists():
            st.video(str(foco), autoplay=True, loop=True, muted=True)
        else:
            st.caption("Sin video de seguimiento para este vehículo.")

    f = ficha_vehiculo(tid, tray, tracks, cruces, giros_clip)
    titulo = f"Vehículo #{tid}" + (f" · {f.tipo}" if f.tipo else "")
    st.subheader(titulo + (f" · {f.color}" if f.color else ""))
    if f.movimiento:
        st.markdown(f"**Recorrido:** {f.movimiento}")

    def fmt(v: float, unidad: str) -> str:
        return "—" if pd.isna(v) else f"{v:.0f} {unidad}"

    m = st.columns(5)
    m[0].metric("Velocidad (mediana)", fmt(f.velocidad_kmh, "km/h"))
    m[1].metric("Máxima", fmt(f.velocidad_max_kmh, "km/h"))
    m[2].metric("En escena", fmt(f.en_escena_s, "s"))
    m[3].metric("Detenido", fmt(f.detenido_s, "s"))
    m[4].metric("Recorrió", fmt(f.distancia_m, "m"))
    t = tray[tray["track_id"] == tid].sort_values("ts")
    a, b = st.columns([2, 3])
    with a:
        plano = _plano(t, carpeta)
        if plano is not None:
            st.caption("Recorrido visto desde arriba, en metros (color = velocidad).")
            st.altair_chart(plano, width="content")
        foto = _imagen(datos, f.thumb)
        if foto:
            st.image(foto, caption="Recorte del vehículo", width=160)
    with b:
        grafico = _velocidad_en_el_tiempo(t, f.cruces)
        if grafico is not None:
            st.caption("Velocidad en cada momento; las líneas punteadas son los cruces.")
            st.altair_chart(grafico, width="stretch")
        if not f.cruces.empty:
            tabla = f.cruces[["aproximacion", "sentido", "ts_cruce"]].round(1)
            st.dataframe(
                tabla.rename(columns={"aproximacion": "línea", "ts_cruce": "segundo"}),
                hide_index=True,
                width="stretch",
            )


LEYENDA_VIDEO = """
**Qué se ve en pantalla**

- **Cajas naranjas:** vehículos detectados, con su número de seguimiento. **Azules:** peatones.
- **Cajas verdes:** vehículos que ya cruzaron una línea y fueron contados.
- **Líneas amarillas:** líneas de conteo de cada aproximación. La **flecha blanca** marca el
  sentido de circulación permitido.
- **Zona verde:** zona de análisis. **Zonas grises rayadas:** objetos fijos que el detector
  confunde con vehículos (una valla, una vidriera) y se ignoran.
- **Panel de arriba:** cruces acumulados por aproximación y nivel de congestión del
  intervalo (LOW, MODERATE o HIGH) con la densidad.
- **Borde rojo y "SENTIDO CONTRARIO":** alerta de circulación a contramano.

La cámara se mueve porque los clips se filmaron con el teléfono en la mano: el sistema
compensa ese movimiento y por eso las líneas acompañan a la calle.
"""


def video(datos) -> None:
    videos = {r.parent.name: r for r in sorted((datos.outputs / "runs").glob("*/video_web.mp4"))}
    if not videos:
        st.info("No hay videos procesados: corré `atlas-traffic batch --render`.")
        return
    st.caption("Video ya procesado (no es una transmisión en vivo): así ve el sistema la "
               "esquina mientras cuenta.")  # fmt: skip
    clip = st.selectbox("Clip", list(videos))
    reproductor, leyenda = st.columns([2, 3])
    with reproductor:
        st.video(str(videos[clip]))
    with leyenda:
        s = datos.summary[datos.summary["clip_id"] == clip]
        if not s.empty:
            fila = s.iloc[0]
            luz = f" · {fila['luz']}" if "luz" in s else ""
            niveles = ", ".join(s["congestion"].dropna().unique())
            fecha = pd.Timestamp(fila["fecha"])
            st.markdown(f"**{clip}** · {fecha:%d/%m/%Y} {fila['hora_inicio']}{luz} · "
                        f"{fila['clima']}")  # fmt: skip
            st.markdown(f"Congestión: **{niveles}**")
        st.markdown(LEYENDA_VIDEO)


# --------------------------------------------------------------------------- estilo

ESTILO = """
<style>
.block-container {padding-top: 3.4rem;}
[data-testid="stMetricValue"] {font-size: 1.75rem;}
[class*="st-key-ficha"] img {height: 120px; object-fit: cover; border-radius: 6px;}
.atlas-header {display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;
  gap:10px; background:linear-gradient(90deg,#0f1b30,#0a101c); border:1px solid #1f2d44;
  border-radius:12px; padding:14px 20px; margin-bottom:6px;}
.atlas-logo {font-weight:800; letter-spacing:.12em; font-size:1.35rem; color:#e6edf7;}
.atlas-logo span {color:#00c2ff;}
.atlas-sub {color:#8aa0bd; font-size:.82rem; letter-spacing:.04em;}
.atlas-chips {display:flex; flex-wrap:wrap; gap:6px;}
.chip {display:inline-flex; align-items:center; gap:6px; padding:4px 11px; border-radius:999px;
  background:#13213a; border:1px solid #23395c; color:#cfe0f5; font-size:.76rem;}
.dot {width:8px; height:8px; border-radius:50%; background:#2ecc71; box-shadow:0 0 8px #2ecc71;}
.dot.ambar {background:#ffb020; box-shadow:0 0 8px #ffb020;}
[data-testid="stMetric"] {background:#111a2b; border:1px solid #1f2d44; border-radius:10px;
  padding:10px 14px;}
[data-testid="stMetricLabel"] p {text-transform:uppercase; letter-spacing:.06em;
  font-size:.72rem !important; color:#8aa0bd;}
.badge {display:inline-block; padding:1px 8px; border-radius:6px; font-size:.7rem; font-weight:700;
  text-transform:uppercase; letter-spacing:.05em;}
.badge.grave {background:#ff2d55; color:#fff;}
.badge.moderado {background:#ff9f1c; color:#14181f;}
.badge.leve {background:#ffd166; color:#14181f;}
.alerta {border-left:4px solid #ff2d55; background:#111a2b; border-radius:8px; padding:8px 12px;
  margin-bottom:6px;}
.alerta.moderado {border-left-color:#ff9f1c;} .alerta.leve {border-left-color:#ffd166;}
.alerta .t {font-weight:700; color:#e6edf7;} .alerta .d {color:#9fb2cc; font-size:.82rem;}
.panel-titulo {text-transform:uppercase; letter-spacing:.1em; font-size:.74rem; color:#8aa0bd;
  margin:4px 0 6px 0;}
.ficha {font-size:.82rem; color:#cfe0f5; line-height:1.35;}
.ficha b {color:#fff;}
</style>
"""

GRAVEDAD = {"grave": "Grave", "moderado": "Moderada", "leve": "Leve"}
TIPOS_VEHICULO = {"car": "Auto", "truck": "Camión", "bus": "Colectivo", "motorcycle": "Moto",
                  "bicycle": "Bicicleta", "person": "Persona"}  # fmt: skip
SENTIDOS = {"entrante": "entra al cruce", "saliente": "sale del cruce"}


def _tipo(c) -> str:
    return TIPOS_VEHICULO.get(c, str(c))


def _hora(h) -> str:
    return "—" if pd.isna(h) else f"{pd.Timestamp(h):%d/%m/%Y %H:%M:%S}"


def _camaras(cfg) -> CamarasFile:
    for ruta in (cfg.rutas.camaras, RAIZ / "configs" / "camaras.yaml"):
        if Path(ruta).exists():
            return load_camaras(Path(ruta))
    return CamarasFile()


def _limites(camaras: CamarasFile) -> dict[str, dict[str, float]]:
    return {e: dict(c.limites_kmh) for c in camaras.camaras for e in c.encuadres}


def _nombre_calle(camaras: CamarasFile, calle: str) -> str:
    for c in camaras.camaras:
        if calle in c.calles:
            return c.calles[calle]
    return _calle(calle + "_x")


def _link_seguir(clip: str, tid) -> str:
    return f"?vista=seguir&clip={clip}&vehiculo={int(tid)}"


def encabezado(datos, camaras: CamarasFile) -> None:
    s = datos.summary
    n_clips = s["clip_id"].nunique()
    ultimo = pd.to_datetime(s["inicio"]).max() if "inicio" in s else None
    n_cam = max(1, len(camaras.camaras))
    act = f"Último registro {pd.Timestamp(ultimo):%d/%m/%Y %H:%M}" if pd.notna(ultimo) else ""
    st.html(f"""
    <div class="atlas-header">
      <div>
        <div class="atlas-logo">ATLAS <span>·</span> CENTRO DE MONITOREO VIAL</div>
        <div class="atlas-sub">Seguridad, control y análisis de tránsito con visión artificial</div>
      </div>
      <div class="atlas-chips">
        <span class="chip"><span class="dot"></span>Sistema operativo</span>
        <span class="chip">{n_cam} cámara{"s" if n_cam > 1 else ""}</span>
        <span class="chip">{n_clips} grabaciones procesadas</span>
        <span class="chip"><span class="dot ambar"></span>Modo diferido (no es en vivo)</span>
        <span class="chip">{act}</span>
      </div>
    </div>""")


# --------------------------------------------------------------------------- centro de control


def _tarjeta_alerta(a, compacta: bool = False) -> str:
    g = a["gravedad"]
    detalle = "" if compacta else f'<div class="d">{a["detalle"]}</div>'
    return (f'<div class="alerta {g}"><span class="badge {g}">{GRAVEDAD.get(g, g)}</span> '
            f'<span class="t">{a["tipo"]}</span>'
            f'<div class="d">{_hora(a["hora"])} · {a["clip_id"]} · vehículo #{int(a["track_id"])}'
            f'</div>{detalle}</div>')  # fmt: skip


def centro_de_control(datos, camaras: CamarasFile, al: pd.DataFrame) -> None:
    c = datos.cruces
    veh = c[c["clase"] == "vehiculo"] if "clase" in c else c
    pea = c[c["clase"] == "peaton"] if "clase" in c else c.iloc[0:0]
    graves = int((al["gravedad"] == "grave").sum()) if not al.empty else 0
    t = datos.tracks
    p85 = None
    if not t.empty and "velocidad_kmh" in t:
        v = t[t["clase"] == "vehiculo"] if "clase" in t else t
        v = v["velocidad_kmh"].dropna()
        p85 = float(v.quantile(0.85)) if len(v) else None
    k = st.columns(6)
    k[0].metric("Cámaras", max(1, len(camaras.camaras)),
                help=f"{datos.summary['clip_id'].nunique()} grabaciones procesadas")  # fmt: skip
    k[1].metric("Vehículos", veh.drop_duplicates(["clip_id", "track_id"]).shape[0],
                help="Vehículos que cruzaron una línea de conteo")  # fmt: skip
    k[2].metric("Peatones", pea.drop_duplicates(["clip_id", "track_id"]).shape[0],
                help="Personas que cruzaron una senda peatonal")  # fmt: skip
    k[3].metric("Alertas", len(al), help=f"{graves} graves")
    k[4].metric("km/h p85", "—" if p85 is None else f"{p85:.0f}",
                help="Velocidad que no supera el 85 % de los vehículos")  # fmt: skip
    niv = datos.summary["congestion"].dropna()
    k[5].metric("Congestión", niv.mode().iat[0] if len(niv) else "—",
                help="Nivel más frecuente en los intervalos de 30 s")  # fmt: skip

    izq, medio, der = st.columns([1.1, 1.6, 1.3])
    with izq:
        st.html('<div class="panel-titulo">Mapa de cámaras</div>')
        if camaras.camaras:
            pts = pd.DataFrame([{"lat": cam.lat, "lon": cam.lon} for cam in camaras.camaras])
            st.map(pts, latitude="lat", longitude="lon", size=25, color="#00c2ff", zoom=15,
                   height=260)  # fmt: skip
            for cam in camaras.camaras:
                lim = " · ".join(f"{cam.calles.get(calle, calle)} {v:.0f} km/h"
                                 for calle, v in cam.limites_kmh.items())  # fmt: skip
                st.html(f'<div class="ficha"><b>{cam.nombre}</b><br>{cam.direccion}<br>'
                        f'Barrios: {", ".join(cam.barrios)} · Comunas {", ".join(cam.comunas)}'
                        f'<br>Máximas: {lim}</div>')  # fmt: skip
        else:
            st.caption("Sin cámaras registradas (configs/camaras.yaml).")
    with medio:
        videos = sorted((datos.outputs / "runs").glob("*/video_web.mp4"))
        st.html('<div class="panel-titulo">Última grabación analizada</div>')
        if videos:
            orden = datos.summary.groupby("clip_id")["inicio"].min() if "inicio" in \
                datos.summary else pd.Series(dtype=object)  # fmt: skip
            por_clip = {v.parent.name: v for v in videos}
            candidatos = [cl for cl in orden.sort_values(ascending=False).index if cl in por_clip]
            clip = candidatos[0] if candidatos else next(iter(por_clip))
            st.video(str(por_clip[clip]), autoplay=True, loop=True, muted=True)
            st.caption(f"{clip} · grabación procesada, no es una transmisión en vivo")
        else:
            st.info("Sin videos procesados: `atlas-traffic batch --render`.")
    with der:
        st.html('<div class="panel-titulo">Últimas alertas</div>')
        if al.empty:
            st.success("Sin alertas en las grabaciones procesadas.")
        else:
            st.html("".join(_tarjeta_alerta(a, compacta=True) for _, a in al.head(6).iterrows()))
            st.caption("Detalle y evidencia en la pestaña Alertas.")

    st.html('<div class="panel-titulo">Actividad</div>')
    a, b = st.columns([3, 2])
    with a:
        s = datos.summary
        if "inicio" in s and "flujo_veh_min" in s:
            st.altair_chart(alt.Chart(s).mark_line(point=True).encode(
                x=alt.X("inicio:T", title="hora"), y=alt.Y("flujo_veh_min:Q", title="veh/min"),
                color=alt.Color("clip_id:N", title="grabación"),
                tooltip=["clip_id", alt.Tooltip("inicio:T", format="%d/%m %H:%M:%S"),
                         alt.Tooltip("flujo_veh_min:Q", format=".1f"), "congestion"],
            ).properties(height=230, title="Vehículos por minuto"), width="stretch")  # fmt: skip
    with b:
        if not al.empty:
            st.altair_chart(alt.Chart(al).mark_bar().encode(
                y=alt.Y("tipo:N", title=None), x=alt.X("count():Q", title="alertas"),
                color=alt.Color("gravedad:N", scale=COLOR_GRAVEDAD, title="gravedad"),
                tooltip=["tipo", "gravedad", "count()"],
            ).properties(height=230, title="Alertas por tipo"), width="stretch")  # fmt: skip
        else:
            st.caption("Sin alertas para graficar.")


COLOR_GRAVEDAD = alt.Scale(domain=["grave", "moderado", "leve"],
                           range=["#ff2d55", "#ff9f1c", "#ffd166"])  # fmt: skip


# --------------------------------------------------------------------------- alertas

EXPLICACION_ALERTAS = """
- **Riesgo peatonal:** un vehículo pasa a menos de 2 m del lugar donde está (o estuvo) un
  peatón en la calzada, con menos de 3 s de diferencia (*PET*, indicador estándar de
  seguridad vial). Grave: menos de 1 s · moderada: 1 a 2 s · leve: 2 a 3 s.
- **Senda bloqueada:** un vehículo detenido 2 s o más con al menos el 30 % de su apoyo
  sobre una senda peatonal. Grave: 15 s o más · moderada: 5 a 15 s · leve: menos de 5 s;
  un nivel más si había personas en la senda (peatones o ciclistas: el detector no
  los distingue).
- **Detención prolongada:** un vehículo detenido 20 s o más en la calzada mientras el
  tránsito de al lado circula (no es una cola). Puede ser doble fila, estacionamiento
  prohibido o ascenso y descenso: el sistema no sabe dónde está el carril de
  estacionamiento, así que hay que revisarla. No incluye colectivos (paran en la parada).
- **Sentido contrario:** un vehículo cruza una línea en el sentido no permitido.
- **Exceso de velocidad:** velocidad del vehículo (percentil 90 de su recorrido) por encima
  de la máxima de la calle (Ley 24.449: avenidas 60 km/h, calles 40 km/h). Margen ±5 %.

Cada alerta guarda su evidencia: foto y video corto con el vehículo marcado en rojo y el
peatón en amarillo. Las alertas se revisan a ojo antes de actuar.
"""


def alertas_vista(datos, al: pd.DataFrame) -> None:
    with st.expander("Qué genera cada alerta"):
        st.markdown(EXPLICACION_ALERTAS)
    if al.empty:
        st.success("No hay alertas en las grabaciones procesadas.")
        return
    f1, f2, f3 = st.columns(3)
    tipos = f1.multiselect("Tipo", sorted(al["tipo"].unique()), placeholder="Todos")
    grav = f2.multiselect("Gravedad", [g for g in ORDEN_GRAVEDAD if g in set(al["gravedad"])],
                          format_func=GRAVEDAD.get, placeholder="Todas")  # fmt: skip
    clips = f3.multiselect("Grabación", sorted(al["clip_id"].unique()), placeholder="Todas")
    a = al
    if tipos:
        a = a[a["tipo"].isin(tipos)]
    if grav:
        a = a[a["gravedad"].isin(grav)]
    if clips:
        a = a[a["clip_id"].isin(clips)]
    m = st.columns(4)
    m[0].metric("Alertas", len(a))
    for col, g in zip(m[1:], ORDEN_GRAVEDAD, strict=True):
        col.metric(GRAVEDAD[g], int((a["gravedad"] == g).sum()))
    if a.empty:
        st.info("Ninguna alerta cumple el filtro.")
        return
    a = a.reset_index(drop=True)
    lista, evidencia = st.columns([3, 2])
    clave = "alerta_elegida"
    if st.session_state.get(clave, 0) >= len(a):
        st.session_state[clave] = 0
    with lista:
        for i, r in a.head(50).iterrows():
            foto, texto, boton = st.columns([1, 4, 1])
            ruta = _imagen(datos, r["thumb"])
            if ruta:
                foto.image(ruta, width="stretch")
            texto.html(_tarjeta_alerta(r))
            if boton.button("Ver", key=f"ver_alerta_{i}", width="stretch"):
                st.session_state[clave] = i
    with evidencia:
        r = a.iloc[st.session_state.get(clave, 0)]
        st.html('<div class="panel-titulo">Evidencia</div>')
        st.html(_tarjeta_alerta(r))
        video = datos.outputs / r["video"] if isinstance(r["video"], str) else None
        if video is not None and video.exists():
            st.video(str(video), autoplay=True, loop=True, muted=True)
        else:
            ruta = _imagen(datos, r["thumb"])
            if ruta:
                st.image(ruta, width="stretch")
        st.markdown(f"[Seguir a este vehículo →]({_link_seguir(r['clip_id'], r['track_id'])})")


# --------------------------------------------------------------------------- buscador


def _de_url(clave: str, opciones: list[str]) -> list[str]:
    """Valores iniciales de un filtro tomados de la URL (?color=blanco&color=rojo).

    Permite compartir una búsqueda con un link. Valores que no existen se ignoran.
    """
    return [v for v in st.query_params.get_all(clave) if v in opciones]


def _numero_de_url(clave: str, defecto: float) -> float:
    try:
        return float(st.query_params.get(clave, defecto))
    except ValueError:
        return defecto


def _hora_de_url(clave: str, defecto: dt.time) -> dt.time:
    try:
        return dt.time.fromisoformat(st.query_params.get(clave, ""))
    except ValueError:
        return defecto


def filtros_busqueda(datos, camaras: CamarasFile) -> Filtro:
    c = datos.cruces
    st.caption(
        "Buscá vehículos por cómo son y por dónde y cuándo pasaron. Los criterios se pueden "
        "pasar por URL para compartir una búsqueda: "
        "`?vista=buscador&color=blanco&tipo=car&hora_desde=19:00&hora_hasta=20:00`"
    )

    def multi(col, etiqueta, clave, opciones, **kw):
        return col.multiselect(etiqueta, opciones, default=_de_url(clave, opciones),
                               placeholder="Todos", key=f"b_{clave}", **kw)  # fmt: skip

    with st.container(border=True):
        st.html('<div class="panel-titulo">Vehículo</div>')
        a, b, d = st.columns(3)
        tipos = multi(a, "Tipo", "tipo", sorted(c["clase_detalle"].dropna().unique()),
                      format_func=_tipo)  # fmt: skip
        colores = multi(b, "Color", "color", sorted(c["color"].dropna().unique()))
        vel_min = d.number_input("Velocidad mínima (km/h)", 0, 150,
                                 int(_numero_de_url("vel", 0)), step=5, key="b_vel",
                                 help="0 = cualquiera")  # fmt: skip
        st.html('<div class="panel-titulo">Lugar y momento</div>')
        a, b, d, e = st.columns(4)
        es_veh = c["clase"] == "vehiculo" if "clase" in c else pd.Series(True, index=c.index)
        lineas_veh = c.loc[es_veh, "aproximacion"].dropna()
        calles_disp = sorted({calle_de_linea(x) for x in lineas_veh})
        calles = multi(a, "Calle", "calle", calles_disp,
                       format_func=lambda x: _nombre_calle(camaras, x))  # fmt: skip
        barrios_disp = sorted({br for cam in camaras.camaras for br in cam.barrios})
        barrios = multi(b, "Barrio", "barrio", barrios_disp)
        h_desde = d.time_input("Desde (hora)", _hora_de_url("hora_desde", dt.time(0, 0)),
                               key="b_hora_desde", step=60)  # fmt: skip
        h_hasta = e.time_input("Hasta (hora)", _hora_de_url("hora_hasta", dt.time(23, 59)),
                               key="b_hora_hasta", step=60)  # fmt: skip
        fechas = pd.to_datetime(c["fecha"]).dt.date
        rango_fechas = None
        if fechas.nunique() > 1:
            rango_fechas = st.date_input("Fechas", (fechas.min(), fechas.max()), key="b_fechas")
            if not isinstance(rango_fechas, tuple) or len(rango_fechas) != 2:
                rango_fechas = None
        with st.expander("Más criterios"):
            x, y = st.columns(2)
            clips = multi(x, "Grabación", "clip", sorted(c["clip_id"].unique()))
            movimientos = multi(
                y, "Línea y sentido", "movimiento", movimientos_disponibles(c),
                help="Aproximación y sentido del cruce. Ej.: av_cordoba_frente · entrante",
            )  # fmt: skip
            t_max = float(c["ts_cruce"].max()) if not c.empty else 60.0
            tope = float(max(1, round(t_max + 1)))  # float: el slider no acepta tipos mezclados
            desde = min(max(0.0, _numero_de_url("desde", 0.0)), tope)
            hasta = min(max(desde, _numero_de_url("hasta", tope)), tope)
            segundos = x.slider("Segundo dentro de la grabación", 0.0, tope, (desde, hasta),
                                step=1.0, key="b_segundos")  # fmt: skip
            conf = y.slider("Confianza mínima de color", 0.0, 1.0,
                            min(1.0, max(0.0, _numero_de_url("confianza", 0.0))), 0.05,
                            key="b_confianza")  # fmt: skip
    encuadres = [e for cam in camaras.camaras if set(cam.barrios) & set(barrios)
                 for e in cam.encuadres]  # fmt: skip
    if barrios and not encuadres:
        encuadres = ["(ninguno)"]
    franja = None if (h_desde, h_hasta) == (dt.time(0, 0), dt.time(23, 59)) else (h_desde, h_hasta)
    return Filtro(clips=clips, fechas=rango_fechas, segundos=segundos, clases=tipos,
                  colores=colores, movimientos=[parsear_movimiento(m) for m in movimientos],
                  confianza_color_min=conf, franja=franja, calles=calles, encuadres=encuadres,
                  velocidad_min_kmh=float(vel_min))  # fmt: skip


def buscador(datos, camaras: CamarasFile) -> None:
    if datos.cruces.empty:
        st.info("No hay cruces consolidados: corré `atlas-traffic batch`.")
        return
    filtro = filtros_busqueda(datos, camaras)
    res = consultar(datos.cruces, filtro)
    c1, c2 = st.columns([3, 1])
    c1.metric("Vehículos que cumplen el filtro", len(res))
    c2.download_button("Descargar CSV", para_csv(res), "busqueda_atlas.csv", "text/csv",
                       width="stretch")  # fmt: skip
    if res.empty:
        st.info("Ningún vehículo cumple los criterios.")
        return
    st.caption(
        "Cada ficha es un vehículo, con el primer cruce que cumple la búsqueda. La marca y "
        "la patente no se pueden leer con la resolución de estos videos."
    )
    filas = [res.iloc[i : i + COLUMNAS_GRILLA] for i in range(0, len(res), COLUMNAS_GRILLA)]
    for fila in filas[:20]:
        cols = st.columns(COLUMNAS_GRILLA)
        for col, r in zip(cols, fila.itertuples(), strict=False):
            with col.container(border=True, key=f"ficha_{r.clip_id}_{r.track_id}"):
                ruta = _imagen(datos, r.thumb)
                if ruta:
                    st.image(ruta, width="stretch")
                vel = getattr(r, "velocidad_kmh", None)
                vel = f" · {vel:.0f} km/h" if vel is not None and pd.notna(vel) else ""
                calle = _nombre_calle(camaras, calle_de_linea(r.aproximacion))
                st.html(f'<div class="ficha"><b>#{r.track_id} · {_tipo(r.clase_detalle)}</b>'
                        f' · {r.color}{vel}<br>{_hora(getattr(r, "hora", None))}<br>{calle} · '
                        f'{SENTIDOS.get(r.sentido, r.sentido)}<br>'
                        f'<a href="{_link_seguir(r.clip_id, r.track_id)}" target="_self">'
                        f'Seguir →</a></div>')  # fmt: skip
    if len(filas) > 20:
        st.caption(f"Mostrando 120 de {len(res)}. El CSV tiene todos.")
    with st.expander("Tabla"):
        st.dataframe(res.drop(columns=["thumb"], errors="ignore"), hide_index=True)


# --------------------------------------------------------------------------- main


def barra_lateral(camaras: CamarasFile) -> None:
    sb = st.sidebar
    sb.markdown("### ATLAS · Monitoreo vial")
    for cam in camaras.camaras:
        sb.markdown(f"**{cam.nombre}**  \n{cam.direccion}  \n{', '.join(cam.barrios)}")
    sb.divider()
    sb.markdown(
        "**Qué hace el sistema**\n\n"
        "- Detecta y sigue vehículos y peatones\n"
        "- Cuenta por calle y sentido, y mide velocidad en km/h\n"
        "- Genera alertas con evidencia (foto y video)\n"
        "- Permite buscar vehículos por tipo, color, calle y hora"
    )
    sb.caption(
        "Fuente: grabaciones procesadas (no es una transmisión en vivo). Los recortes son "
        "de baja resolución: no permiten leer patentes ni ver caras."
    )


def main() -> None:
    st.set_page_config(page_title="ATLAS · Monitoreo vial", page_icon="🛰️", layout="wide")
    st.html(ESTILO)
    cfg = load_app_config(_config_path())
    datos = _cargar(str(cfg.rutas.outputs.resolve()))
    demo = False
    if datos.vacio and (DEMO / "summary.parquet").exists():
        # sin datos locales (por ejemplo, en la nube): se muestran los publicados
        datos = _cargar(str(DEMO))
        demo = True
    if datos.vacio:
        st.title("ATLAS · Centro de Monitoreo Vial")
        st.warning("No hay datos consolidados en outputs/. Corré `atlas-traffic batch` primero.")
        st.stop()
    camaras = _camaras(cfg)
    encabezado(datos, camaras)
    if demo:
        st.info("Datos de demostración: tres clips de 23 a 50 s enviados por WhatsApp "
                "(464 × 832) y uno de 25 s con el video original del iPhone. Ver limitaciones "
                "en el README del repositorio.")  # fmt: skip
    barra_lateral(camaras)
    al = alertas(datos, _limites(camaras))
    vistas = {
        "control": ("Centro de control", lambda: centro_de_control(datos, camaras, al)),
        "alertas": (f"Alertas ({len(al)})", lambda: alertas_vista(datos, al)),
        "buscador": ("Buscador", lambda: buscador(datos, camaras)),
        "seguir": ("Seguir un vehículo", lambda: seguir(datos)),
        "camaras": ("Cámaras", lambda: video(datos)),
        "estadisticas": ("Estadísticas", lambda: panorama(datos)),
    }
    # nombres viejos de las vistas (links ya compartidos)
    alias = {"panorama": "estadisticas", "video": "camaras", "eventos": "alertas",
             "consulta": "buscador"}  # fmt: skip
    directa = st.query_params.get("vista")
    directa = alias.get(directa, directa)
    if directa in vistas:
        titulo, dibujar = vistas[directa]
        st.header(titulo)
        _seguro(titulo, dibujar)
        return
    for pestana, (titulo, dibujar) in zip(
        st.tabs([t for t, _ in vistas.values()]), vistas.values(), strict=True
    ):
        with pestana:
            _seguro(titulo, dibujar)


def _seguro(titulo: str, dibujar) -> None:
    """Dibuja una pestaña; si falla, muestra el error ahí y las demás siguen funcionando."""
    try:
        dibujar()
    except Exception as e:  # noqa: BLE001 - una pestaña rota no debe tirar el dashboard
        st.error(f"No se pudo mostrar {titulo}: {type(e).__name__}: {e}")


main()
