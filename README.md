# ATLAS Traffic

[![CI](https://github.com/licgermancardenas-crypto/ATLAS-TRAFFIC/actions/workflows/ci.yml/badge.svg)](https://github.com/licgermancardenas-crypto/ATLAS-TRAFFIC/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Licencia](https://img.shields.io/badge/licencia-MIT-green)

Pipeline de visión por computadora que analiza el tránsito de una bocacalle filmada desde
un balcón (Av. Córdoba y Rodríguez Peña, CABA):
- cuenta vehículos **una sola vez y por aproximación**;
- clasifica la **congestión por densidad + velocidad**;
- detecta **circulación en sentido contrario**;
- consolida varios clips en un dataset que se puede consultar vehículo por vehículo.

<p align="center">
  <img src="docs/assets/demo.gif" width="300" alt="Demo: conteo por aproximación con cola de autos (HIGH)">
  &nbsp;
  <img src="docs/assets/anotado.jpg" width="520" alt="Mismo cruce, tránsito fluido (LOW) y congestionado (HIGH)">
</p>

## Qué lo diferencia

Detectar autos lo hace cualquier tutorial. Esto resuelve cuatro cosas más:

1. **Conteo único, direccional y por aproximación.** Cada brazo del cruce tiene su línea
   virtual con sentido. El flujo se reporta por movimiento ("entran por Córdoba", "salen por
   el fondo") y no como un total.
2. **Congestión = densidad + velocidad, no volumen.** Mucho flujo rápido es tránsito fluido;
   mucha densidad lenta es congestión. Los umbrales salen de los propios datos
   (percentiles 33 y 66), no de números inventados.
3. **Comparabilidad entre clips.** Cada corrida alimenta un dataset con la fecha, el horario
   y el tipo de día de cada clip.
4. **Atributos consultables por vehículo.** Tipo, color, movimiento, momento del cruce y
   recorte de evidencia: "autos blancos que entraron por Córdoba entre los segundos 10 y 30"
   se responde en tres clics y se verifica mirando los recortes.

Además:
- **Usa un detector ajustado a esta esquina.** Es `yolov8s` reentrenado con imágenes de
  la propia esquina (fine-tuning): detecta el 88 % de los vehículos contra el 81 % del
  modelo general, y además motos. Ver [docs/entrenamiento.md](docs/entrenamiento.md).
- **Arma la matriz de giros:** por dónde entra y por dónde sale cada vehículo, con el
  porcentaje de cada giro y su margen de error. Ver [docs/giros.md](docs/giros.md).
- **Une los seguimientos cortados.** Cuando algo tapa un objeto un instante, o una
  persona se pierde en un grupo al cruzar una línea, los dos pedazos se unen para que
  el cruce cuente (`src/atlas_traffic/stitching.py`).
- **Compensa el movimiento de la cámara.** Los clips están filmados con el teléfono en
la mano y la vista se corre hasta 130 px. Cada frame se alinea con uno de referencia por
homografía, y el conteo trabaja en esas coordenadas fijas.

## Dashboard

| Panorama | Consulta |
|---|---|
| ![Panorama](docs/assets/dashboard_panorama.png) | ![Consulta: autos blancos que entraron por Córdoba entre los segundos 10 y 30](docs/assets/dashboard_consulta.png) |

- **Panorama:** flujo, densidad y velocidad por intervalo; hábil contra fin de semana;
  congestión por franja horaria; tabla de clips.
- **Video:** el video procesado de cada clip, con las cajas, las líneas, los contadores y
  el nivel de congestión en pantalla, y una leyenda que explica qué es cada cosa. Es una
  grabación ya procesada, no una transmisión en vivo.
- **Eventos:** cada alerta de sentido contrario con su recorte al lado.
- **Consulta:** filtros por clip, fechas, ventana dentro del video, tipo, color, movimiento
  y confianza de color. Muestra el conteo de vehículos únicos, la grilla de recortes y la
  descarga a CSV. Cualquier consulta se comparte por URL, por ejemplo
  `?vista=consulta&color=blanco&desde=10&hasta=30`.

**Online:** https://atlas-traffic.streamlit.app — el dashboard se publica en Streamlit
Community Cloud desde este repo
(`dashboard/app.py`, dependencias en `dashboard/requirements.txt`). Muestra los datos de
`demo/outputs/`, que se actualizan con `python scripts/publicar_demo.py` después de
`atlas-traffic batch`. Al subir a `main`, la app se actualiza sola.

## Instalación

```bash
git clone https://github.com/licgermancardenas-crypto/ATLAS-TRAFFIC atlas-traffic
cd atlas-traffic
uv venv --python 3.11
uv pip install -e ".[detect,dashboard,dev]"
```

| Extra | Contiene |
|---|---|
| `detect` | YOLO + ByteTrack (Ultralytics, AGPL-3.0) |
| `dashboard` | Streamlit + Altair |
| `dev` | pytest + ruff |

El núcleo y los tests no necesitan `detect`.

**Con Docker:**

```bash
docker build -t atlas-traffic .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" -v "$PWD/configs:/app/configs" atlas-traffic batch --render
docker run --rm -p 8501:8501 -v "$PWD/outputs:/app/outputs" -v "$PWD/configs:/app/configs" atlas-traffic dashboard
```

## Uso

```bash
# 1. Registrar el clip en configs/clips.yaml y copiar el video a data/raw/
# 2. Marcar la zona de análisis y las líneas de conteo (una vez por encuadre)
atlas-traffic geometry --clip muestra_2         # editor con el mouse
atlas-traffic geometry --clip muestra_2 --ver   # dibuja lo guardado a PNG

# 3. Procesar todo el dataset de punta a punta
atlas-traffic batch --render

# 4. Explorar
atlas-traffic dashboard
```

**Otros comandos:**

| Comando | Qué hace |
|---|---|
| `atlas-traffic run --clip <id> [--render] [--debug-video]` | procesa un clip |
| `atlas-traffic run --clip <id> --reusar-detecciones` | recuenta en segundos, sin volver a correr YOLO (útil al corregir la geometría) |
| `atlas-traffic calibrate` | recalcula los umbrales de congestión |
| `... --modelo yolov8s.pt` | usa el YOLO general en vez del modelo ajustado a la esquina |

## Cómo funciona

```
video ─► Detector (protocolo) ─► tracks con ID ─► estabilización (homografía a un frame de referencia)
                                                          │
          ┌──────────────────┬───────────────────┬────────┴─────────┬──────────────────┐
          ▼                  ▼                   ▼                  ▼                  ▼
   conteo por línea   métricas por 30 s     color por voto    sentido contrario   recortes
   (cruce + sentido)  (flujo, densidad,     (HSV relativo     (3 filtros anti-    de evidencia
                       velocidad)            al asfalto)        falso positivo)
          └──────────────────┴─────────┬─────────┴──────────────────┘
                                       ▼
                 congestión (velocidad normalizada por encuadre, cortes p33/p66, matriz)
                                       ▼
                      summary / tracks / cruces / eventos ─► dashboard
```

| Módulo | Responsabilidad |
|---|---|
| `detect.py` | Protocolo `Detector` + `UltralyticsDetector` + `FakeDetector` para tests. El vocabulario COCO no sale del adaptador. |
| `stabilize.py` | Homografía por frame (ORB + RANSAC, sin las cajas de vehículos), suavizada en el tiempo. |
| `geometry.py` | Zona de análisis, líneas, zonas de exclusión y convención de lado; editor interactivo. |
| `counting.py` | Cruce con banda muerta, corte real del segmento, conteo único por línea y sentido. |
| `metrics.py` | Flujo, densidad y velocidad por ventana; velocidad por desplazamiento en ventana (robusta al temblor). |
| `attributes.py` | Color por votación y recortes. |
| `alerts.py` | Regla de sentido contrario. |
| `congestion.py` | Calibración y matriz de congestión. |
| `correction.py` | Conteos corregidos por tasa de detección, con intervalo de confianza; detección día / noche. |
| `aggregate.py` | Consolidación multi-clip. |
| `consulta.py` | Lógica del dashboard. |
| `render.py` | Video anotado y GIF. |

**Salidas por clip** (`outputs/runs/<clip>/`):
- `detecciones.parquet`, `tracks.parquet`, `cruces.parquet`, `intervalos.parquet`, `eventos.parquet`;
- `thumbs/` y `eventos/` (recortes);
- `anotado.mp4` y `muestra.gif`.

**Consolidados** (`outputs/`): `summary.parquet`, `tracks_todos.parquet`,
`cruces_todos.parquet`, `eventos_todos.parquet`.

**Configuración:**
- `configs/default.yaml`: parámetros, todos validados con Pydantic.
- `configs/clips.yaml`: registro de clips.
- `configs/geometria/*.yaml`: una geometría por encuadre (zona de análisis, líneas y
  zonas de exclusión).
- `configs/calibracion.yaml`: umbrales de congestión, versionados para auditarlos.

## Resultados y validación

Cada criterio de aceptación del brief se midió y está documentado, incluso cuando no se
cumplió del todo.

| Fase | Criterio | Resultado | Detalle |
|---|---|---|---|
| Conteo | ±10 % por línea contra conteo manual | ✅ muestra_2: −4.5 %, −5.6 % y 0 %. ⚠️ muestra_3: frente ✅, fondo ✗ (motos). ❌ noche. Cero cruces falsos en los tres clips | [validacion_conteo.md](docs/validacion_conteo.md) |
| Personas | (fuera del brief) | ⚠️ vereda 4 de 7, ciclovía ~4 de 9. Sin conteos falsos | [validacion_peatones.md](docs/validacion_peatones.md) |
| Color | saturados > 85 %, acromáticos > 65 % | ✅ día 88 % y 76 %; noche 100 % y 71 % con balance de blancos | [validacion_color.md](docs/validacion_color.md) |
| Sentido contrario | cero falsos positivos | ✅ 0 eventos en 3 clips; la prueba controlada dispara en 81 de 84 cruces | [validacion_alertas.md](docs/validacion_alertas.md) |
| Congestión | congestionado = HIGH, liviano = LOW, sin ajuste manual | ✅ con la salvedad del supuesto de flujo libre | [validacion_congestion.md](docs/validacion_congestion.md) |
| Batch | un comando reprocesa todo | ✅ 3 clips en 577 s | — |
| Dashboard | la consulta del brief en tres clics, verificable con recortes | ✅ 9 vehículos, 9 recortes | captura de arriba |

**Cómo se validó:**
- **Conteo:** el conteo de referencia no usa el detector. Se armó un barrido temporal de
  los píxeles sobre cada línea (`scripts/barrido_temporal.py`) y se contó a ciegas.
- **Color:** los umbrales se eligieron con una muestra y se midieron una sola vez sobre
  otra, disjunta.
- **Quién etiquetó:** los conteos y las etiquetas de referencia los hizo Claude (Claude
  Code) mirando imágenes, no una persona. Conviene repetirlos a mano.

## Números con margen de error

Los conteos automáticos pierden vehículos, sobre todo motos lejanas y autos oscuros de
noche, pero no inventan: en la validación no hubo ni un cruce falso. Por eso cada conteo
se corrige con la **tasa de detección** medida contra los conteos manuales
(`docs/validacion_conteo_registros.csv`), por línea y por luz (día / noche), y se reporta
con **intervalo de confianza del 95 %** (Wilson).

| Clip | Entran al cruce (contados) | Corregido | Intervalo 95 % |
|---|---|---|---|
| muestra_2 (día), 0–30 s | 72 veh/min | 72 | 72–79 |
| muestra_1 (noche) | 27 veh/min | 38 | 31–60 |

- **La métrica principal es el flujo que entra al cruce.** Cada vehículo entra una sola
  vez, y la entrada de Córdoba es la línea más cercana a la cámara, donde los vehículos se
  ven grandes.
- **Cómo leer el margen:** es angosto de día, donde la línea del frente detecta el 100 %
  de los vehículos de la validación, y ancho de noche, donde detecta ~71 % con menos datos.
- **Qué queda afuera:** las líneas sin validar no se corrigen y quedan marcadas.
- **El día o la noche** se detecta solo, por el tinte naranja del alumbrado en la calle, o
  se declara con `luz:` en `clips.yaml`.
- **Salvedad:** en estos tres clips la corrección es parcialmente circular, porque las
  tasas se midieron sobre ellos mismos. Está pensada para los clips que se procesen
  después.

## Limitaciones

- **Videos de muestra:** pasaron por WhatsApp, que los bajó a 464 × 832 y les metió
  compresión. Fueron filmados con la cámara en mano y cada uno tiene un encuadre distinto.
  El sistema lo compensa, pero los resultados van a mejorar con video original y teléfono
  fijo en la baranda (brief §8).
- **Personas:**
  - Cada línea declara qué cuenta (vehículos o personas). Hay líneas de personas sobre
    las sendas peatonales, la vereda de Córdoba y la ciclovía (el modelo ve a los
    ciclistas como personas).
  - Contra un conteo manual, el sistema detecta ~57 % de la gente que camina por la
    vereda (4 de 7) y ~45 % de los ciclistas, siempre por omisión y sin conteos falsos.
    Son personas de ~25 px que caminan en grupo. Ver
    [docs/validacion_peatones.md](docs/validacion_peatones.md).
  - Nadie cruzó la calle por las sendas durante los clips: las personas cercanas esperan
    en la esquina mientras los autos tienen verde.
- **Motos:** el modelo ajustado a la esquina empieza a detectarlas (en el fondo de
  muestra_3 recupera 2 de 5), pero las lejanas se siguen perdiendo. Subir la resolución
  de inferencia no lo resuelve: con la compresión de WhatsApp esa información ya se perdió.
- **Noche:**
  - La detección degrada: el conteo da −36 % y −22 % en las líneas de Córdoba, siempre por
    omisión, sin cruces falsos.
  - La cámara se corre y saca del cuadro parte de las líneas.
  - La luz naranja del alumbrado hacía que los autos blancos salieran "amarillo". Un
    balance de blancos local, que toma el asfalto como gris de referencia, lo corrige y
    lleva los acromáticos nocturnos de 24 % a 71 % (medición optimista; ver el documento).
  - El clip nocturno queda acá y no en el demo principal.
- **Color acromático:** blanco, plata y gris se confunden según la luz, y un blanco en
  sombra mide gris. Los colores saturados son mucho más confiables.
- **Congestión:** normalizar la velocidad por encuadre supone que cada encuadre tiene algún
  momento de flujo libre. Con un clip por encuadre no se cumple: la cola de muestra_3 hace
  que su "flujo libre" sea lento, y la clasificación termina dependiendo sobre todo de la
  densidad. Con 5 intervalos, además, los umbrales son poco estables.
- **Objetos fijos confundidos con vehículos:** una valla metálica, una vidriera iluminada
  y el techo de una parada.
  - Se resuelven con **zonas de exclusión** marcadas en la geometría de cada encuadre,
    que descartan solo tracks de vehículos (no peatones).
  - Se verificó que no le sacan ningún cruce al conteo.
  - No se usa un filtro automático de "objetos quietos", porque se llevaría puestos a los
    autos detenidos en la cola, que son justo los que importan para la congestión.
- **Datos provisorios:** la fecha y la hora de los clips son estimadas, hasta tener las
  reales.

## Privacidad y alcance

- **Fuera de alcance, a propósito:** lectura de patentes, reconocimiento facial y
  seguimiento de personas entre cámaras. Cambiarían la categoría regulatoria del proyecto
  (Ley 25.326 de Protección de Datos Personales). El sistema trabaja con conteos agregados
  y eventos de tránsito.
- **Material público:** el GIF y las capturas del README tienen baja resolución (320 a
  380 px de ancho), así que no se pueden leer patentes ni distinguir caras.
- **Videos:** `data/raw/` está fuera del repositorio.

## Licencia

- **Código propio:** MIT.
- **Ultralytics (YOLO), extra `detect`:** AGPL-3.0. Para este proyecto abierto el uso es
  conforme. Si derivara en un producto cerrado, el detector está detrás de un protocolo
  (`atlas_traffic.detect.Detector`) y se puede cambiar por un motor Apache 2.0 (RF-DETR,
  RT-DETR, YOLOX) sin tocar el resto del pipeline.

Especificación completa del proyecto: [docs/ATLAS-Traffic-Brief.pdf](docs/ATLAS-Traffic-Brief.pdf).
