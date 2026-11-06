# Validación con video original (sin WhatsApp)

Clip `esquina_20261105_1946`, grabado con iPhone el jueves 5/11/2026 a las 19:46, al
atardecer con el alumbrado encendido. Dura 25 s.

## Del archivo al análisis

- **Original:** 4K HEVC a 60 fps, vertical, 199 MB. Se pasó como archivo, sin WhatsApp.
- **Conversión:** a 1080 × 1920 H.264 (`data/raw/esquina_20261105_1946.mp4`). Tiene 2.3
  veces más píxeles por lado que los clips de WhatsApp (464 × 832) y no tiene artefactos
  de compresión.
- **Ajustes por clip** (`configs/clips.yaml`):
  - `frame_referencia: 600`: la cámara se acomoda los primeros ~5 s, así que se
    estabiliza contra un cuadro del tramo quieto (10 s).
  - `imgsz: 1280`: con video original, inferir a 1280 detecta 2.7 veces más personas que
    a 640 (41 contra 15 en 9 cuadros de prueba). Con WhatsApp no sumaba nada.
  - `correccion: false`: las tasas de detección se midieron con videos de WhatsApp y no
    aplican a este.
- **Geometría** (`encuadre_n1`): marcada sobre el cuadro de referencia, guiada por las
  trayectorias de la corrida. Es otra escala en píxeles, así que no comparte la
  referencia de flujo libre con los encuadres m1–m3.
- **Tiempo:** 11 minutos de detección en CPU para 25 s de video.

## Conteo de vehículos contra conteo manual

Conteo manual cuadro a cuadro, cada 0.5 s, sobre las líneas de Córdoba:

| Línea | Manual | Sistema | Instantes |
|---|---|---|---|
| Córdoba, entrada (frente) | 10 | **10** | coinciden uno a uno (2.8, 4.2, 5.5, 6.9, 6.9, 8.9, 9.1, 10.5, 14.0, 15.4 s) |
| Córdoba, salida (fondo) | 15 | **15** | coinciden uno a uno |

Con los videos de WhatsApp, la línea del fondo de día detectaba ~94 % y de noche ~83 %.
Con video original, al atardecer, dio 100 % en las dos líneas.

Detalles:
- **Fuera del conteo:** dos autos que cruzaban la salida en el primer cuarto de segundo,
  cuando ya estaban pasando la línea al arrancar el video. Quedaron afuera tanto del
  conteo manual como del automático.
- **La moto del final** llega a la línea de entrada justo cuando termina el video.

## Lo que falló

- **Un colectivo verde que cruza Rodríguez Peña** (de 20 a 24 s) no se detecta en ningún
  cuadro. Es probable que lo confundan las luces violetas del interior. Hacen falta
  ejemplos así para reentrenar el modelo.
- **La entrada por Rodríguez Peña estaba demasiado al borde.** Un auto aparecía ya
  pasada la línea. Se corrió 40 px hacia adentro y ahora cuenta.

## Personas

- Se ven claramente: cruces por las sendas y por la vereda izquierda.
- El semáforo peatonal de Córdoba se abre a los 19 s y el video termina a los 25 s.
  Caminando a ~30 px/s, casi nadie llega a la mitad de la senda, que es donde está la
  línea de conteo. Por eso hay un solo cruce contado ahí, aunque cruza mucha gente.
- Para medir personas hacen falta videos de varios minutos.

## Para la próxima grabación

1. **Teléfono fijo** en la baranda o en un trípode, sin tocarlo durante la grabación.
2. **Inclinarlo un poco hacia abajo:** hoy un tercio de la imagen es cielo.
3. **5 a 15 minutos por toma**, de día y de noche, para que los conteos, los giros y las
   personas tengan muestra suficiente.
