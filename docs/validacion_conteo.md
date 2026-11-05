# Validación del conteo (Fase 2)

**Criterio del brief:** el conteo automático por aproximación tiene que coincidir con un
conteo manual del clip de 20 s, con un margen de ±10 % en cada línea.

## Método

El conteo de referencia no usa el detector, para no comparar el sistema contra sí mismo.
Para cada línea se arma un **barrido temporal** (`scripts/barrido_temporal.py`): en cada
frame se toman los píxeles que caen sobre la línea (en coordenadas estabilizadas, así la
línea queda fija sobre la calle aunque la cámara se mueva) y se apilan en el tiempo. Cada
vehículo que cruza deja una mancha. El conteo se hizo **a ciegas** sobre esas imágenes y
recién después se comparó, cruce por cruce, con `cruces.parquet`. Los casos dudosos se
verificaron mirando los frames originales.

> El conteo de referencia lo hizo Claude (Claude Code) mirando las imágenes, no una
> persona. Conviene repetirlo a mano en algún clip.

**Casos de borde:**
- Los vehículos que en el primer frame ya pasaron la línea no cuentan.
- Los que la cruzan después del segundo 20 tampoco.
- Las cuñas negras del barrido son el borde de la imagen cuando la cámara se corre, no vehículos.

## Resultado — muestra_2 (día, tránsito fluido), primeros 20 s

| Línea | Manual | `yolov8n` | `yolov8s` (actual) | Falsos positivos |
|---|---|---|---|---|
| av_cordoba_frente | 22 | 20 (−9.1 %) | **21 (−4.5 %) ✓** | 0 |
| av_cordoba_fondo | 18 | 17 (−5.6 %) | **17 (−5.6 %) ✓** | 0 |
| rodriguez_pena_der | 0 | 0 | **0 ✓** | 0 |

**Cruces perdidos:**
- **Una moto**, en ambas líneas, con los dos modelos.
- **Un auto del carril izquierdo**, en el borde inferior del cuadro. `yolov8n` lo perdía y
  `yolov8s` lo recupera.

No hubo ningún cruce falso, y todos los cruces de la avenida salieron en el sentido de
circulación (Córdoba es mano única).

## Resultado — muestra_3 (día, cola de autos), primeros 20 s, `yolov8s`

| Línea | Manual | Automático | Diferencia | Falsos positivos |
|---|---|---|---|---|
| av_cordoba_frente | ~18 | 17 | ~−6 % ✓ | 0 |
| av_cordoba_fondo | 10 (5 autos + 5 motos) | 5 | **−50 % ✗** (autos: 5/5) | 0 |
| rodriguez_pena_izq | 1 | 1 | ✓ | 0 |
| rodriguez_pena_der | ¿0? | 0 | dudoso | 0 |

- **Frente:** el conteo manual es aproximado. Los autos de la cola avanzan muy lento y en
  el barrido aparecen como manchas altas.
- **Fondo:** los 5 autos (taxis) coinciden uno a uno. Los 5 que faltan son **motos**,
  confirmadas mirando el video: la línea queda lejos de la cámara y ahí miden muy pocos
  píxeles.
- **Rodríguez Peña izquierda:** el único cruce es un colectivo que está parado sobre la
  línea y avanza recién a los ~16 s.
- **Rodríguez Peña derecha:** hay una banda oscura instantánea a los 14 s que parece un
  salto de la imagen, no un vehículo.

## Resultado — muestra_1 (noche, lluvia), primeros 20 s, `yolov8s`

| Línea | Manual | Automático | Diferencia | Falsos positivos |
|---|---|---|---|---|
| av_cordoba_frente | ~14 | 9 | **~−36 % ✗** | 0 |
| av_cordoba_fondo | ~23 | 18 | **~−22 % ✗** | 0 |
| rodriguez_pena_izq | 2 | 1 | ✗ (n muy chico) | 0 |
| rodriguez_pena_der | — | 0 | no medible | — |

- **Precisión intacta:** todos los cruces automáticos corresponden a vehículos reales.
- **Lo que se pierde:**
  - Autos oscuros de noche.
  - El tramo derecho de la línea del frente, y toda la línea de Rodríguez Peña derecha,
    que **quedan fuera del cuadro** desde el segundo 2 o 3, porque la cámara se corre
    hacia la izquierda. Con ese paneo, ninguna línea cubre todo el ancho de la avenida
    durante el clip completo.
- **El conteo manual también es aproximado:** con la calle mojada, las luces traseras dejan
  reflejos rojos que se confunden con vehículos.
- **Es la limitación nocturna que anticipa el brief.** Va al README, no al demo.

## Resumen

| Clip | ¿Cumple ±10 %? | Comentario |
|---|---|---|
| muestra_2 (día, fluido) | ✅ todas las líneas | El caso de referencia del brief. |
| muestra_3 (día, cola) | ⚠️ frente sí, fondo no | Los autos se cuentan bien; las motos lejanas, no. |
| muestra_1 (noche) | ❌ | Detección nocturna y cámara que saca las líneas del cuadro. |

En los tres clips **no hubo ni un solo cruce falso**: el sistema se equivoca por omisión,
nunca inventando vehículos.

## Correcciones que salieron de la validación

La primera comparación daba −19 % en el frente y −15 % en el fondo. Las causas eran dos:

1. **Línea del frente demasiado cerca del borde inferior.** Los autos del carril izquierdo
   se detectaban recién después de cruzarla. Se subió ~45 px hacia el cruce.
2. **Segmento del fondo demasiado corto.** Los autos que salían por el borde izquierdo
   pasaban por fuera del segmento. Se alargó hasta el borde de la imagen.

Además, un track que aparece dentro de la banda muerta de ±4 px toma como lado inicial
el de su primer punto. Antes, esos vehículos nunca contaban.

**Lección para marcar geometrías:**
- Las líneas tienen que quedar donde los vehículos ya están detectados de forma estable,
  lejos de los bordes del cuadro.
- Tienen que cubrir todo el ancho de la calzada.

## Pendiente

- **Motos:** se pierden incluso con `yolov8s`. Se probó subir la resolución de inferencia
  (primeros 20 s de muestra_2 y muestra_3):

  | Resolución | m2 frente | m2 fondo | m3 frente | m3 fondo | m3 R. Peña izq | Tiempo |
  |---|---|---|---|---|---|---|
  | **640 (actual)** | −5 % | −6 % | −6 % | −50 % | ✓ | 1× |
  | 960 | −5 % | −6 % | −11 % | −50 % | +200 % | ~2× |
  | 1280 | 0 % | −6 % | −11 % | −50 % | +300 % | ~3× |

  No ayuda: las motos lejanas siguen sin detectarse, el frente de m3 empeora y un
  colectivo detenido se parte en varios IDs que cuentan de más. Con la compresión de
  WhatsApp esa información ya se perdió; hace falta el video original.
- **Noche y paneo de cámara:** se resuelven grabando con el teléfono fijo, no con
  software.
