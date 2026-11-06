# Velocidad en km/h

Para pasar de píxeles a metros, cada encuadre tiene una **calibración métrica del suelo**
(`configs/suelo/<encuadre>.yaml`): una homografía que lleva cualquier punto de la calle
en la imagen a su posición en metros. Con eso, la velocidad de cada vehículo sale en km/h
(`src/atlas_traffic/velocidad.py`).

## Referencia de medida: las sendas peatonales

El [Manual de Señalamiento Horizontal](https://aacarreteras.org.ar/pdfs/documentos-tecnicos/msh-ebook.pdf)
de Vialidad Nacional (Res. 2501/2012, punto IV.3) fija las franjas de la senda peatonal
en 0,40 a 0,50 m de ancho, con un espacio igual a su ancho. Se usa **franja de 0,50 m y
espacio de 0,50 m: 1,00 m de una franja a la siguiente**.

Verificación de que son de 0,50 m y no de 0,40 m: con 1,00 m, los carriles de Córdoba
miden 3,0 a 3,5 m, lo normal en una avenida. Con 0,80 m medirían 2,4 a 2,8 m, menos que
el ancho de un colectivo.

## Cómo se calibra (`scripts/calibrar_suelo.py`)

1. **Fondo limpio:** mediana de 41 cuadros estabilizados. Los vehículos desaparecen y
   quedan las marcas de la calle.
2. **Franjas:** a lo largo de cada senda se mide el brillo y se ubica el centro de cada
   franja. Se descartan las cortadas y las de ancho raro, como la ciclovía.
3. **Encuadre de referencia** (video del iPhone, el más nítido): se ajusta la
   transformación con dos condiciones:
   - las franjas de las cuatro sendas quedan cada 1 m;
   - Córdoba y Rodríguez Peña quedan perpendiculares.

   Las sendas que cruzan Córdoba dan la escala a lo ancho de la avenida. Las que cruzan
   Rodríguez Peña la dan a lo largo, que es la dirección en que circulan los autos.
4. **Encuadres de WhatsApp:** muestran las mismas sendas desde otro ángulo. Se calibran
   con la posición en metros de esas franjas, ya conocida por la referencia, probando a
   qué franja corresponde cada una. Alcanzan dos sendas, aunque se vean en parte, como
   en la cola de muestra_3.

| Encuadre | Franjas | Error de ajuste | Período medido |
|---|---|---|---|
| n1 (iPhone, referencia) | 50 | 2,3 px | 0,99 ± 0,08 m |
| m1 (noche, lluvia) | 29 | 1,0 px | 0,97 ± 0,09 m |
| m2 | 32 | 0,8 px | 1,00 ± 0,08 m |
| m3 (cola) | 20 | 0,7 px | 1,01 ± 0,12 m |

Hay una vista desde arriba de cada encuadre en `outputs/suelo/`. En las cuatro, las
sendas, el techo de la parada y las esquinas caen en el mismo lugar de la cuadrícula.

## Cómo se mide la velocidad

- **Punto que se mide:** el centro del borde inferior de la caja, donde el vehículo toca
  el suelo. El centro de la caja está a ~1 m de altura y exageraría la velocidad.
- **Velocidad instantánea:** desplazamiento en una ventana de 0,5 s.
- **Por vehículo:** mediana mientras se mueve (más de 3 km/h) y máximo robusto
  (percentil 90). Solo dentro de la zona de análisis.
- **Por intervalo:** velocidad media en la zona de análisis (incluye a los detenidos:
  mide cuán fluido está el tránsito) y percentil 85, el indicador estándar en ingeniería
  de tránsito.

## Resultados

| Clip | Mediana | p85 | Máxima | Situación |
|---|---|---|---|---|
| muestra_1 (noche) | 27 km/h | 31 km/h | 35 km/h | fluido |
| muestra_2 (día) | 32 km/h | 39 km/h | 43 km/h | fluido |
| muestra_3 (día) | 6 km/h | 17 km/h | 29 km/h | cola |
| esquina_20261105_1946 | 29 km/h | 35 km/h | 37 km/h | fluido |

En avenidas de CABA la máxima es 60 km/h. En una esquina con semáforo y sendas, 25 a
40 km/h en tránsito fluido es lo esperable.

## Verificaciones

- **Dos formas de medir coinciden:** la velocidad instantánea y la distancia entre las
  líneas de entrada y salida dividida por el tiempo de recorrido difieren en 0,6 a
  1,4 km/h (medianas por clip).
- **La escala no depende de la distancia:** un auto del video del iPhone mantiene
  ~28 km/h mientras se aleja de 0 a 78 m de la cámara, achicándose casi 10 veces en la
  imagen. Si la transformación estuviera mal, su velocidad derivaría.
- **Margen estimado: ±5 %.** Viene sobre todo de la medida real de las franjas y de la
  precisión del ajuste.

## Limitaciones

- **Supone la calle plana.** Lejos de la zona de las sendas (más de ~40 m) la calle
  puede tener pendiente.
- **Calibración de otra esquina:** hay que repetir el proceso con sus propias sendas.
  Si no tienen franjas, sirve cualquier medida conocida del suelo, como el ancho de un
  carril medido en sitio.
- **Validación independiente:** la ideal es un auto que pase a velocidad conocida
  (velocímetro o GPS) frente a la cámara.
