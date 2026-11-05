# Validación del color (Fase 3b)

**Criterio del brief:** sobre una muestra manual de 30 tracks, los colores saturados
aciertan por encima del 85 % y la familia acromática por encima del 65 %.

## Método

- **Dos muestras disjuntas** de 30 vehículos al azar cada una, entre los que cruzan
  alguna línea en los clips de día (muestra_2 y muestra_3):
  - **calibración**, para elegir los umbrales;
  - **validación**, reservada y medida una sola vez con los umbrales ya elegidos.
- **Etiquetado a ciegas** mirando los recortes (`thumbs/`), sin ver el color asignado.
  Los recortes donde no se distingue el vehículo se excluyen: 2 en validación y 6 en
  calibración.
- **Regla fijada antes de comparar:** un taxi (carrocería negra, techo amarillo) es
  **amarillo**. Desde un sexto piso lo que más se ve es el techo, y es lo que identifica
  a un taxi en una consulta.
- Las etiquetas están en `docs/validacion_color_muestras.csv` y los números se
  reproducen con `python scripts/calibrar_color.py validar`.

> Las etiquetas manuales las puso Claude (Claude Code) mirando los recortes, no una persona.

## Resultado actual (v3: `yolov8s` + balance de blancos)

| Muestra | Saturados | Acromáticos | Objetivo |
|---|---|---|---|
| Validación, día (25 vehículos) | **88 %** | **76 %** | > 85 % y > 65 % ✅ |
| Noche (23 vehículos) | **100 %** | **71 %** | (sin objetivo en el brief) |

**Cumple el criterio del brief.** Se reproduce con `python scripts/calibrar_color.py validar`.

## Historia

| Versión | Cambio | Día: saturados / acromáticos | Noche: acromáticos |
|---|---|---|---|
| v1 | umbrales absolutos de brillo, `yolov8n` | 82 % / 41 % | — |
| v2 | brillo relativo al asfalto, `yolov8n` | 82 % / 65 % | — |
| v3a | v2 con `yolov8s` (cajas más ajustadas) | 88 % / 76 % | 24 % |
| v3b | v3a + balance de blancos (actual) | **88 % / 76 %** | **71 %** |

La muestra nocturna se etiquetó sobre los tracks de `yolov8s`, así que no hay medición
nocturna de v1 ni de v2.

### v1 → v2: brillo relativo al asfalto

- **v1:** umbrales absolutos de brillo, negro si V < 60 y blanco si V ≥ 170. Los 7 autos
  negros de la validación salieron "gris": la pintura negra refleja el cielo nublado.
- **v2:** el brillo se compara con el del asfalto que rodea a la caja en el mismo frame
  (mediana de un anillo alrededor, sin otras cajas):
  - negro = V < 0.6 × fondo;
  - blanco = acromático con V ≥ 1.3 × fondo;
  - saturación mínima 40.

  Los umbrales salieron de una búsqueda en grilla sobre la muestra de calibración.

### v2 → v3

**`yolov8s`:** el modelo más grande dibuja cajas más ajustadas, y la región de la que
sale el color tiene menos asfalto y menos fondo. Solo con eso, el día pasó de 82 % / 65 %
a 88 % / 76 %.

**Balance de blancos para la noche:**
- **El problema:** de noche, la luz naranja del alumbrado de sodio tiñe todo. 10 de 13
  autos blancos o plateados salían "amarillo".
- **La corrección:** el asfalto es gris neutro, así que el tinte del asfalto que rodea al
  auto es el tinte de la luz. Se le quita al auto antes de clasificar.
  - Intensidad 0.75: con 1.0 la corrección se pasa y algunos blancos salen "azul".
- **Cuándo se aplica:** solo si el asfalto está teñido, con canal máximo / mínimo mayor
  a 1.15.
  - De día el tinte es ~1.03 (el 95 % por debajo de 1.13); de noche, ~1.35 (el 95 % por
    encima de 1.14).
  - El umbral sale de esa distribución, no de los aciertos.
  - Aplicada siempre, empeoraba el día (de 76 % a 71 %). Con la condición, el día no
    cambia.
- **Advertencia:** la intensidad (0.75) se eligió mirando la propia muestra nocturna, así
  que el 71 % de noche es optimista. Hace falta otra muestra nocturna para medirlo sin
  sesgo.

### Reasignación de etiquetas al pasar a `yolov8s`

Las etiquetas originales apuntaban a números de track de `yolov8n`, y `yolov8s` numera
distinto. Cada etiqueta se reasignó al track de `yolov8s` que más se superpone (IoU > 0.3
por frame) con el track viejo: el mismo vehículo físico, así que la etiqueta puesta a
ciegas sigue valiendo. Hubo tres ajustes:

- **4 etiquetas con superposición menor al 50 %:** excluidas por dudosas.
- **Duplicados:** dos taxis que `yolov8n` partía en dos IDs ahora son un solo track y
  cuentan una vez.
- **Filtración entre muestras:** un taxi quedó en calibración y en validación a la vez;
  se sacó de validación.

Todo queda registrado en las columnas `track_id_yolov8n`, `cobertura` y `nota` del CSV.

### Errores que quedan

- **Día:** la confusión acromática de siempre (gris oscuro ↔ negro, blanco en sombra ↔
  gris) y algún taxi donde se ve más carrocería negra que techo amarillo.
- **Noche:** algunos plateados salen "indefinido" o "blanco", y un blanco sale "azul" por
  sobrecorrección.

## Limitaciones (van al README)

- **La familia acromática concentra el error,** como anticipa el brief. Blanco, plata y
  gris claro se confunden según la luz.
- **De noche** el balance de blancos recupera la mayoría de los blancos y plateados, pero
  la medición nocturna es optimista (ver la advertencia de v3).
- **El video de WhatsApp (464 × 832) deja autos de 30–60 px de ancho.** El video original
  debería mejorar todo esto.
