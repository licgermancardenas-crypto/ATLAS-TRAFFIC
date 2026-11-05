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

## Resultado (muestra de validación, 28 vehículos etiquetables)

| Familia | Umbrales absolutos (v1) | Brillo relativo al asfalto (v2, actual) | Objetivo |
|---|---|---|---|
| Saturados (rojo, azul, verde, amarillo) | 82 % (9/11) | **82 %** (9/11) | > 85 % ✗ |
| Acromáticos (blanco, gris, negro) | 41 % (7/17) | **65 %** (11/17) | > 65 % ≈ en el límite |

**No cumple estrictamente ninguno de los dos umbrales.** Con muestras tan chicas, un
vehículo mueve el resultado entre 6 y 9 puntos: los saturados quedan a un acierto del
objetivo y los acromáticos justo en el límite. Para una estimación firme hace falta una
muestra más grande (~100 tracks).

### Errores que quedan

**En validación:**
- **Gris ↔ negro:** dos grises oscuros salieron negro y dos negros brillantes salieron
  gris (la pintura negra refleja el cielo nublado).
- **Blanco ↔ gris:** un blanco en sombra salió gris y un plateado salió blanco.
- **Taxis:** dos salieron "negro", porque en esos frames se ve más la carrocería que el
  techo.

**En calibración**, con los mismos umbrales: saturados 92 %, acromáticos 82 %.

## Qué cambió de v1 a v2

- **v1:** umbrales absolutos de brillo, negro si V < 60 y blanco si V ≥ 170. Los 7 autos
  negros de la validación salieron "gris".
- **v2:** el brillo se compara con el del asfalto que rodea a la caja en el mismo frame
  (mediana de un anillo alrededor, sin otras cajas):
  - negro = V < 0.6 × fondo;
  - blanco = acromático con V ≥ 1.3 × fondo;
  - saturación mínima 40.

  Los umbrales salieron de la búsqueda en grilla sobre la muestra de calibración.

## Limitaciones (van al README)

- **La familia acromática concentra el error,** como anticipa el brief. Blanco, plata y
  gris claro se confunden según la luz.
- **De noche el color no es confiable.** En muestra_1, de unos 12 vehículos clasificados
  "amarillo", solo ~4 son taxis: el resto son autos blancos o plateados teñidos por la luz
  naranja del alumbrado. Una corrección de balance de blancos por frame podría mitigarlo;
  no está implementada.
- **El video de WhatsApp (464 × 832) deja autos de 30–60 px de ancho.** El video original
  debería mejorar todo esto.
