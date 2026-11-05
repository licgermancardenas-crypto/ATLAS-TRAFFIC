# Validación de la alerta de sentido contrario (Fase 3c)

**Criterio del brief:** sobre los clips disponibles, cada evento disparado es verificable
mirando el recorte. Cero falsos positivos en el set de prueba, aunque haya que endurecer
umbrales.

## Regla

Cada línea declara `sentido_permitido` (`entrante`, `saliente` o `ambos`). Hay alerta
cuando un vehículo cruza en el sentido no permitido y pasa los tres filtros:

| Filtro | Valor | Para qué |
|---|---|---|
| Historia mínima | 15 detecciones | descartar IDs efímeros del tracker |
| Desplazamiento neto mínimo | 40 px | descartar vehículos detenidos con temblor |
| Cruce limpio | 3 detecciones de cada lado, fuera de la banda de ±4 px | descartar cruces dudosos |

> **Desvío del brief.** El brief usa `sentido_positivo` para decidir qué es contramano.
> Acá `sentido_positivo` es una convención geométrica (cruzar hacia el centro del cruce
> = entrante) y no dice qué está permitido. Aplicar la regla tal cual habría marcado como
> contramano todo el tránsito normal. Por eso se separó en un campo propio.

**Sentidos cargados:**
- **Av. Córdoba:** mano única hacia el fondo. Confirmado por las trayectorias: ningún
  cruce en sentido opuesto en los tres clips.
- **Rodríguez Peña:** de derecha a izquierda mirando desde el balcón. Deducido de lo
  observado en muestra_3 y confirmado por el usuario.

## Resultado

| Clip | Eventos | Falsos positivos |
|---|---|---|
| muestra_1 (23 s) | 0 | 0 |
| muestra_2 (50 s) | 0 | 0 |
| muestra_3 (46 s) | 0 | 0 |

Ningún vehículo circula a contramano en las grabaciones, así que lo esperable es cero
eventos. Cumple el criterio.

## Prueba controlada (la regla sí dispara)

Para verificar que la alerta funciona con video real, se invirtió temporalmente el
sentido permitido de Córdoba en muestra_2 (sin cambiar la configuración):
- Cada auto normal pasa a ser un "contramano".
- De los **84 cruces** reales sobre Córdoba, **81 (96 %)** pasaron los tres filtros y
  generaron evento.
- En los recortes revisados se ve el vehículo marcado sobre la línea, con la línea bien
  ubicada aunque la cámara se haya movido.

Los filtros descartan pocos cruces legítimos, así que la alerta no queda ciega por exceso
de rigor.

## Pendiente

- Grabar un evento real de contramano (por ejemplo, una bicicleta o una moto) para
  validar con un caso positivo genuino.
