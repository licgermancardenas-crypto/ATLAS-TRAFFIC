# Senda bloqueada y detención prolongada

Código: `src/atlas_traffic/infracciones.py`. Parámetros: sección `infracciones:` de
`configs/default.yaml`. Zonas de las sendas: `configs/camaras.yaml`.

## Senda bloqueada

Un vehículo detenido (menos de 3 km/h) durante 2 s o más con al menos el 30 % de su apoyo
sobre una senda peatonal. Pasa sobre todo con cola: el semáforo corta y el auto queda
encima de la senda, y los peatones tienen que pasar entre los autos.

- **Qué parte del vehículo se mira.** La parte baja de la caja (el 40 % inferior),
  llevada al suelo en metros con la calibración de las sendas
  (ver [velocidad.md](velocidad.md)). Un solo punto no alcanza: un auto de 4,5 m puede
  tener la cola antes de la senda y la trompa encima.
- **Las sendas.** Son rectángulos en metros, medidos sobre la vista desde arriba
  (`outputs/suelo/*.jpg`). Todos los encuadres de la esquina comparten el mismo sistema de
  metros, y las sendas coinciden entre encuadres a ±0,5 m.
- **Qué sendas usa cada encuadre.** Solo las que se ven y se midieron en él. Fuera de esa
  zona la calibración no vale: en la primera prueba, las sendas de Rodríguez Peña
  aparecían sobre un árbol o en el borde de la imagen en encuadres donde no se calibraron.
- **Gravedad:**
  - grave: 15 s o más;
  - moderada: de 5 a 15 s;
  - leve: menos de 5 s.
  - Sube un nivel si había personas en la senda mientras estaba bloqueada. Cuentan quienes
    se vieron 10 veces o más dentro de la senda; incluye ciclistas, porque el detector no
    los distingue de los peatones.

### Validación a ojo

Primero, sin limitar las sendas por encuadre, hubo 13 candidatos:
- 6 caían sobre sendas no calibradas en ese encuadre: descartados con la regla de arriba;
- 7 eran vehículos detenidos encima de las franjas en la cola de `muestra_3`. Se
  confirmaron con recortes ampliados.

Con la versión final hay 8 alertas, todas en `muestra_3`:

| Vehículo | Senda | Duración | Personas en la senda | Gravedad | A ojo |
|---|---|---|---|---|---|
| #5 | Córdoba, lado cámara | 8 s | 0 | moderada | ✅ encima de las franjas |
| #58 | Córdoba, lado opuesto | 13 s | 0 | moderada | ✅ |
| #216 | Córdoba, lado cámara | 14 s | 2 | grave | ✅ |
| #2 (camioneta) | Córdoba, lado cámara | 8 s | 2 | grave | ✅ (repartidores en bicicleta pasando al lado) |
| #231 | Córdoba, lado cámara | 5 s | 1 | grave | ✅ |
| #115 | Córdoba, lado opuesto | 5 s | 0 | leve | ⚠️ límite: la cola del auto apenas toca el borde de la senda |
| #2 (camioneta) | Córdoba, lado cámara | 4 s | 1 | moderada | ✅ (la misma camioneta, que avanzó un poco y volvió a frenar) |
| #216 | Córdoba, lado opuesto | 4 s | 0 | leve | ✅ (el mismo auto, ya en la otra senda) |

**El caso límite (#115).** Se superpone con la senda un 42 %, casi igual que la camioneta
#2 (40 %), que sí está encima: el umbral no los separa. Para estar seguros de que el
vehículo está encima habría que bajar a pocos centímetros de error. Con video original y
cámara fija va a ser más preciso.

## Detención prolongada

Un vehículo detenido 20 s o más en la calzada mientras el tránsito de al lado circula. Para
que cuente:
- pasan al menos 5 vehículos a menos de 6 m;
- no hay ningún otro vehículo parado al lado: si lo hay, es una cola de semáforo.

Se excluyen los colectivos, porque paran en la parada.

**No es una infracción confirmada.** Puede ser doble fila, estacionamiento prohibido o
ascenso y descenso de pasajeros. El sistema no sabe dónde está el carril de
estacionamiento ni el cordón, así que la alerta es para que alguien la revise.

**Validación a ojo.** Hay 1 alerta: un taxi en `muestra_2`, detenido 35 s junto al cordón
justo pasando la esquina, mientras pasaban 11 vehículos al lado. El tracker lo vuelve a
tomar con otro número (#295) unos segundos después, en el mismo lugar.
- **Descartado 1:** una detención en `muestra_3` a 34 m, fuera de la zona donde el suelo
  está medido y en plena cola.
- **Descartado 2:** en una prueba, el último auto de una cola tenía un solo vecino parado y
  pasaba el filtro. Por eso la regla exige que no haya ningún otro vehículo parado al lado.

## Evidencia

Cada alerta guarda en `outputs/runs/<clip>/infracciones/`:
- una foto, de la mitad de la detención;
- un video corto, de 1 s antes hasta 1 s después, con un tope de 15 s.

El vehículo está marcado en rojo.
