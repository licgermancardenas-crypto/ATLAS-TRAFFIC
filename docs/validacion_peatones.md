# Validación del conteo de personas

Conteo manual a ciegas (sin mirar lo que contó el sistema) sobre muestra_3, el único clip
con gente caminando por las líneas de personas. Clip completo (46 s), día.

## Método

- **Vereda:** recortes de la vereda cada 0.5 s, y cada 0.25 s en el tramo con gente
  (27–37 s), en coordenadas estabilizadas y con la línea dibujada. Se siguió a cada
  persona cuadro a cuadro.
- **Ciclovía:** barrido temporal de la línea (`scripts/barrido_temporal.py`): cada
  ciclista que cruza deja una raya.
- **Advertencia:** estos 46 s incluyen cuadros usados para entrenar el modelo de la
  esquina (solo los primeros 20 s quedaron reservados, y en ellos no camina nadie por
  estas líneas). El resultado puede ser algo optimista.

## Resultado

| Línea | Manual | Sistema antes | Sistema ahora | Detección |
|---|---|---|---|---|
| Vereda de Córdoba (edificios), bajando | 7 (+1 dudosa) | 2 | **4** | ~57 % |
| Ciclovía de Córdoba, subiendo | ~9 (aprox.) | 4 | 4 | ~45 % |
| Sendas peatonales (los 3 clips) | 0 | 0 | 0 | sin cruces falsos |

- **Sin conteos falsos:** todo lo que contó el sistema es una persona real cruzando. El
  error es siempre por omisión, igual que con los vehículos.
- **La dudosa:** una persona que entra por el borde de la imagen justo sobre la línea, a
  los 35 s.
- **Ciclovía:** los ciclistas van en grupo y se superponen, así que el conteo manual es
  aproximado (±2).

## Qué se corrigió

1. **La ROI descartaba a las personas de la vereda.** La ROI marca la calzada para
   ignorar el tránsito del fondo. Una persona que caminaba toda su vida por la vereda,
   fuera de esa zona, se descartaba antes de contar. Ahora la ROI filtra solo vehículos.
2. **Unión de tracks cortados** (`src/atlas_traffic/stitching.py`):
   - **Pasada estricta, para todas las clases:** une pedazos separados por menos de
     0.5 s que encajan en posición, velocidad y tamaño.
   - **Pasada de personas en movimiento:** caminando en grupo, el tracker pierde a la
     gente 1–1.5 s justo al cruzar la línea. Para contar no hace falta saber quién es
     quién: alcanza con emparejar a quien desaparece caminando hacia la línea con quien
     aparece del otro lado siguiendo igual. Las personas quietas no se unen.
   - **Huecos más largos, no:** a esta resolución ni el movimiento ni los colores
     distinguen a una persona de su vecina (se probó y se descartó la comparación de
     colores).
3. **La línea "de la parada" estaba sobre la ciclovía.** Se renombró `ciclovia_cordoba`.
   El modelo no tiene clase bicicleta y ve al ciclista como persona, así que esa línea
   cuenta personas en bici.
4. **Corrección:** no había doble conteo por cambio de número. Las "dos personas de rosa"
   que parecían la misma contada dos veces son dos repartidores en bici que iban juntos.

## Por qué todavía se pierden personas

- **Son chicas:** ~25 px de alto en un video de 464 × 832 comprimido por WhatsApp.
- **Caminan en grupo:** las cajas se tocan y el tracker las confunde.
- **Un árbol tapa la vereda** justo arriba de la línea.

Lo que más mejoraría el conteo de personas es video original (sin WhatsApp): duplica la
resolución. También sirve reentrenar el modelo con personas etiquetadas a mano, y agregar
la clase bicicleta para separar ciclistas de peatones.
