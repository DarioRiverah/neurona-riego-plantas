# Neurona artificial para el riego de plantas

Neurona artificial (una sola unidad, activación sigmoide) que decide si una planta necesita riego a partir de la **humedad del suelo** y la **temperatura ambiental**. Salida `1` = regar, `0` = no regar.

> Los datos son didácticos y no representan una recomendación agronómica para una especie real.

## Problema y variables

| Variable | Descripción | Escala |
|---|---|---|
| `x1` | Humedad del suelo (%) | 0 – 100 |
| `x2` | Temperatura ambiental (°C) | 0 – 50 |
| `y` | 1 = regar, 0 = no regar | binaria |

Datos de entrenamiento: 10 casos (5 de "no regar" con humedad ≥ 50 % y 5 de "regar" con humedad ≤ 40 %).

## Cómo ejecutar

Requiere [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<tu-usuario>/neurona-riego-plantas.git
cd neurona-riego-plantas
uv sync
uv run .\src\neurona_riego_plantas\__init__.py
```

En Linux/macOS: `uv run src/neurona_riego_plantas/__init__.py`

## Cómo funciona

1. **Normalización:** `X_normalizado = X / escala`, con `escala = [100, 50]`. Los datos nuevos se dividen por la misma `escala`.
2. **Suma ponderada:** `z = X_normalizado @ pesos + sesgo`
3. **Activación:** `sigmoide(z) = 1 / (1 + e^(-z))` → probabilidad entre 0 y 1.
4. **Error:** error cuadrático medio (MSE) entre la probabilidad y la salida esperada.
5. **Gradientes:** `gradiente_z = error · p · (1 − p)`, `gradiente_pesos = X_normalizado.T @ gradiente_z / n`, `gradiente_sesgo = mean(gradiente_z)`.
6. **Actualización:** `pesos -= tasa · gradiente_pesos`, `sesgo -= tasa · gradiente_sesgo`.
7. **Decisión:** `respuesta = (probabilidad >= 0.5).astype(int)`.

Pesos y sesgo se inicializan aleatoriamente en [-1, 1] con semilla fija (`42`) para que los resultados sean reproducibles.

## Resultados del entrenamiento base (tasa 0.5, 10 000 épocas)

| Parámetro | Valor |
|---|---|
| Peso humedad | **-14.9497** |
| Peso temperatura | **+2.6066** |
| Sesgo | 5.2147 |
| Error final (MSE) | 0.030334 |
| Aciertos | 10/10 |

| Caso | Humedad | Temp. | Probabilidad | Predicción | Esperado |
|---|---|---|---|---|---|
| 1 | 80 % | 18 °C | 0.0030 | 0 | 0 |
| 2 | 70 % | 22 °C | 0.0163 | 0 | 0 |
| 3 | 65 % | 28 °C | 0.0455 | 0 | 0 |
| 4 | 55 % | 25 °C | 0.1539 | 0 | 0 |
| 5 | 50 % | 32 °C | 0.3562 | 0 | 0 |
| 6 | 40 % | 30 °C | 0.6897 | 1 | 1 |
| 7 | 35 % | 25 °C | 0.7834 | 1 | 1 |
| 8 | 30 % | 32 °C | 0.9167 | 1 | 1 |
| 9 | 20 % | 35 °C | 0.9829 | 1 | 1 |
| 10 | 10 % | 38 °C | 0.9967 | 1 | 1 |

**Significado del signo de los pesos**

- **Peso de la humedad negativo (-14.95):** a más humedad, menor `z` y menor probabilidad de regar. Tiene sentido: un suelo húmedo no necesita agua. Además es el peso de mayor magnitud, así que la humedad es la variable que más pesa en la decisión.
- **Peso de la temperatura positivo (+2.61):** a más temperatura, mayor probabilidad de regar (más calor → más evaporación). Su magnitud es mucho menor, por lo que la temperatura actúa como un factor secundario que "empuja" los casos dudosos.

## Predicciones con condiciones nuevas

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0117 | 0 – No regar |
| 45 % | 34 °C | 0.5646 | 1 – Regar |
| 25 % | 22 °C | 0.9324 | 1 – Regar |
| 50 % | 25 °C | 0.2775 | 0 – No regar |
| 30 % | 40 °C | 0.9435 | 1 – Regar |

El caso 45 % / 34 °C queda muy cerca de 0.5: está justo en la frontera entre los dos grupos de entrenamiento (40 % y 50 %), y la temperatura alta lo inclina ligeramente a "regar".

## Experimentos con los parámetros

En cada prueba se cambió un solo parámetro respecto a la base (misma semilla y mismos datos).

| Experimento | Épocas | Tasa | Error final | Aciertos | P(45 %, 34 °C) | Aprendizaje |
|---|---|---|---|---|---|---|
| Prueba base | 10 000 | 0.5 | 0.030334 | 10/10 | 0.5646 | Adecuado |
| Pocas épocas | 100 | 0.5 | 0.216162 | 10/10 | 0.4914 | Insuficiente |
| Cantidad intermedia | 1 000 | 0.5 | 0.089950 | 10/10 | 0.5521 | Lento / aún incompleto |
| Más épocas | 20 000 | 0.5 | 0.019455 | 10/10 | 0.5442 | Mejora pequeña |
| Tasa pequeña | 10 000 | 0.01 | 0.177776 | 10/10 | 0.5046 | Muy lento / insuficiente |
| Tasa moderada | 10 000 | 0.1 | 0.066760 | 10/10 | 0.5688 | Lento |
| Tasa alta | 10 000 | 1.0 | 0.019454 | 10/10 | 0.5442 | Rápido |
| Tasa muy alta | 10 000 | 2.0 | 0.011639 | 10/10 | 0.5231 | Rápido, sin inestabilidad en este problema |

Observaciones:

- Todas las configuraciones clasifican bien los 10 casos, pero **no con la misma confianza**. Con 100 épocas las probabilidades quedaron entre 0.42 y 0.56: el caso 6 (40 %, 30 °C) dio ≈ 0.4998 y quedó en "regar" prácticamente por azar. El error (0.216) muestra que la neurona casi no aprendió.
- "Tasa alta, 10 000 épocas" y "Más épocas, tasa 0.5" dieron el mismo error (0.019454 ≈ 0.019455): el avance total depende de `tasa × épocas`, así que duplicar la tasa equivale aproximadamente a duplicar las épocas.
- Con tasa 0.01 las probabilidades quedaron entre 0.33 y 0.65: aprendió la dirección correcta pero no alcanzó a separar bien los grupos.

## Prueba con el umbral (sin reentrenar)

| Dato | Humedad | Temp. | Probabilidad | u = 0.4 | u = 0.5 | u = 0.6 |
|---|---|---|---|---|---|---|
| Caso 5 | 50 % | 32 °C | 0.3562 | 0 | 0 | 0 |
| Caso 6 | 40 % | 30 °C | 0.6897 | 1 | 1 | 1 |
| Nuevo 2 | 45 % | 34 °C | 0.5646 | 1 | 1 | **0** |
| Nuevo 4 | 50 % | 25 °C | 0.2775 | 0 | 0 | 0 |

(El resto de casos tienen probabilidades lejos de 0.4–0.6 y no cambian; la tabla completa aparece al ejecutar el programa.)

Solo cambia **45 % / 34 °C**: con umbral 0.6 pasa a "no regar" porque su probabilidad (0.5646) está entre 0.5 y 0.6. Con 0.4 no cambia nada porque ningún caso tiene probabilidad entre 0.4 y 0.5. El umbral solo se aplica **después** de calcular la probabilidad; no participa en el cálculo del error ni de los gradientes, así que los pesos y el sesgo aprendidos son exactamente los mismos. Lo que cambia es qué tan "exigente" es la decisión: un umbral bajo riega más fácilmente, uno alto es más conservador.

## Análisis

**1. ¿Por qué fue necesario normalizar?**
La humedad llega a 100 y la temperatura a 50. Sin normalizar, la humedad aporta valores más grandes a `z` y a los gradientes, domina la actualización de los pesos y la sigmoide se satura rápido (probabilidades pegadas a 0 o 1 con gradiente casi nulo). Al dividir por `[100, 50]` ambas quedan entre 0 y 1 y el entrenamiento es más estable y parejo.

**2. ¿Dónde se usó `X_normalizado` y para qué se conservó `X`?**
`X_normalizado` se usó en la suma ponderada (`z = X_normalizado @ pesos + sesgo`) y en el gradiente de los pesos (`X_normalizado.T @ gradiente_z`). `X` se conservó para mostrar los valores originales (% y °C) en las tablas, que son los que una persona entiende.

**3. ¿Qué ocurrió con 100 épocas?**
El error quedó alto (0.216) y todas las probabilidades se quedaron cerca de 0.5 (0.42–0.56). Aunque el conteo da 10/10, varios casos quedan a centésimas del umbral: la neurona apenas empezó a aprender y sus decisiones no son confiables.

**4. ¿Más épocas siempre mejoran de forma importante?**
No. De 100 a 1 000 épocas el error bajó de 0.216 a 0.090 y de 1 000 a 10 000 bajó a 0.030, pero de 10 000 a 20 000 solo bajó a 0.019 sin cambiar ningún acierto. Llega un punto de rendimientos decrecientes: más épocas solo hacen a la neurona más "segura" en casos que ya clasificaba bien.

**5. ¿Qué efecto tuvo una tasa demasiado pequeña?**
Con 0.01 los pasos son tan cortos que en 10 000 épocas el error solo llegó a 0.178, similar a entrenar con tasa 0.5 durante unas 200 épocas. El aprendizaje es muy lento y las probabilidades quedan poco separadas (0.33–0.65).

**6. ¿Qué efecto tuvo una tasa alta o muy alta?**
Con 1.0 y 2.0 el error bajó más rápido (0.019 y 0.012). En este problema no hubo inestabilidad porque los datos son pocos, están normalizados y son linealmente separables; además, la derivada de la sigmoide (máximo 0.25) amortigua los pasos. Aun así, una tasa muy alta puede hacer que el error oscile o diverja en problemas con más datos o sin normalizar, por lo que no conviene subirla sin control.

**7. ¿Qué representa el signo del peso de la humedad?**
Es negativo: más humedad → menor probabilidad de regar. Es una relación inversa y es la variable dominante.

**8. ¿Qué representa el signo del peso de la temperatura?**
Es positivo: más temperatura → mayor probabilidad de regar. Relación directa pero con menos influencia que la humedad.

**9. ¿Por qué convertir la probabilidad en 0 o 1 con un umbral?**
La sigmoide entrega un valor continuo (ej. 0.5646), pero el problema pide una acción concreta: regar o no regar. El umbral transforma esa probabilidad en una decisión binaria y permite ajustar qué tan conservador es el sistema sin reentrenar.

**10. Limitaciones frente a una planta real**
- Solo usa dos variables; ignora especie, tipo de suelo, luz solar, humedad del aire, lluvia pronosticada, etapa de crecimiento y último riego.
- Entrenó con 10 datos inventados; no hay datos reales ni de validación.
- Una sola neurona solo traza una frontera lineal; relaciones más complejas (ej. que la temperatura importe distinto según la humedad) no se pueden representar.
- Solo decide *si* regar, no *cuánta* agua ni *cuándo*.
- La normalización asume rangos fijos (0–100 % y 0–50 °C); valores fuera de ese rango darían resultados poco confiables.

## Estructura del repositorio

```
neurona-riego-plantas
├─ .python-version
├─ pyproject.toml
├─ README.md
├─ src
│  └─ neurona_riego_plantas
│     └─ __init__.py   # Neurona, entrenamiento, pruebas y experimentos
└─ uv.lock
```