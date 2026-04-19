# Respuestas — Práctica Final: Análisis y Modelado de Datos

---

## Ejercicio 1 — Análisis Estadístico Descriptivo


**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset Diamonds proviene de la librería seaborn. Tiene 53.940 filas y 10 columnas: 7 numéricas (`carat`, `depth`, `table`, `price`, `x`, `y`, `z`) y 3 categóricas ordinales (`cut`, `color`, `clarity`).  Tiene sentido hacer una regresión lineal para correlacionar las features de un diamante con su valor de mercado. 

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> La distribucion es principalmente hacia la derecha, donde la mayoría de los diamantes se concentrar en menor tamaño. Por lo que si son "gigantes" van a ser más caros, pero al ser atípicos por eso están "en la cola" derecha.
>En cuanto a los outliers, sí existen pero al ser sobre sus medidas decidí no borrarlos porque en el total de datos recolectados no van a tener una influencia sobre la regresión lineal porque el modelo es lo suficientemente robusto. En mi opinion esos pequeños datos pueden enriquecer más la realidad del modelo a hacerlo "perfecto" sacando esos datos.


**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Viendo el heatmap de correlaciones:
> - `carat` ↔ `price`: r ≈ **0.92** — la más alta con diferencia, el peso del diamante explica la mayor parte del precio.
> - `x` ↔ `price`: r ≈ **0.88** — la dimensión longitudinal, muy correlacionada con carat.
> - `y` ↔ `price`: r ≈ **0.87** — la dimensión transversal, mismo motivo.
>
> Hay que tener en cuenta que `x`, `y`, `z` y `carat` están muy correlacionadas entre sí (r > 0.95), así que en la práctica aportan información muy parecida. Incluir las cuatro en el modelo introduce multicolinealidad.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> El dataset Diamonds no tiene valores nulos propiamente dichos (todas las columnas tienen 53.940 registros completos).

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---

El pipeline seguido fue: carga del CSV → eliminación de duplicados → codificación de variables categóricas con `LabelEncoder` → escalado con `StandardScaler` → split 80/20 con `random_state=42` → entrenamiento con `LinearRegression` → evaluación con MAE, RMSE y R². El objetivo era predecir `price` usando todas las demás variables como features.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> En el test obtuvimos una MAE  de 840,74 USD, el RMSE de 1.312,45 usd y el R² de 0,887
> El modelo funciona bien porque, al darle información detallada de las features (como el peso o las dimensiones), logra captar la lógica del valor de los diamantes, similar a como si supiéramos el valor por kilo de un metal.
> Descarto el overfitting porque la diferencia de R² entre entrenamiento y test es mínima (-0.0022). Esto demuestra que el modelo no se memorizó los datos, sino que aprendió a generalizar el precio para diamantes que nunca había visto. El error del 22% (MAE) se debe principalmente a que la relación real entre tamaño y precio no es una línea recta perfecta, algo que se nota en que los errores crecen (RMSE) en las piezas más grandes.



---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---

En este ejercicio se implementó OLS (Mínimos Cuadrados Ordinarios) desde cero usando álgebra matricial con NumPy, sin tocar scikit-learn para el ajuste del modelo. Se usaron datos sintéticos generados con semilla fija para poder comparar los coeficientes estimados con los valores reales conocidos.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> La fórmula es la solución analítica de Mínimos Cuadrados Ordinarios (OLS) y lo que hace es tener en cuenta todas las features simultáneamente para entregarnos la predicción más aproximada a la realidad. En lugar de procesar los datos de forma aislada, el álgebra matricial evalúa todas las observaciones en conjunto para minimizar el error global frente al ruido, definiendo matemáticamente el peso exacto (los coeficientes $\beta$) de cada una de las variables.
> La columna de unos se crea para que el modelo tenga un "punto de partida" base, lo que técnicamente se conoce como el intercepto ($\beta_0$). Es necesario añadirla porque si nuestras variables valen 0, matemáticamente la ecuación daría como resultado 0 (ya que todo producto por 0 es 0). Al omitir esta columna, estaríamos obligando al modelo a pasar por el origen de coordenadas, lo cual sesgaría completamente el ajuste, ya que en la realidad los fenómenos casi nunca parten de un cero absoluto

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado | Error absoluto |
|-----------|:----------:|:--------------:|:--------------:|
| β₀        | 5.0        | 4.8650         | 0.1350         |
| β₁        | 2.0        | 2.0636         | 0.0636         |
| β₂        | -1.0       | -1.1170        | 0.1170         |
| β₃        | 0.5        | 0.4385         | 0.0615         |

> Como se puede ver en la tabla, la función logró recuperar los coeficientes originales bastante bien. Es verdad que el intercepto ($\beta_0$) fue el que más le costó ajustar, con una diferencia de 0.1350 frente al valor real. Sin embargo, esto es algo que ya me esperaba: al entrenar el modelo con tan pocas muestras (solo 160) y con un ruido interno de $\sigma=1.5$, es lógico que haya un pequeño margen de error en ese punto de partida.
> Lo que realmente me deja tranquilo de la prueba es la precisión que logró en los coeficientes principales ($\beta_1$, $\beta_2$ y $\beta_3$). En estos tres, el error fue bajísimo, lo que demuestra en la práctica que el código matricial en NumPy funciona bien y es capaz de capturar la relación real entre las variables sin ningún problema.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> - **MAE = 1.1665** — error absoluto medio de poco más de 1 unidad.
> - **RMSE = 1.4612** — muy cercano a σ_ε = 1.5 (el ruido real del modelo generador).
> - **R² = 0.6897** — el modelo explica el 68.97% de la varianza en test.

> Si contrasto esto con lo que pedía el enunciado, veo que tanto el error absoluto (MAE) como el RMSE cayeron perfecto dentro de los márgenes esperados. De hecho, me parece súper interesante que el RMSE refleje de forma casi exacta el nivel de ruido que le inyectamos a los datos al principio ($\sigma = 1.5$).
> El único punto que a simple vista podría llamar la atención es el $R^2$, que me dio cerca de 0.69 frente al 0.80 de referencia. Al principio lo revisé, pero analizándolo a fondo, es el resultado correcto. Por cómo está armado el experimento (con ese nivel de dispersión y esas características aleatorias), el modelo tiene un techo matemático que ronda justamente el 0.70. Es decir, el modelo no rinde menos de lo esperado, sino que aprendió el 100% de lo que esos datos podían enseñarle. Tratar de forzar un $R^2$ más alto sin cambiar la naturaleza del problema sería pedirle peras al olmo.

---

## Ejercicio 4 — Series Temporales

---

La serie temporal sintética cubre 6 años de datos diarios (2018-01-01 → 2023-12-31), con 2.191 observaciones en total. Fue generada con un modelo aditivo conocido: tendencia + estacionalidad + ciclo + ruido. El objetivo del ejercicio es descomponer esa serie y verificar que el residuo resultante se comporta como ruido blanco gaussiano.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> Sí, la serie tiene una tendencia **lineal creciente**. Se puede ver claramente en el gráfico de la serie original y también en la componente de tendencia de la descomposición. La pendiente es de 0.05 unidades por día, lo que acumula aproximadamente 18 unidades por año y unas 109 unidades en todo el periodo (de un valor inicial alrededor de 50 hasta valores cercanos a 160 al final). La tendencia es suave y constante, sin aceleración ni desaceleración, lo típico de un proceso lineal determinista.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Sí hay estacionalidad, y bastante clara. El periodo dominante es **anual (≈365 días)**, que se aprecia perfectamente en la componente estacional de la descomposición: el mismo patrón de subidas y bajadas se repite cada año durante los 6 años de la serie. La amplitud del patrón ronda las **±15–20 unidades** respecto al valor de tendencia. Además hay un segundo armónico (el término coseno con periodo 365/2), que hace que el patrón no sea una senoide perfecta sino ligeramente asimétrico.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> Sí, la serie tiene un ciclo de largo plazo con periodo de aproximadamente **4 años (1.461 días)** y amplitud de ±8 unidades. La diferencia con la tendencia es que el ciclo es oscilante: sube y baja de forma periódica alrededor de la tendencia, mientras que la tendencia es siempre creciente. En el gráfico de la serie original no es fácil separarlo visualmente porque queda enmascarado por la estacionalidad anual, pero en el componente de tendencia de la descomposición sí se distingue una ligera "ola" sobre la rampa lineal.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> Sí, el residuo se ajusta bastante bien a un ruido blanco gaussiano. Los estadísticos obtenidos son:
> - **Media = 0.127** — prácticamente cero, sin sesgo sistemático.
> - **Desv. estándar = 3.222** — coherente con σ_ε = 3.5 del ruido generador (la pequeña diferencia se debe a que la media móvil de la descomposición absorbe algo de ruido en la estimación de tendencia).
> - **Asimetría = −0.051** y **curtosis = −0.061** — ambos prácticamente nulos, compatibles con una normal.
>
> El **test de Jarque-Bera** dio un p-valor de **0.577**, muy por encima del umbral de 0.05, así que no se rechaza la hipótesis de normalidad. El **test ADF** dio un estadístico de −39.92 con p-valor ≈ 0, confirmando que el residuo es **estacionario** (media y varianza constantes en el tiempo). Los gráficos ACF y PACF muestran que prácticamente todos los lags caen dentro de las bandas de confianza, sin autocorrelación significativa. En conjunto, todo apunta a que la descomposición ha capturado bien la estructura temporal de la serie y lo que queda es efectivamente ruido blanco.

---

*Fin del documento de respuestas*
