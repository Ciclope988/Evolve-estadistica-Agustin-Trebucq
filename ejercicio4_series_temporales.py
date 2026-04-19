"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 4
Análisis y Descomposición de Series Temporales
=============================================================================

DESCRIPCIÓN
-----------
En este ejercicio trabajarás con una serie temporal sintética generada con
una semilla fija. Tendrás que:

  1. Visualizar la serie completa.
  2. Descomponerla en sus componentes: Tendencia, Estacionalidad y Residuo.
  3. Analizar cada componente y responder las preguntas del fichero
     Respuestas.md (sección Ejercicio 4).
  4. Evaluar si el ruido (residuo) se ajusta a un ruido ideal (gaussiano
     con media ≈ 0 y varianza constante).

LIBRERÍAS PERMITIDAS
--------------------
  - numpy, pandas
  - matplotlib, seaborn
  - statsmodels   (para seasonal_decompose y adfuller)
  - scipy.stats   (para el test de normalidad del ruido)

SALIDAS ESPERADAS (carpeta output/)
------------------------------------
  - output/ej4_serie_original.png      → Gráfico de la serie completa
  - output/ej4_descomposicion.png      → Los 4 subgráficos de descomposición
  - output/ej4_acf_pacf.png           → Gráfico ACF y PACF del residuo
  - output/ej4_histograma_ruido.png   → Histograma + curva normal del residuo
  - output/ej4_analisis.txt            → Estadísticos numéricos del análisis

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)


# =============================================================================
# GENERACIÓN DE LA SERIE TEMPORAL SINTÉTICA — NO MODIFICAR ESTE BLOQUE
# =============================================================================

def generar_serie_temporal(semilla=42):
    """
    Genera una serie temporal sintética con componentes conocidos.
    """
    rng = np.random.default_rng(semilla)

    # Índice temporal: 6 años de datos diarios
    fechas = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
    n = len(fechas)
    t = np.arange(n)

    # --- Componentes ---
    # 1. Tendencia lineal
    tendencia = 0.05 * t + 50

    # 2. Estacionalidad anual (periodo = 365.25 días)
    estacionalidad = 15 * np.sin(2 * np.pi * t / 365.25) \
                   +  6 * np.cos(4 * np.pi * t / 365.25)

    # 3. Ciclo de largo plazo (periodo ~ 4 años = 1461 días)
    ciclo = 8 * np.sin(2 * np.pi * t / 1461)

    # 4. Ruido gaussiano
    ruido = rng.normal(loc=0, scale=3.5, size=n)

    # Serie completa (modelo aditivo)
    valores = tendencia + estacionalidad + ciclo + ruido

    serie = pd.Series(valores, index=fechas, name="valor")
    return serie


# =============================================================================
# TAREA 1 — Visualizar la serie completa
# =============================================================================

def visualizar_serie(serie):
    """
    Genera y guarda un gráfico de la serie temporal completa.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(serie.index, serie.values, color="steelblue", linewidth=0.8, alpha=0.9)
    ax.set_title("Serie Temporal Sintética — Observaciones Diarias (2018–2023)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/ej4_serie_original.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# TAREA 2 — Descomposición de la serie
# =============================================================================

def descomponer_serie(serie):
    """
    Descompone la serie en Tendencia, Estacionalidad y Residuo usando
    statsmodels.tsa.seasonal.seasonal_decompose y guarda el gráfico.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    resultado = seasonal_decompose(serie, model='additive', period=365)
    fig = resultado.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle("Descomposición Aditiva de la Serie Temporal",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig("output/ej4_descomposicion.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    return resultado


# =============================================================================
# TAREA 3 — Análisis del residuo (ruido)
# =============================================================================

def analizar_residuo(residuo):
    """
    Analiza el componente de residuo para determinar si se parece
    a un ruido ideal (gaussiano, media ≈ 0, varianza constante, sin autocorr.).
    """
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from scipy.stats import jarque_bera, norm

    # TODO: Limpia el residuo (elimina NaN al inicio/fin)
    residuo_limpio = residuo.dropna()

    # TODO: Calcula estadísticos básicos
    media     = residuo_limpio.mean()
    std       = residuo_limpio.std()
    asimetria = residuo_limpio.skew()
    curtosis  = residuo_limpio.kurtosis()

    # Test de estacionariedad (ADF)
    resultado_adf = adfuller(residuo_limpio)
    p_adf = resultado_adf[1]

    # Test de normalidad Jarque-Bera
    jb_stat, jb_p = jarque_bera(residuo_limpio)

    # TODO: Gráfico ACF y PACF del residuo → output/ej4_acf_pacf.png
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    plot_acf(residuo_limpio, lags=50, ax=axes[0])
    axes[0].set_title("ACF del Residuo", fontweight="bold")
    plot_pacf(residuo_limpio, lags=50, ax=axes[1], method="ywm")
    axes[1].set_title("PACF del Residuo", fontweight="bold")
    plt.tight_layout()
    plt.savefig("output/ej4_acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.close()

    # TODO: Histograma del residuo con curva normal superpuesta
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuo_limpio, bins=60, density=True, color="steelblue",
            edgecolor="white", alpha=0.75, label="Residuo")
    x_range = np.linspace(residuo_limpio.min(), residuo_limpio.max(), 300)
    ax.plot(x_range, norm.pdf(x_range, media, std), "r-", linewidth=2,
            label=f"Normal(μ={media:.2f}, σ={std:.2f})")
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Densidad")
    ax.set_title("Histograma del Residuo + Curva Normal Ajustada", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/ej4_histograma_ruido.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Guardar estadísticos numéricos
    with open("output/ej4_analisis.txt", "w", encoding="utf-8") as f:
        f.write("Análisis del Residuo — Serie Temporal Sintética\n")
        f.write("=" * 50 + "\n\n")
        f.write("Estadísticos básicos:\n")
        f.write(f"  Media:     {media:.6f}\n")
        f.write(f"  Std:       {std:.6f}\n")
        f.write(f"  Asimetría: {asimetria:.6f}\n")
        f.write(f"  Curtosis:  {curtosis:.6f}\n\n")
        f.write("Test de estacionariedad (ADF):\n")
        f.write(f"  Estadístico ADF: {resultado_adf[0]:.6f}\n")
        f.write(f"  p-valor:         {p_adf:.6f}\n")
        f.write(f"  Conclusión:      {'Estacionario (p<0.05)' if p_adf < 0.05 else 'No estacionario (p>=0.05)'}\n\n")
        f.write("Test de normalidad (Jarque-Bera):\n")
        f.write(f"  Estadístico JB: {jb_stat:.6f}\n")
        f.write(f"  p-valor:        {jb_p:.6f}\n")
        f.write(f"  Conclusión:     {'No se rechaza normalidad (p>0.05)' if jb_p > 0.05 else 'Se rechaza normalidad (p<=0.05)'}\n")
    print("  Estadísticos guardados: output/ej4_analisis.txt")


# =============================================================================
# MAIN — Ejecuta el pipeline completo
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("EJERCICIO 4 — Análisis de Series Temporales")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Paso 1: Generar la serie (NO modificar la semilla)
    # ------------------------------------------------------------------
    SEMILLA = 42
    serie = generar_serie_temporal(semilla=SEMILLA)

    print(f"\nSerie generada:")
    print(f"  Periodo:      {serie.index[0].date()} → {serie.index[-1].date()}")
    print(f"  Observaciones: {len(serie)}")
    print(f"  Media:         {serie.mean():.2f}")
    print(f"  Std:           {serie.std():.2f}")
    print(f"  Min / Max:     {serie.min():.2f} / {serie.max():.2f}")

    # ------------------------------------------------------------------
    # Paso 2: Visualizar la serie completa
    # ------------------------------------------------------------------
    print("\n[1/3] Visualizando la serie original...")
    visualizar_serie(serie)

    # ------------------------------------------------------------------
    # Paso 3: Descomponer
    # ------------------------------------------------------------------
    print("[2/3] Descomponiendo la serie...")
    resultado = descomponer_serie(serie)

    # ------------------------------------------------------------------
    # Paso 4: Analizar el residuo
    # ------------------------------------------------------------------
    print("[3/3] Analizando el residuo...")
    if resultado is not None:
        analizar_residuo(resultado.resid)

    # ------------------------------------------------------------------
    # Resumen de salidas esperadas
    # ------------------------------------------------------------------
    print("\nSalidas esperadas en output/:")
    salidas = [
        "ej4_serie_original.png",
        "ej4_descomposicion.png",
        "ej4_acf_pacf.png",
        "ej4_histograma_ruido.png",
        "ej4_analisis.txt",
    ]
    for s in salidas:
        existe = os.path.exists(f"output/{s}")
        estado = "✓" if existe else "✗ (pendiente)"
        print(f"  [{estado}] output/{s}")

    print("\n¡Recuerda completar las respuestas en Respuestas.md!")
