"""
Ejercicio 1: Análisis Descriptivo
==================================
Dataset: Diamonds (seaborn) — 53.940 registros, 10 columnas.
Variables categóricas: cut, color, clarity
Variables numéricas continuas: carat, depth, table, price, x, y, z

Salidas generadas en output/:
  - ej1_descriptivo.csv
  - ej1_histogramas.png
  - ej1_boxplots.png
  - ej1_heatmap.png
  - ej1_categoricas.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Columnas del dataset
COLS_NUM = ["carat", "depth", "table", "price", "x", "y", "z"]
COLS_CAT = ["cut", "color", "clarity"]


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_dataset() -> pd.DataFrame:
    """
    Carga el dataset Diamonds desde seaborn y lo persiste en data/diamonds.csv.
    
    """
    csv_path = os.path.join(DATA_DIR, "diamonds.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Dataset cargado desde caché: {csv_path}")
    else:
        print("Descargando dataset Diamonds desde seaborn...")
        df = sns.load_dataset("diamonds")
        df.to_csv(csv_path, index=False)
        print(f"Dataset guardado en: {csv_path}")
    return df


def calcular_estadisticas(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Calcula media, mediana, desviación estándar, asimetría y curtosis.

    """
    tabla = pd.DataFrame({
        "Media":               df[columnas].mean(),
        "Mediana":             df[columnas].median(),
        "Desv. Estandar":      df[columnas].std(),
        "Asimetria (Skewness)":df[columnas].skew(),
        "Curtosis (Kurtosis)": df[columnas].kurt(),
        "Min":                 df[columnas].min(),
        "Max":                 df[columnas].max(),
    })
    return tabla.round(4)


def graficar_histogramas(df: pd.DataFrame, columnas: list) -> None:
    """
    Genera y guarda histogramas para las variables numéricas.

    """
    n = len(columnas)
    cols_plot = 4
    rows_plot = (n + cols_plot - 1) // cols_plot

    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(18, 4 * rows_plot))
    axes = axes.flatten()

    for i, col in enumerate(columnas):
        axes[i].hist(df[col].dropna(), bins=50, color="steelblue",
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(f"Histograma: {col}", fontsize=10, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frecuencia")
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribución de Variables Numéricas — Diamonds",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ej1_histogramas.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Histogramas guardados: {path}")


def graficar_boxplots(df: pd.DataFrame, columnas: list) -> None:
    """
    Genera y guarda boxplots para las variables numéricas.

    """
    n = len(columnas)
    cols_plot = 4
    rows_plot = (n + cols_plot - 1) // cols_plot

    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(18, 4 * rows_plot))
    axes = axes.flatten()

    for i, col in enumerate(columnas):
        bp = axes[i].boxplot(
            df[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor="lightcoral", color="darkred"),
            medianprops=dict(color="darkred", linewidth=2),
            whiskerprops=dict(color="darkred"),
            capprops=dict(color="darkred"),
            flierprops=dict(marker="o", markerfacecolor="darkred",
                            markersize=2, alpha=0.3),
        )
        axes[i].set_title(f"Boxplot: {col}", fontsize=10, fontweight="bold")
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Boxplots de Variables Numéricas — Diamonds",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ej1_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Boxplots guardados: {path}")


def graficar_heatmap(df: pd.DataFrame, columnas: list) -> None:
    """
    Genera y guarda el mapa de calor de correlaciones entre variables numéricas.

    """
    corr = df[columnas].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 10},
    )
    ax.set_title("Mapa de Calor — Correlaciones Numéricas (Diamonds)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ej1_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap guardado: {path}")


def graficar_categoricas(df: pd.DataFrame, columnas: list) -> None:
    """
    Genera y guarda gráficos de barras para las variables categóricas.

    """
    colores = ["mediumseagreen", "cornflowerblue", "mediumpurple"]
    fig, axes = plt.subplots(1, len(columnas), figsize=(18, 5))

    for i, col in enumerate(columnas):
        conteo = df[col].value_counts()
        axes[i].bar(
            conteo.index.astype(str),
            conteo.values,
            color=colores[i % len(colores)],
            edgecolor="white",
            alpha=0.85,
        )
        axes[i].set_title(f"Frecuencia: {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Cantidad")
        axes[i].tick_params(axis="x", rotation=30)
        axes[i].grid(True, axis="y", alpha=0.3)

        # Etiquetas de valor encima de cada barra
        for rect, val in zip(axes[i].patches, conteo.values):
            axes[i].text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 100,
                f"{val:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle("Variables Categóricas — Diamonds",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ej1_categoricas.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Categóricas guardadas: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Función principal del ejercicio 1: carga, análisis y visualización."""
    print("=" * 65)
    print("EJERCICIO 1: ANALISIS DESCRIPTIVO — DIAMONDS DATASET")
    print("=" * 65)

    df = cargar_dataset()
    print(f"\nShape: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    print(f"Nulos por columna:\n{df.isnull().sum().to_string()}")

    # Estadísticas descriptivas
    tabla = calcular_estadisticas(df, COLS_NUM)
    print("\n--- Estadísticas Descriptivas ---")
    print(tabla.to_string())

    # Guardar CSV
    csv_out = os.path.join(OUTPUT_DIR, "ej1_descriptivo.csv")
    tabla.to_csv(csv_out)
    print(f"\nEstadísticas guardadas: {csv_out}")

    # Gráficos
    print("\nGenerando gráficos...")
    graficar_histogramas(df, COLS_NUM)
    graficar_boxplots(df, COLS_NUM)
    graficar_heatmap(df, COLS_NUM)
    graficar_categoricas(df, COLS_CAT)

    print("\n[OK] Ejercicio 1 completado. Revisar carpeta output/")


if __name__ == "__main__":
    main()
