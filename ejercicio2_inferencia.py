"""
Ejercicio 2: Predicción con Scikit-Learn (Regresión Lineal)
============================================================
Usa el mismo dataset Diamonds del ejercicio 1.
Variable objetivo: price (precio del diamante en USD).

Pipeline:
  1. Limpieza (duplicados, nulos)
  2. Codificación de categóricas con LabelEncoder
  3. Escalado con StandardScaler
  4. Split 80/20 (random_state=42)
  5. Regresión Lineal — sin AutoML
  6. Métricas: MAE, RMSE, R²
  7. Análisis de residuos

Salidas generadas en output/:
  - ej2_metricas_regresion.txt
  - ej2_residuos.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLS_CAT = ["cut", "color", "clarity"]
TARGET = "price"


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_y_limpiar() -> pd.DataFrame:

    csv_path = os.path.join(DATA_DIR, "diamonds.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró {csv_path}. "
            "Ejecutá primero ejercicio1_descriptivo.py para descargar el dataset."
        )
    df = pd.read_csv(csv_path)
    antes = len(df)
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    print(f"  Registros: {antes:,} → {len(df):,} (eliminados {antes - len(df):,})")
    return df


def codificar_y_escalar(df: pd.DataFrame):
    """
    Codifica variables categóricas con LabelEncoder y escala con StandardScaler.

    """
    df_enc = df.copy()

    # Codificación ordinal con LabelEncoder
    le = LabelEncoder()
    for col in COLS_CAT:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    X = df_enc.drop(columns=[TARGET])
    y = df_enc[TARGET].values

    # Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split estratificado no necesario para regresión; usamos random_state fijo
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def entrenar_modelo(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Entrena un modelo de Regresión Lineal.

    """
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula MAE, RMSE y R² dados valores reales y predichos. MAPE no es adecuado
    el precio puede ser 0 o cercano y arruina la métrica.
    """
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":   r2_score(y_true, y_pred),
    }


def guardar_metricas(met_train: dict, met_test: dict) -> None:

    path = os.path.join(OUTPUT_DIR, "ej2_metricas_regresion.txt")
    diff_r2 = met_train["R2"] - met_test["R2"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== METRICAS DE REGRESION LINEAL — DIAMONDS ===\n\n")

        f.write("--- Conjunto de ENTRENAMIENTO ---\n")
        for k, v in met_train.items():
            f.write(f"  {k:6s}: {v:,.4f}\n")

        f.write("\n--- Conjunto de TEST ---\n")
        for k, v in met_test.items():
            f.write(f"  {k:6s}: {v:,.4f}\n")

        f.write("\n--- ANALISIS DEL MODELO ---\n")
        f.write(f"  Diferencia R² (train - test): {diff_r2:.4f}\n\n")

        if diff_r2 > 0.10:
            f.write("! POSIBLE OVERFITTING: la caida de R2 entre train y test\n"
                    "  supera 0.10. Considerar regularizacion (Ridge/Lasso).\n")
        else:
            f.write("OK El modelo generaliza bien. La diferencia de R2 es\n"
                    "   aceptable (< 0.10).\n")

        if met_test["R2"] >= 0.85:
            calidad = "BUENA  (R2 >= 0.85)"
        elif met_test["R2"] >= 0.70:
            calidad = "ACEPTABLE  (0.70 <= R2 < 0.85)"
        else:
            calidad = "BAJA — revisar features o probar modelos no lineales"

        f.write(f"\n  Calidad del modelo: {calidad}\n")

    print(f"  Metricas guardadas: {path}")


def graficar_residuos(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Genera el gráfico de análisis de residuos con dos subplots:
      - Residuos vs. Valores Predichos
      - Distribución (histograma) de residuos
    """
    residuos = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel izquierdo: Residuos vs. Predichos ---
    axes[0].scatter(y_pred, residuos, alpha=0.25, color="steelblue", s=8)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Valores Predichos (USD)")
    axes[0].set_ylabel("Residuo (Real − Predicho)")
    axes[0].set_title("Residuos vs. Valores Predichos", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # --- Panel derecho: Distribución de residuos ---
    axes[1].hist(residuos, bins=80, color="salmon", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residuo (USD)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de Residuos", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Análisis de Residuos — Regresión Lineal (Diamonds)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ej2_residuos.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico de residuos guardado: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Función principal del ejercicio 2: pipeline completo de regresión lineal."""
    print("=" * 65)
    print("EJERCICIO 2: REGRESION LINEAL CON SCIKIT-LEARN — DIAMONDS")
    print("=" * 65)

    # 1. Carga y limpieza
    print("\n[1/4] Cargando y limpiando datos...")
    df = cargar_y_limpiar()

    # 2. Codificación, escalado y split
    print("\n[2/4] Codificando, escalando y dividiendo (80/20)...")
    X_train, X_test, y_train, y_test = codificar_y_escalar(df)

    # 3. Entrenamiento
    print("\n[3/4] Entrenando modelo de Regresión Lineal...")
    modelo = entrenar_modelo(X_train, y_train)

    # 4. Evaluación
    print("\n[4/4] Evaluando modelo...")
    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)

    met_train = calcular_metricas(y_train, y_pred_train)
    met_test  = calcular_metricas(y_test, y_pred_test)

    print("\n  --- Métricas TRAIN ---")
    for k, v in met_train.items():
        print(f"    {k:6s}: {v:,.4f}")

    print("\n  --- Métricas TEST ---")
    for k, v in met_test.items():
        print(f"    {k:6s}: {v:,.4f}")

    # 5. Salidas
    guardar_metricas(met_train, met_test)
    graficar_residuos(y_test, y_pred_test)

    print("\n[OK] Ejercicio 2 completado. Revisar carpeta output/")


if __name__ == "__main__":
    main()
