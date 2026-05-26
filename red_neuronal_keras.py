"""
Red Neuronal con Keras — Violencia Intrafamiliar Colombia 2015–2024
===================================================================
Implementa tres redes neuronales (una por pregunta de investigación):
  Q1 — Regresión: predecir casos mensuales de VIF
  Q2 — Clasificación: ¿la educación reduce la gravedad del abuso?
  Q3 — Clasificación: ¿los factores contextuales clasifican la gravedad?

Genera gráficas de eficacia en graficas/redes_neuronales/
"""

import os, warnings
os.environ["KERAS_BACKEND"] = "jax"   # usa JAX (compatible con Python 3.13 arm64)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import keras
from keras import layers, callbacks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    ConfusionMatrixDisplay,
)

np.random.seed(42)
import jax
jax.config.update("jax_enable_custom_prng", False) if hasattr(jax.config, "jax_enable_custom_prng") else None

SEED = 42
OUT_DIR = "graficas/redes_neuronales"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 11})
sns.set_style("whitegrid")

# ─────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
print("Cargando dataset…")
df = pd.read_csv("V1.csv", sep=";", low_memory=False)
print(f"Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")

# ─────────────────────────────────────────────────────────────
# 2. INGENIERÍA DE CARACTERÍSTICAS (replica del notebook)
# ─────────────────────────────────────────────────────────────

# Severidad (target compartido Q2/Q3)
mapa_sev = {
    "Cero": 0, "Cero días": 0, "Sin días de incapacidad": 0,
    "1 a 30": 1,
    "31 a 90": 2, "Más de 90": 2,
    "Sin información": np.nan, "Cero días y sin información": np.nan,
}
df["severidad_n"] = (
    df["Días de Incapacidad Medicolegal"].astype(str).str.strip().map(mapa_sev)
)

# ── Features Q2: perfil víctima ──────────────────────────────
mapa_edu = {
    "Ninguna": 0, "Sin escolaridad": 0, "No aplica": 0, "Sin información": np.nan,
    "Preescolar": 1, "Educación inicial y educación preescolar": 1,
    "Básica primaria": 2, "Educación básica primaria": 2,
    "Básica secundaria": 3, "Educación básica secundaria o secundaria baja": 3,
    "Educación media o secundaria alta": 3,
    "Tecnológica": 4, "Educación técnica profesional y tecnológica": 4,
    "Profesional": 5, "Universitario": 5,
    "Especialización, Maestría o equivalente": 6, "Maestría": 6,
    "Doctorado o equivalente": 6,
}
df["edu_n"] = df["Escolaridad"].astype(str).str.strip().map(mapa_edu)
df["sexo_n"] = df["Sexo de la victima"].map({"Mujer": 1, "Hombre": 0}).fillna(0.5)
df["es_menor"] = (
    df["Grupo Mayor Menor de Edad"]
    .astype(str).str.contains("Menor", case=False, na=False)
    .astype(int)
)
mapa_ciclo = {
    "Primera infancia (0 a 5)": 0, "(0 a 4)": 0,
    "Infancia (6 a 11)": 1, "(5 a 9)": 1, "(10 a 14)": 1,
    "Adolescencia (12 a 17)": 2, "(12 a 17) Adolescencia": 2,
    "Juventud (18 a 28)": 3, "(18 a 19)": 3, "(20 a 24)": 3, "(25 a 29)": 3,
    "Adultez (29 a 59)": 4, "(30 a 34)": 4, "(35 a 39)": 4, "(40 a 44)": 4,
    "(45 a 49)": 4, "(50 a 54)": 4, "(55 a 59)": 4,
    "Persona mayor (60 y más)": 5, "(60 a 64)": 5, "(65 a 69)": 5,
    "(70 a 74)": 5, "(75 a 79)": 5, "(80 y más)": 5,
}
df["ciclo_n"] = df["Ciclo Vital"].astype(str).str.strip().map(mapa_ciclo)
df["tiene_discapacidad"] = (
    ~df["Tipo de Discapacidad"]
    .astype(str).str.strip()
    .isin({"Ninguna", "Sin información", "No aplica", "No Sabe / No Informa"})
).astype(int)
mapa_civil = {
    "Soltero (a)": 0, "Sin información": np.nan, "No aplica": np.nan,
    "Unión libre (Concubinato)": 1, "Casado (a)": 2,
    "Separado (a)": 3, "Divorciado (a)": 3, "Viudo (a)": 4,
}
df["civil_n"] = df["Estado Civil"].astype(str).str.strip().map(mapa_civil)

# ── Features Q3: contextuales ────────────────────────────────
mapa_hora = {
    "(0:00 a 2:59)": 0, "(3:00 a 5:59)": 1, "(6:00 a 8:59)": 2,
    "(9:00 a 11:59)": 3, "(12:00 a 14:59)": 4, "(15:00 a 17:59)": 5,
    "(18:00 a 20:59)": 6, "(21:00 a 23:59)": 7, "Sin información": np.nan,
}
df["hora_n"] = df["Rango de Hora del Hecho X 3 Horas"].astype(str).str.strip().map(mapa_hora)
df["es_noche"] = df["hora_n"].isin([0, 1, 7]).astype(float)
df["es_noche"] = df["es_noche"].where(df["hora_n"].notna(), np.nan)
df["zona_n"] = df["Zona del Hecho"].map({"Cabecera municipal": 1}).fillna(0)
df["zona_n"] = df["zona_n"].where(
    ~df["Zona del Hecho"].astype(str).isin({"Sin información", "No aplica"}), np.nan
)
escenarios_top = df["Escenario del Hecho"].value_counts().head(8).index.tolist()
le_escenario = LabelEncoder()
df["escenario_str"] = df["Escenario del Hecho"].astype(str).apply(
    lambda x: x if x in escenarios_top else "Otro"
)
df["escenario_n"] = le_escenario.fit_transform(df["escenario_str"])
mapa_agresor = {
    "Pareja (Cónyuge, Compañero permanente)": 5,
    "Ex pareja (Ex Cónyuge, Ex Compañero permanente)": 4,
    "Padre": 3, "Madre": 3, "Padrastro": 3, "Madrastra": 3,
    "Hijo (a)": 2, "Hermano (a)": 2,
    "Otro familiar": 1, "Sin información": np.nan, "No aplica": np.nan,
}
df["agresor_n"] = df["Presunto Agresor Detallado"].astype(str).str.strip().map(mapa_agresor)
df["agresor_n"] = df["agresor_n"].fillna(
    df["Presunto Agresor Detallado"].astype(str).apply(
        lambda x: 1 if ("familiar" in x.lower() or "pariente" in x.lower()) else np.nan
    )
)
mapa_factor = {
    "Intolerancia, machismo": 1, "Celos": 2, "Alcoholismo, Drogadicción": 3,
    "Problemas económicos": 1, "Problemas de convivencia": 1,
    "Sin información": np.nan, "No aplica": np.nan, "No Sabe / No Informa": np.nan,
}
df["factor_n"] = df["Factor Desencadenante de la Agresión"].astype(str).str.strip().map(mapa_factor)
mapa_mec = {
    "Cortante": 3, "Cortocontundente": 3, "Cortopunzante": 3, "Punzante": 3,
    "Contundente": 2, "Trauma de miembros": 2,
    "Abrasivo": 1, "Químico": 2, "Térmico": 1, "Sin información": np.nan,
}
df["mecanismo_n"] = df["Mecanismo Causal de la Lesión no Fatal"].astype(str).str.strip().map(mapa_mec)
df["mecanismo_n"] = df["mecanismo_n"].fillna(
    df["Mecanismo Causal de la Lesión no Fatal"].astype(str).apply(
        lambda x: 2 if "contund" in x.lower() else (3 if "cort" in x.lower() else np.nan)
    )
)

# ── Features Q1: serie temporal ──────────────────────────────
mapa_mes = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
    "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12,
}
df["mes_num"] = df["Mes del hecho"].map(mapa_mes)
ts = (
    df.dropna(subset=["mes_num"])
    .groupby(["Año del hecho", "mes_num"])
    .size()
    .reset_index(name="casos")
)
ts = ts.sort_values(["Año del hecho", "mes_num"]).reset_index(drop=True)
ts["t"] = np.arange(1, len(ts) + 1)
ts["mes_sin"] = np.sin(2 * np.pi * ts["mes_num"] / 12)
ts["mes_cos"] = np.cos(2 * np.pi * ts["mes_num"] / 12)
ts["trimestre"] = (ts["mes_num"] - 1) // 3 + 1
ts["es_pandemia"] = ts["Año del hecho"].isin([2020, 2021]).astype(int)
ts["es_post_pandemia"] = (ts["Año del hecho"] >= 2022).astype(int)

# ─────────────────────────────────────────────────────────────
# 3. DEFINICIÓN DE FEATURES Y DATASETS
# ─────────────────────────────────────────────────────────────
FEATS_Q1 = ["t", "mes_sin", "mes_cos", "trimestre", "es_pandemia", "es_post_pandemia"]
FEATS_Q2 = ["edu_n", "sexo_n", "es_menor", "ciclo_n", "tiene_discapacidad", "civil_n"]
FEATS_Q3 = ["hora_n", "es_noche", "zona_n", "escenario_n", "agresor_n", "factor_n", "mecanismo_n"]
TARGET = "severidad_n"

df_q2 = df.dropna(subset=FEATS_Q2 + [TARGET]).reset_index(drop=True)
df_q3 = df.dropna(subset=FEATS_Q3 + [TARGET]).reset_index(drop=True)

MAX_SAMPLE = 80_000
for name, dset in [("Q2", df_q2), ("Q3", df_q3)]:
    print(f"Registros {name}: {len(dset):,}")

# ─────────────────────────────────────────────────────────────
# 4. CONSTRUCCIÓN DE MODELOS KERAS
# ─────────────────────────────────────────────────────────────

def build_model_clf(n_features: int, n_classes: int) -> keras.Model:
    """Red densa para clasificación multiclase."""
    inp = keras.Input(shape=(n_features,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_reg(n_features: int) -> keras.Model:
    """Red densa ligera para regresión en series cortas."""
    inp = keras.Input(shape=(n_features,))
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-3))(inp)
    x = layers.Dense(16, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse", metrics=["mae"])
    return model


CBACKS = [
    callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
    callbacks.ReduceLROnPlateau(patience=7, factor=0.5, verbose=0),
]

# ─────────────────────────────────────────────────────────────
# 5. ENTRENAMIENTO — Q1 (REGRESIÓN)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Q1 — Red Neuronal: Predicción de casos mensuales de VIF")
print("=" * 55)

# Split temporal: entrenar hasta 2021, evaluar 2022-2024 (igual que en el notebook)
ts_tr = ts[ts["Año del hecho"] <= 2021]
ts_te = ts[ts["Año del hecho"] >= 2022]
X_tr1, y_tr1 = ts_tr[FEATS_Q1].values, ts_tr["casos"].values
X_te1, y_te1 = ts_te[FEATS_Q1].values, ts_te["casos"].values
sc1 = StandardScaler()
X_tr1_s = sc1.fit_transform(X_tr1)
X_te1_s = sc1.transform(X_te1)

model_q1 = build_model_reg(X_tr1_s.shape[1])
history_q1 = model_q1.fit(
    X_tr1_s, y_tr1,
    validation_data=(X_te1_s, y_te1),
    epochs=500, batch_size=8,
    callbacks=CBACKS, verbose=0,
)

pred_q1 = model_q1.predict(X_te1_s, verbose=0).flatten()
mae_q1  = mean_absolute_error(y_te1, pred_q1)
rmse_q1 = np.sqrt(mean_squared_error(y_te1, pred_q1))
r2_q1   = r2_score(y_te1, pred_q1)
print(f"MAE : {mae_q1:.1f}  |  RMSE: {rmse_q1:.1f}  |  R²: {r2_q1:.4f}")

# ─────────────────────────────────────────────────────────────
# 6. ENTRENAMIENTO — Q2 (CLASIFICACIÓN: PERFIL VÍCTIMA)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Q2 — Red Neuronal: Educación vs Severidad del abuso")
print("=" * 55)

X_q2 = df_q2[FEATS_Q2].values
y_q2 = df_q2[TARGET].values.astype(int)

if len(X_q2) > MAX_SAMPLE:
    idx = np.random.choice(len(X_q2), MAX_SAMPLE, replace=False)
    X_q2, y_q2 = X_q2[idx], y_q2[idx]

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_q2, y_q2, test_size=0.2, random_state=SEED, stratify=y_q2
)
sc2 = StandardScaler()
X_tr2_s = sc2.fit_transform(X_tr2)
X_te2_s = sc2.transform(X_te2)

model_q2 = build_model_clf(X_tr2_s.shape[1], n_classes=3)
history_q2 = model_q2.fit(
    X_tr2_s, y_tr2,
    validation_data=(X_te2_s, y_te2),
    epochs=150, batch_size=512,
    callbacks=CBACKS, verbose=0,
    class_weight={0: 1.0, 1: 2.0, 2: 3.0},
)

pred_q2 = model_q2.predict(X_te2_s, verbose=0).argmax(axis=1)
acc_q2 = accuracy_score(y_te2, pred_q2)
f1_q2  = f1_score(y_te2, pred_q2, average="weighted", zero_division=0)
print(f"Accuracy: {acc_q2:.4f}  |  F1 ponderado: {f1_q2:.4f}")

# ─────────────────────────────────────────────────────────────
# 7. ENTRENAMIENTO — Q3 (CLASIFICACIÓN: FACTORES CONTEXTUALES)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Q3 — Red Neuronal: Factores contextuales vs Severidad")
print("=" * 55)

X_q3 = df_q3[FEATS_Q3].values
y_q3 = df_q3[TARGET].values.astype(int)

if len(X_q3) > MAX_SAMPLE:
    idx = np.random.choice(len(X_q3), MAX_SAMPLE, replace=False)
    X_q3, y_q3 = X_q3[idx], y_q3[idx]

X_tr3, X_te3, y_tr3, y_te3 = train_test_split(
    X_q3, y_q3, test_size=0.2, random_state=SEED, stratify=y_q3
)
sc3 = StandardScaler()
X_tr3_s = sc3.fit_transform(X_tr3)
X_te3_s = sc3.transform(X_te3)

model_q3 = build_model_clf(X_tr3_s.shape[1], n_classes=3)
history_q3 = model_q3.fit(
    X_tr3_s, y_tr3,
    validation_data=(X_te3_s, y_te3),
    epochs=150, batch_size=512,
    callbacks=CBACKS, verbose=0,
    class_weight={0: 1.0, 1: 2.0, 2: 3.0},
)

pred_q3 = model_q3.predict(X_te3_s, verbose=0).argmax(axis=1)
acc_q3 = accuracy_score(y_te3, pred_q3)
f1_q3  = f1_score(y_te3, pred_q3, average="weighted", zero_division=0)
print(f"Accuracy: {acc_q3:.4f}  |  F1 ponderado: {f1_q3:.4f}")

# ─────────────────────────────────────────────────────────────
# 8. GRÁFICAS DE EFICACIA
# ─────────────────────────────────────────────────────────────

AZUL   = "#2C3E6B"
ROJO   = "#E74C3C"
VERDE  = "#27AE60"
NARANJA = "#F39C12"
GRIS   = "#95A5A6"

# ── 8.1 Curvas de entrenamiento — Q1 ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Q1 — Curvas de entrenamiento (Regresión)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(history_q1.history["loss"],     color=AZUL,   label="Loss entrenamiento")
ax.plot(history_q1.history["val_loss"], color=ROJO,   label="Loss validación", linestyle="--")
ax.set_xlabel("Época")
ax.set_ylabel("MSE")
ax.set_title("Función de pérdida (MSE)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.plot(history_q1.history["mae"],     color=AZUL,  label="MAE entrenamiento")
ax.plot(history_q1.history["val_mae"], color=ROJO,  label="MAE validación", linestyle="--")
ax.set_xlabel("Época")
ax.set_ylabel("MAE (casos)")
ax.set_title("Error absoluto medio")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/q1_curvas_entrenamiento.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 8.2 Predicción vs Real — Q1 ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
idx_sort = np.argsort(y_te1)
ax.plot(np.arange(len(y_te1)), y_te1[idx_sort], color=AZUL,   linewidth=2, label="Valores reales")
ax.plot(np.arange(len(y_te1)), pred_q1[idx_sort], color=NARANJA, linewidth=2,
        linestyle="--", label="Predicción Keras")
ax.fill_between(
    np.arange(len(y_te1)),
    y_te1[idx_sort], pred_q1[idx_sort],
    alpha=0.15, color=GRIS,
)
ax.set_title(
    f"Q1 — Predicción vs Real  (R²={r2_q1:.3f}  |  MAE={mae_q1:.0f} casos  |  RMSE={rmse_q1:.0f})",
    fontsize=12, fontweight="bold",
)
ax.set_xlabel("Muestra de test (ordenada por valor real)")
ax.set_ylabel("N° de casos mensuales")
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/q1_prediccion_vs_real.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 8.3 Curvas de entrenamiento — Q2 ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Q2 — Curvas de entrenamiento (Clasificación: Perfil Víctima)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(history_q2.history["loss"],     color=AZUL,  label="Loss entrenamiento")
ax.plot(history_q2.history["val_loss"], color=ROJO,  label="Loss validación", linestyle="--")
ax.set_xlabel("Época"); ax.set_ylabel("Entropía cruzada")
ax.set_title("Función de pérdida"); ax.legend()
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.plot(history_q2.history["accuracy"],     color=AZUL,  label="Accuracy entrenamiento")
ax.plot(history_q2.history["val_accuracy"], color=ROJO,  label="Accuracy validación", linestyle="--")
ax.set_xlabel("Época"); ax.set_ylabel("Accuracy")
ax.set_title("Exactitud por época"); ax.legend()
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/q2_curvas_entrenamiento.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 8.4 Curvas de entrenamiento — Q3 ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Q3 — Curvas de entrenamiento (Clasificación: Factores Contextuales)", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(history_q3.history["loss"],     color=AZUL,  label="Loss entrenamiento")
ax.plot(history_q3.history["val_loss"], color=ROJO,  label="Loss validación", linestyle="--")
ax.set_xlabel("Época"); ax.set_ylabel("Entropía cruzada")
ax.set_title("Función de pérdida"); ax.legend()
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.plot(history_q3.history["accuracy"],     color=AZUL,  label="Accuracy entrenamiento")
ax.plot(history_q3.history["val_accuracy"], color=ROJO,  label="Accuracy validación", linestyle="--")
ax.set_xlabel("Época"); ax.set_ylabel("Accuracy")
ax.set_title("Exactitud por época"); ax.legend()
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/q3_curvas_entrenamiento.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 8.5 Matrices de confusión Q2 y Q3 ───────────────────────
for q, (y_te, pred, titulo, fname) in enumerate([
    (y_te2, pred_q2, "Q2 — Keras (Perfil Víctima)", "q2_confusion_keras"),
    (y_te3, pred_q3, "Q3 — Keras (Factores Contextuales)", "q3_confusion_keras"),
], start=2):
    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Leve", "Moderado", "Grave"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    acc = accuracy_score(y_te, pred)
    f1  = f1_score(y_te, pred, average="weighted", zero_division=0)
    ax.set_title(f"{titulo}\nAccuracy: {acc:.3f}  |  F1 ponderado: {f1:.3f}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()

# ── 8.6 Gráfica resumen de eficacia ─────────────────────────
fig = plt.figure(figsize=(14, 7))
fig.suptitle(
    "Resumen de Eficacia — Redes Neuronales Keras\nViolencia Intrafamiliar Colombia 2015–2024",
    fontsize=14, fontweight="bold", y=1.01,
)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

# Panel Q1 — MAE y RMSE (R² no se grafica porque es negativo para series cortas)
ax0 = fig.add_subplot(gs[0])
metricas_q1 = {"MAE": mae_q1, "RMSE": rmse_q1}
bars0 = ax0.bar(metricas_q1.keys(), metricas_q1.values(),
                color=[NARANJA, ROJO], edgecolor="white", width=0.4)
for bar, val in zip(bars0, metricas_q1.values()):
    ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
             f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax0.set_ylim(0, max(metricas_q1.values()) * 1.35)
ax0.set_title(
    f"Q1 — Regresión\n(casos mensuales)  R²={r2_q1:.2f}*",
    fontsize=11, fontweight="bold",
)
ax0.set_ylabel("Casos")
ax0.text(0.5, -0.18,
         "* R² negativo: serie corta (~120 pts)\nRed neuronal no óptima para Q1",
         ha="center", transform=ax0.transAxes, fontsize=8, color=GRIS)
ax0.spines[["top", "right"]].set_visible(False)

# Panel Q2 — F1 por clase
ax1 = fig.add_subplot(gs[1])
f1_clases_q2 = {
    "F1 Leve":     f1_score(y_te2, pred_q2, labels=[0], average="micro", zero_division=0),
    "F1 Moderado": f1_score(y_te2, pred_q2, labels=[1], average="micro", zero_division=0),
    "F1 Grave":    f1_score(y_te2, pred_q2, labels=[2], average="micro", zero_division=0),
    "F1 Global":   f1_q2,
}
colores2 = [VERDE, NARANJA, ROJO, AZUL]
bars1 = ax1.bar(f1_clases_q2.keys(), f1_clases_q2.values(), color=colores2,
                edgecolor="white", width=0.5)
for bar, val in zip(bars1, f1_clases_q2.values()):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_ylim(0, 1.15)
ax1.set_title("Q2 — Clasificación\n(Perfil Víctima)", fontsize=11, fontweight="bold")
ax1.set_ylabel("F1-Score")
ax1.tick_params(axis="x", labelsize=9)
ax1.spines[["top", "right"]].set_visible(False)

# Panel Q3 — F1 por clase
ax2 = fig.add_subplot(gs[2])
f1_clases_q3 = {
    "F1 Leve":     f1_score(y_te3, pred_q3, labels=[0], average="micro", zero_division=0),
    "F1 Moderado": f1_score(y_te3, pred_q3, labels=[1], average="micro", zero_division=0),
    "F1 Grave":    f1_score(y_te3, pred_q3, labels=[2], average="micro", zero_division=0),
    "F1 Global":   f1_q3,
}
bars2 = ax2.bar(f1_clases_q3.keys(), f1_clases_q3.values(), color=colores2,
                edgecolor="white", width=0.5)
for bar, val in zip(bars2, f1_clases_q3.values()):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_ylim(0, 1.15)
ax2.set_title("Q3 — Clasificación\n(Factores Contextuales)", fontsize=11, fontweight="bold")
ax2.set_ylabel("F1-Score")
ax2.tick_params(axis="x", labelsize=9)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/resumen_eficacia_keras.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────
# 9. REPORTE FINAL EN CONSOLA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN FINAL — REDES NEURONALES KERAS")
print("=" * 60)
print(f"Q1  Regresión   →  R²: {r2_q1:.4f}  MAE: {mae_q1:.1f}  RMSE: {rmse_q1:.1f}")
print(f"Q2  Clasificación →  Accuracy: {acc_q2:.4f}  F1 ponderado: {f1_q2:.4f}")
print(f"Q3  Clasificación →  Accuracy: {acc_q3:.4f}  F1 ponderado: {f1_q3:.4f}")
print(f"\nGráficas guardadas en: {OUT_DIR}/")
print("  q1_curvas_entrenamiento.png")
print("  q1_prediccion_vs_real.png")
print("  q2_curvas_entrenamiento.png")
print("  q2_confusion_keras.png")
print("  q3_curvas_entrenamiento.png")
print("  q3_confusion_keras.png")
print("  resumen_eficacia_keras.png")
