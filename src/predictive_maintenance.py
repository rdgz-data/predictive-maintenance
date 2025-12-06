import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
#  CARGA DE DATOS
# =========================
df = pd.read_csv("data.csv")

x = df.drop(columns=["fail"])
y = df["fail"]

print("Entradas (X):", x.shape)
print("Salidas (Y):", y.shape)

# =========================
#  TRAIN / TEST SPLIT
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
#  MODELO XGBOOST
# =========================
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

# =========================
#  MÉTRICAS
# =========================
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred, digits=3))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print(f"ROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC: {pr_auc:.3f}")

scores = cross_val_score(model, x_train, y_train, cv=5, scoring="f1")
print("F1 promedio:", scores.mean())

# =========================
#  SMOTE (opcional)
# =========================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(x_train, y_train)
print("Antes SMOTE:", y_train.value_counts())
print("Después SMOTE:", y_train_res.value_counts())

# =========================
#  VISUALIZACIONES
# =========================

# -------------------------
# 1) Matriz de confusión
# -------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# -------------------------
# 2) Curva ROC
# -------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], linestyle="--")
plt.title("Curva ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.show()

# -------------------------
# 3) Curva PR (Precision-Recall)
# -------------------------
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, linewidth=2)
plt.title("Curva Precision-Recall (PR)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

# -------------------------
# 4) Importancia de variables
# -------------------------
importances = model.feature_importances_
feat_names = x.columns

plt.figure(figsize=(8,6))
sns.barplot(x=importances, y=feat_names)
plt.title("Importancia de Variables - XGBoost")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# =========================
#  GUARDAR MODELO
# =========================
joblib.dump(model, "modelo_fallas.joblib")









