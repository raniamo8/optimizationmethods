import matplotlib
matplotlib.use('Agg')  # Keine GUI nötig – PNG-Bilder werden direkt gerendert

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# 1. Daten laden
df = pd.read_csv("data.csv")

X = df.drop("Bankrupt?", axis=1)
y = df["Bankrupt?"]

# 2. Aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Standardisierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Logistische Regression + GridSearchCV mit Multithreading und Fortschrittsanzeige
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

log_reg = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(
    log_reg,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

print("🔎 Starte GridSearchCV...")
grid_search.fit(X_train_scaled, y_train)
print("✅ GridSearch abgeschlossen!")

# 5. Beste Parameter ausgeben
best_params = grid_search.best_params_
print("\n🎯 Beste Parameter:", best_params)

# 6. Modellbewertung
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# 7. Konfusionsmatrix visualisieren & speichern
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Konfusionsmatrix")
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Wahre Klasse")
plt.tight_layout()
plt.savefig("konfusionsmatrix.png", dpi=300)
plt.close()
print("✅ Konfusionsmatrix als 'konfusionsmatrix.png' gespeichert.")

# 8. ROC-Kurve visualisieren & speichern
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_test, y_proba):.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Zufallsmodell')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-Kurve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.close()
print("✅ ROC-Kurve als 'roc_curve.png' gespeichert.")

# 9. Feature-Wichtigkeit (nur bei L1 verfügbar)
if best_params["penalty"] == "l1":
    feature_importance = pd.Series(
        np.abs(best_model.coef_[0]),
        index=X.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=feature_importance.values[:15],
        y=feature_importance.index[:15],
        palette="viridis"
    )
    plt.title("Wichtigste Merkmale (Top 15)")
    plt.xlabel("Betrag der Koeffizienten")
    plt.ylabel("Merkmal")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    plt.close()
    print("✅ Feature-Wichtigkeit als 'feature_importance.png' gespeichert.")
else:
    print("\nℹ️ Feature-Wichtigkeit nur für L1-Penalty verfügbar – aktuelles Modell nutzt L2.")

print("\n🎉 Alles fertig! Alle Visualisierungen wurden als PNG-Dateien gespeichert.")
