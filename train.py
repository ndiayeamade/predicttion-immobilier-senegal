"""
train.py
Entraîne un modèle de Machine Learning pour prédire le prix d'un bien
immobilier au Sénégal à partir de ses caractéristiques.

Étapes :
1. Chargement et exploration des données
2. Prétraitement (encodage du quartier)
3. Entraînement (Régression Linéaire + Forêt Aléatoire)
4. Évaluation et comparaison des modèles
5. Sauvegarde du meilleur modèle

Auteur : Amade Gueye Ndiaye
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------
# 1. Chargement des données
# ----------------------------------------------------------------------
print("📂 Chargement des données...")
data = pd.read_csv("data/house_data_senegal.csv")
print(data.head())
print(f"\nNombre de lignes : {len(data)}")
print(f"Valeurs manquantes :\n{data.isnull().sum()}")

# ----------------------------------------------------------------------
# 2. Prétraitement : encodage du quartier (variable catégorielle -> numérique)
# ----------------------------------------------------------------------
print("\n🧹 Prétraitement des données...")
encoder = LabelEncoder()
data["Quartier_encoded"] = encoder.fit_transform(data["Quartier"])

features = [
    "Quartier_encoded", "Superficie_m2", "Chambres", "Salles_bain",
    "Age_annees", "Parking", "Piscine", "Etage", "Distance_mer_km"
]
target = "Prix_FCFA"

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train : {len(X_train)} lignes | Test : {len(X_test)} lignes")

# ----------------------------------------------------------------------
# 3. Entraînement de deux modèles à comparer
# ----------------------------------------------------------------------
print("\n🤖 Entraînement des modèles...")

# Modèle 1 : Régression Linéaire (simple, rapide, bon pour comprendre)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Modèle 2 : Forêt Aléatoire (plus puissant, capte les relations non-linéaires)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12)
rf_model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 4. Évaluation et comparaison
# ----------------------------------------------------------------------
def evaluer(model, nom):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"\n--- {nom} ---")
    print(f"MAE  : {mae:,.0f} FCFA")
    print(f"RMSE : {rmse:,.0f} FCFA")
    print(f"R²   : {r2:.3f}")
    return r2

print("\n📏 Évaluation des performances :")
r2_lr = evaluer(lr_model, "Régression Linéaire")
r2_rf = evaluer(rf_model, "Forêt Aléatoire (Random Forest)")

# On garde le meilleur modèle (celui qui a le R² le plus élevé)
if r2_rf >= r2_lr:
    meilleur_modele = rf_model
    print("\n🏆 Meilleur modèle : Forêt Aléatoire")
else:
    meilleur_modele = lr_model
    print("\n🏆 Meilleur modèle : Régression Linéaire")

# ----------------------------------------------------------------------
# 5. Sauvegarde du modèle + de l'encodeur (nécessaires pour app.py)
# ----------------------------------------------------------------------
joblib.dump(meilleur_modele, "models/model.pkl")
joblib.dump(encoder, "models/encoder.pkl")
joblib.dump(list(data["Quartier"].unique()), "models/quartiers.pkl")

print("\n✅ Modèle et encodeur sauvegardés dans models/")
