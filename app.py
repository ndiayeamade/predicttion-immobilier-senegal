"""
app.py — Prédicteur de Prix Immobilier au Sénégal
Auteur : Amade Gueye Ndiaye

Fichier UNIQUE et AUTONOME :
- pas de subprocess
- pas de fichiers externes à générer
- pas de modèle à charger depuis le disque
Tout est calculé en mémoire, à chaque démarrage, en moins de 2 secondes.
"""

import streamlit as st

# 1) TOUJOURS en premier, avant tout autre appel à "st."
st.set_page_config(page_title="Prédicteur Immobilier Sénégal", page_icon="🏠")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


@st.cache_resource
def charger_modele():
    """Génère les données ET entraîne le modèle, une seule fois, en mémoire."""
    rng = np.random.default_rng(42)
    n = 600

    quartiers_coef = {
        "Almadies": 1.00, "Ngor": 0.95, "Mermoz": 0.85, "Sacré-Cœur": 0.80,
        "Plateau": 0.90, "Yoff": 0.70, "Ouakam": 0.75, "Liberté 6": 0.65,
        "Parcelles Assainies": 0.55, "Guédiawaye": 0.40, "Pikine": 0.38,
        "Diamniadio": 0.50, "Rufisque": 0.42, "Thiès": 0.35, "Fatick": 0.28,
        "Mbour": 0.45, "Saly": 0.60,
    }
    noms = list(quartiers_coef.keys())
    poids = np.array(list(quartiers_coef.values()))
    proba = (1 / poids)
    proba = proba / proba.sum()
    quartier = rng.choice(noms, size=n, p=proba)

    superficie = np.clip(rng.normal(180, 90, n), 35, 800)
    chambres = rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.05, 0.20, 0.30, 0.25, 0.15, 0.05])
    salles_bain = np.clip(chambres - rng.choice([0, 1], n, p=[0.6, 0.4]), 1, None)
    age = np.clip(rng.exponential(8, n).round(), 0, 40)
    parking = rng.choice([0, 1], n, p=[0.35, 0.65])
    piscine = rng.choice([0, 1], n, p=[0.90, 0.10])
    etage = rng.choice([0, 1, 2, 3, 4], n, p=[0.45, 0.25, 0.15, 0.10, 0.05])
    dist_mer = np.clip(rng.exponential(4, n), 0.1, 30)

    coef = np.array([quartiers_coef[q] for q in quartier])
    prix = (
        superficie * 350_000 * coef
        + chambres * 1_500_000
        + salles_bain * 1_000_000
        - age * 800_000
        + parking * 3_000_000
        + piscine * 12_000_000
        - etage * 500_000
        - dist_mer * 400_000
        + rng.normal(0, 8_000_000, n)
    )
    prix = np.clip(prix, 5_000_000, None)

    df = pd.DataFrame({
        "Quartier": quartier, "Superficie_m2": superficie.round(1),
        "Chambres": chambres, "Salles_bain": salles_bain, "Age_annees": age,
        "Parking": parking, "Piscine": piscine, "Etage": etage,
        "Distance_mer_km": dist_mer.round(2), "Prix_FCFA": prix.round(0),
    })

    encoder = LabelEncoder()
    df["Quartier_encoded"] = encoder.fit_transform(df["Quartier"])

    cols = ["Quartier_encoded", "Superficie_m2", "Chambres", "Salles_bain",
            "Age_annees", "Parking", "Piscine", "Etage", "Distance_mer_km"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[cols], df["Prix_FCFA"], test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    return model, encoder, sorted(df["Quartier"].unique())


# 2) Charger le modèle (mis en cache : ne s'exécute qu'une fois)
model, encoder, quartiers = charger_modele()

# 3) Interface utilisateur
st.title("🏠 Prédicteur de Prix Immobilier — Sénégal")
st.write(
    "Estimez le prix d'un bien immobilier au Sénégal grâce à un modèle "
    "de Machine Learning (Random Forest)."
)

col1, col2 = st.columns(2)
with col1:
    quartier = st.selectbox("📍 Quartier", quartiers)
    superficie = st.number_input("📐 Superficie (m²)", 20, 1000, 150)
    chambres = st.slider("🛏️ Chambres", 1, 6, 3)
    salles_bain = st.slider("🛁 Salles de bain", 1, 5, 2)
with col2:
    age = st.number_input("📅 Âge (années)", 0, 50, 5)
    dist_mer = st.number_input("🌊 Distance mer (km)", 0.0, 50.0, 3.0)
    parking = st.checkbox("🚗 Parking")
    piscine = st.checkbox("🏊 Piscine")
    etage = st.slider("🏢 Étage", 0, 4, 0)

if st.button("🔮 Prédire le prix", type="primary", use_container_width=True):
    x = pd.DataFrame([{
        "Quartier_encoded": encoder.transform([quartier])[0],
        "Superficie_m2": superficie, "Chambres": chambres,
        "Salles_bain": salles_bain, "Age_annees": age,
        "Parking": int(parking), "Piscine": int(piscine),
        "Etage": etage, "Distance_mer_km": dist_mer,
    }])
    prix_estime = model.predict(x)[0]
    st.success(f"💰 Prix estimé : **{prix_estime:,.0f} FCFA**".replace(",", " "))
    marge = 9_500_000
    st.info(
        f"Fourchette : {prix_estime - marge:,.0f} — {prix_estime + marge:,.0f} FCFA".replace(",", " ")
    )

st.caption("Projet d'Amade Gueye Ndiaye — UAM Diamniadio · données synthétiques à but pédagogique")
