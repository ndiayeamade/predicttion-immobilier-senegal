"""
app.py
Application Streamlit qui prédit le prix d'un bien immobilier
au Sénégal à partir de ses caractéristiques.

Lancer avec : streamlit run app.py
Auteur : Amade Gueye Ndiaye
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------
# Génération du dataset directement en mémoire (pas de subprocess,
# pas de fichier CSV nécessaire -> plus rapide et plus fiable sur le cloud)
# ----------------------------------------------------------------------
@st.cache_data
def generer_donnees():
    np.random.seed(42)
    N = 600

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
    quartier_choisi = np.random.choice(noms, size=N, p=proba)

    superficie = np.random.normal(180, 90, N).clip(35, 800)
    chambres = np.random.choice([1, 2, 3, 4, 5, 6], N, p=[0.05, 0.20, 0.30, 0.25, 0.15, 0.05])
    salles_bain = (chambres - np.random.choice([0, 1], N, p=[0.6, 0.4])).clip(1, None)
    age_bien = np.random.exponential(8, N).clip(0, 40).round().astype(int)
    parking = np.random.choice([0, 1], N, p=[0.35, 0.65])
    piscine = np.random.choice([0, 1], N, p=[0.90, 0.10])
    etage = np.random.choice([0, 1, 2, 3, 4], N, p=[0.45, 0.25, 0.15, 0.10, 0.05])
    distance_mer = np.random.exponential(4, N).clip(0.1, 30)

    coef = np.array([quartiers_coef[q] for q in quartier_choisi])
    prix = (
        superficie * 350_000 * coef
        + chambres * 1_500_000
        + salles_bain * 1_000_000
        - age_bien * 800_000
        + parking * 3_000_000
        + piscine * 12_000_000
        - etage * 500_000
        - distance_mer * 400_000
    )
    bruit = np.random.normal(0, 8_000_000, N)
    prix = (prix + bruit).clip(5_000_000, None)

    return pd.DataFrame({
        "Quartier": quartier_choisi,
        "Superficie_m2": superficie.round(1),
        "Chambres": chambres,
        "Salles_bain": salles_bain,
        "Age_annees": age_bien,
        "Parking": parking,
        "Piscine": piscine,
        "Etage": etage,
        "Distance_mer_km": distance_mer.round(2),
        "Prix_FCFA": prix.round(0).astype(int),
    })


# ----------------------------------------------------------------------
# Entraînement du modèle directement en mémoire (rapide : <1s)
# ----------------------------------------------------------------------
@st.cache_resource
def entrainer_modele():
    data = generer_donnees()

    encoder = LabelEncoder()
    data["Quartier_encoded"] = encoder.fit_transform(data["Quartier"])

    features = [
        "Quartier_encoded", "Superficie_m2", "Chambres", "Salles_bain",
        "Age_annees", "Parking", "Piscine", "Etage", "Distance_mer_km"
    ]
    X = data[features]
    y = data["Prix_FCFA"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    quartiers = sorted(data["Quartier"].unique())
    return model, encoder, quartiers


model, encoder, quartiers = entrainer_modele()

# ----------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Prédicteur Immobilier Sénégal",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Prédicteur de Prix Immobilier — Sénégal")
st.markdown(
    "Estimez le prix d'un bien immobilier au Sénégal grâce à un modèle "
    "d'intelligence artificielle (Random Forest) entraîné sur un jeu de "
    "données du marché local."
)

st.divider()

# ----------------------------------------------------------------------
# Formulaire de saisie des caractéristiques du bien
# ----------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    quartier = st.selectbox("📍 Quartier / Ville", quartiers)
    superficie = st.number_input("📐 Superficie (m²)", min_value=20, max_value=1000, value=150)
    chambres = st.slider("🛏️ Nombre de chambres", 1, 6, 3)
    salles_bain = st.slider("🛁 Nombre de salles de bain", 1, 5, 2)

with col2:
    age = st.number_input("📅 Âge du bien (années)", min_value=0, max_value=50, value=5)
    distance_mer = st.number_input("🌊 Distance de la mer (km)", min_value=0.0, max_value=50.0, value=3.0)
    parking = st.checkbox("🚗 Parking disponible")
    piscine = st.checkbox("🏊 Piscine")
    etage = st.slider("🏢 Étage", 0, 4, 0)

st.divider()

# ----------------------------------------------------------------------
# Prédiction
# ----------------------------------------------------------------------
if st.button("🔮 Prédire le prix", type="primary", use_container_width=True):
    quartier_encoded = encoder.transform([quartier])[0]

    X_new = pd.DataFrame([{
        "Quartier_encoded": quartier_encoded,
        "Superficie_m2": superficie,
        "Chambres": chambres,
        "Salles_bain": salles_bain,
        "Age_annees": age,
        "Parking": int(parking),
        "Piscine": int(piscine),
        "Etage": etage,
        "Distance_mer_km": distance_mer,
    }])

    prediction = model.predict(X_new)[0]

    st.success(f"💰 Prix estimé : **{prediction:,.0f} FCFA**".replace(",", " "))

    marge = 9_500_000
    st.info(
        f"Fourchette probable : entre **{prediction - marge:,.0f}** et "
        f"**{prediction + marge:,.0f} FCFA**".replace(",", " ")
    )

    st.caption(
        "⚠️ Estimation générée par un modèle entraîné sur des données "
        "synthétiques à but pédagogique. Ne pas utiliser pour une "
        "transaction réelle."
    )

st.divider()
st.caption("Projet réalisé par Amade Gueye Ndiaye — Licence Économie Appliquée, UAM Diamniadio")
