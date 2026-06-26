"""
app.py
Application Streamlit qui prédit le prix d'un bien immobilier
au Sénégal à partir de ses caractéristiques.

Lancer avec : streamlit run app.py
Auteur : Amade Gueye Ndiaye
"""

import os
import sys
import subprocess
import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------
# Génération automatique des données et du modèle si absents
# (nécessaire pour un déploiement cloud, où personne ne lance
# generate_data.py / train.py à la main avant le premier démarrage)
# ----------------------------------------------------------------------
@st.cache_resource
def charger_ou_entrainer():
    if not os.path.exists("data/house_data_senegal.csv"):
        with st.spinner("Génération du jeu de données..."):
            subprocess.run([sys.executable, "generate_data.py"], check=True)

    if not os.path.exists("models/model.pkl"):
        with st.spinner("Entraînement du modèle (premier lancement)..."):
            subprocess.run([sys.executable, "train.py"], check=True)

    model = joblib.load("models/model.pkl")
    encoder = joblib.load("models/encoder.pkl")
    quartiers = sorted(joblib.load("models/quartiers.pkl"))
    return model, encoder, quartiers

model, encoder, quartiers = charger_ou_entrainer()

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

    # Petite fourchette d'incertitude basée sur le MAE du modèle (~9.5M FCFA)
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
