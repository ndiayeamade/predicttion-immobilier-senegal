"""
generate_data.py
Génère un jeu de données synthétique réaliste sur le marché immobilier
au Sénégal (Dakar et environs) pour le projet de prédiction de prix.

Auteur : Amade Gueye Ndiaye
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 600  # nombre de biens immobiliers

# Quartiers/villes avec un coefficient de prix au m² (FCFA)
# Plus le coefficient est élevé, plus le quartier est cher.
quartiers = {
    "Almadies": 1.00,
    "Ngor": 0.95,
    "Mermoz": 0.85,
    "Sacré-Cœur": 0.80,
    "Plateau": 0.90,
    "Mermoz": 0.85,
    "Yoff": 0.70,
    "Ouakam": 0.75,
    "Liberté 6": 0.65,
    "Parcelles Assainies": 0.55,
    "Guédiawaye": 0.40,
    "Pikine": 0.38,
    "Diamniadio": 0.50,
    "Rufisque": 0.42,
    "Thiès": 0.35,
    "Fatick": 0.28,
    "Mbour": 0.45,
    "Saly": 0.60,
}

noms_quartiers = list(quartiers.keys())
poids_quartiers = np.array(list(quartiers.values()))

# Tirage des quartiers (pondéré pour que les quartiers chers soient un peu plus rares)
proba = (1 / poids_quartiers)
proba = proba / proba.sum()
quartier_choisi = np.random.choice(noms_quartiers, size=N, p=proba)

# Caractéristiques du bien
superficie = np.random.normal(loc=180, scale=90, size=N).clip(35, 800)  # m²
chambres = np.random.choice([1, 2, 3, 4, 5, 6], size=N, p=[0.05, 0.20, 0.30, 0.25, 0.15, 0.05])
salles_bain = (chambres - np.random.choice([0, 1], size=N, p=[0.6, 0.4])).clip(1, None)
age_bien = np.random.exponential(scale=8, size=N).clip(0, 40).round().astype(int)
parking = np.random.choice([0, 1], size=N, p=[0.35, 0.65])
piscine = np.random.choice([0, 1], size=N, p=[0.90, 0.10])
etage = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.45, 0.25, 0.15, 0.10, 0.05])
distance_mer_km = np.random.exponential(scale=4, size=N).clip(0.1, 30)

# Coefficient du quartier pour chaque ligne
coef_quartier = np.array([quartiers[q] for q in quartier_choisi])

# Prix de base au m² à Dakar (zone moyenne) : ~350 000 FCFA/m²
prix_m2_base = 350_000

# Construction du prix (en FCFA)
prix = (
    superficie * prix_m2_base * coef_quartier
    + chambres * 1_500_000
    + salles_bain * 1_000_000
    - age_bien * 800_000
    + parking * 3_000_000
    + piscine * 12_000_000
    - etage * 500_000
    - distance_mer_km * 400_000
)

# Bruit aléatoire (les prix réels ne suivent jamais une formule parfaite)
bruit = np.random.normal(loc=0, scale=8_000_000, size=N)
prix = (prix + bruit).clip(5_000_000, None)  # prix minimum 5M FCFA

# Construction du DataFrame final
data = pd.DataFrame({
    "Quartier": quartier_choisi,
    "Superficie_m2": superficie.round(1),
    "Chambres": chambres,
    "Salles_bain": salles_bain,
    "Age_annees": age_bien,
    "Parking": parking,
    "Piscine": piscine,
    "Etage": etage,
    "Distance_mer_km": distance_mer_km.round(2),
    "Prix_FCFA": prix.round(0).astype(int),
})
import os
os.makedirs("data", exist_ok=True)
data.to_csv("data/house_data_senegal.csv", index=False, encoding="utf-8")

print(f"✅ Dataset généré : {N} biens immobiliers")
print(f"📁 Fichier : data/house_data_senegal.csv")
print(data.head())
print("\nStatistiques rapides :")
print(data["Prix_FCFA"].describe())
