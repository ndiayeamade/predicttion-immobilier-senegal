"""
explore.py
Exploration visuelle du jeu de données immobilier sénégalais.
Génère des graphiques sauvegardés dans screenshots/.

Auteur : Amade Gueye Ndiaye
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
data = pd.read_csv("data/house_data_senegal.csv")

# 1. Relation superficie / prix
plt.figure(figsize=(8, 5))
plt.scatter(data["Superficie_m2"], data["Prix_FCFA"], alpha=0.5, color="#1f77b4")
plt.xlabel("Superficie (m²)")
plt.ylabel("Prix (FCFA)")
plt.title("Relation entre la superficie et le prix")
plt.tight_layout()
plt.savefig("screenshots/superficie_vs_prix.png", dpi=120)
plt.close()

# 2. Prix moyen par quartier
plt.figure(figsize=(10, 6))
prix_moyen = data.groupby("Quartier")["Prix_FCFA"].mean().sort_values(ascending=False)
prix_moyen.plot(kind="bar", color="#2ca02c")
plt.ylabel("Prix moyen (FCFA)")
plt.title("Prix moyen par quartier")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("screenshots/prix_par_quartier.png", dpi=120)
plt.close()

# 3. Distribution des prix
plt.figure(figsize=(8, 5))
sns.histplot(data["Prix_FCFA"], bins=30, kde=True, color="#ff7f0e")
plt.xlabel("Prix (FCFA)")
plt.title("Distribution des prix immobiliers")
plt.tight_layout()
plt.savefig("screenshots/distribution_prix.png", dpi=120)
plt.close()

# 4. Matrice de corrélation
plt.figure(figsize=(8, 6))
num_cols = ["Superficie_m2", "Chambres", "Salles_bain", "Age_annees",
            "Parking", "Piscine", "Etage", "Distance_mer_km", "Prix_FCFA"]
sns.heatmap(data[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig("screenshots/correlation.png", dpi=120)
plt.close()

print("✅ 4 graphiques sauvegardés dans screenshots/")
