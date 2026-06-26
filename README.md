# 🏠 Prédicteur de Prix Immobilier — Sénégal

Application de Machine Learning qui estime le prix d'un bien immobilier
au Sénégal à partir de ses caractéristiques (quartier, superficie, nombre
de chambres, etc.).

Projet réalisé par **Amade Gueye Ndiaye**, étudiant en Licence Économie
Appliquée à l'UAM Diamniadio, dans le cadre de son apprentissage du
Machine Learning et de sa préparation au Xarala Talent Camp (IA/Data/Automatisation).

## 🎯 Objectif

Prédire le prix d'un bien immobilier (en FCFA) à partir de :
- Quartier / ville
- Superficie (m²)
- Nombre de chambres et de salles de bain
- Âge du bien
- Présence d'un parking / piscine
- Étage
- Distance par rapport à la mer

## 🧠 Stack technique

- **Python** : pandas, numpy
- **Visualisation** : matplotlib, seaborn
- **Machine Learning** : scikit-learn (Régression Linéaire + Random Forest)
- **Déploiement** : Streamlit

## 📂 Structure du projet

```
prediction-immobilier/
├── data/
│   └── house_data_senegal.csv      # Jeu de données (généré)
├── models/
│   ├── model.pkl                   # Modèle entraîné (Random Forest)
│   ├── encoder.pkl                 # Encodeur du quartier
│   └── quartiers.pkl               # Liste des quartiers
├── screenshots/                    # Graphiques d'exploration
├── generate_data.py                # Génère le dataset synthétique
├── train.py                        # Entraîne et évalue les modèles
├── explore.py                      # Génère les visualisations
├── app.py                          # Application Streamlit
├── requirements.txt
└── README.md
```

## 🚀 Installation et lancement

```bash
# 1. Cloner / récupérer le projet, puis se placer dans le dossier
cd prediction-immobilier

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Générer le dataset
python generate_data.py

# 5. Entraîner le modèle
python train.py

# 6. (Optionnel) générer les graphiques d'exploration
python explore.py

# 7. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse
`http://localhost:8501`.

> **Note :** `app.py` génère automatiquement les données et entraîne le
> modèle au premier lancement s'ils n'existent pas encore — pratique
> pour un déploiement cloud (étapes 4 et 5 deviennent optionnelles).

## ☁️ Déploiement en ligne (Streamlit Community Cloud — gratuit)

1. Pousser ce projet sur un repository GitHub (public)
2. Aller sur [share.streamlit.io](https://share.streamlit.io) et se
   connecter avec son compte GitHub
3. Cliquer sur **"New app"**, sélectionner le repository et le fichier
   `app.py`
4. Cliquer sur **"Deploy"** — l'app génère ses données et son modèle
   automatiquement au premier démarrage (1-2 minutes)
5. Récupérer le lien public (ex : `https://ton-projet.streamlit.app`)
   à mettre sur le CV / LinkedIn / dossier Xarala

## 📊 Résultats du modèle

| Modèle | MAE (FCFA) | RMSE (FCFA) | R² |
|---|---|---|---|
| Régression Linéaire | ~12 400 000 | ~15 800 000 | 0.57 |
| **Random Forest** | **~9 500 000** | **~12 300 000** | **0.74** |

Le Random Forest est retenu comme modèle final : il capture mieux les
relations non-linéaires entre les variables (notamment l'effet du
quartier, qui n'évolue pas de façon linéaire avec le prix).

## ⚠️ Limites

- Les données sont **synthétiques** (générées par une formule + bruit
  aléatoire), pas issues de vraies transactions immobilières.
- Le modèle est à but pédagogique : il ne doit pas être utilisé pour
  une décision d'achat ou de vente réelle.

## 🔭 Pistes d'amélioration

- Remplacer les données synthétiques par un vrai dataset (scraping de
  sites d'annonces immobilières sénégalaises, ou collecte via KoboCollect)
- Ajouter une carte interactive (Folium) pour visualiser les biens
- Comparer avec d'autres modèles (XGBoost, Gradient Boosting)
- Afficher un intervalle de confiance statistique autour de la prédiction
- Déployer en ligne via Streamlit Community Cloud
