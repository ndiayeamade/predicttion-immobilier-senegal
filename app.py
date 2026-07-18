"""
app.py — Référentiel National des Prix Immobiliers (RNPI) — Sénégal
Version 4 : Mode Terrain + Mode Bien Construit adapté aux agences cibles
  - Yenne Immo / Diamniadio → Terrains nus, lotissements, coopératives
  - DST Immobilier / Saly / Mbour → Villas, appartements, bord de mer
  - Djidjack / Fatick → Maisons, constructions existantes
Auteur : Amade Gueye Ndiaye — NekuImmo
"""

import streamlit as st
st.set_page_config(
    page_title="RNPI — Prix Immobilier Sénégal",
    page_icon="🏠",
    layout="wide"
)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .bloc-titre {
    background:#0F1F14; padding:1.5rem 2rem; border-radius:12px;
    border-left:5px solid #C85A1A; margin-bottom:1.5rem;
  }
  .bloc-titre h1 { color:white; font-size:1.8rem; margin:0; }
  .bloc-titre p  { color:#8AB0A0; margin:0.3rem 0 0; font-size:0.95rem; }
  .mode-badge {
    display:inline-block; padding:0.4rem 1.2rem;
    border-radius:100px; font-weight:700; font-size:0.9rem;
    margin-bottom:1rem;
  }
  .mode-terrain   { background:#D4A853; color:#0F1F14; }
  .mode-construit { background:#C85A1A; color:white; }
  .prix-card {
    background:#0F1F14; padding:1.5rem; border-radius:12px;
    text-align:center; margin-top:0.5rem;
  }
  .montant  { font-size:2.2rem; font-weight:800; color:#C85A1A; }
  .fourchette { font-size:0.9rem; color:#7A9A84; margin-top:0.3rem; }
  .facteur-card {
    background:white; border-radius:10px; padding:0.7rem 1rem;
    margin-bottom:0.5rem; border-left:4px solid #C85A1A;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);
  }
  .alerte-rouge  { background:#FDECEA; border-left:4px solid #C0392B;
                   padding:0.7rem 1rem; border-radius:8px; margin-top:0.5rem; }
  .alerte-orange { background:#FEF3E2; border-left:4px solid #C85A1A;
                   padding:0.7rem 1rem; border-radius:8px; margin-top:0.5rem; }
  .alerte-verte  { background:#EAF4EE; border-left:4px solid #2A6E3F;
                   padding:0.7rem 1rem; border-radius:8px; margin-top:0.5rem; }
  .alerte-bleue  { background:#EBF5FB; border-left:4px solid #1B4F72;
                   padding:0.7rem 1rem; border-radius:8px; margin-top:0.5rem; }
  .info-agence {
    background:#F7F3EC; border:1.5px solid #E8E0D4;
    border-radius:10px; padding:0.8rem 1rem; margin-bottom:1rem;
    font-size:0.88rem; color:#5C6670;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MODÈLE TERRAIN
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def modele_terrain():
    rng = np.random.default_rng(42)
    n = 800

    # Prix au m² par zone (FCFA) — calibré sur données réelles Diamniadio/Yenne
    zones_coef = {
        "Diamniadio (Pôle urbain)": 35_000,
        "Diamniadio (Hors pôle)": 20_000,
        "Yenne / Daga (Zone AIBD)": 18_000,
        "Saly / Mbour": 22_000,
        "Fatick (Centre-ville)": 8_000,
        "Fatick (Périphérie)": 5_000,
        "Thiès": 12_000,
        "Rufisque": 15_000,
        "Guédiawaye / Pikine": 25_000,
        "Parcelles Assainies": 40_000,
        "Plateau / Centre Dakar": 90_000,
        "Almadies / Ngor": 120_000,
        "Mermoz / Sacré-Cœur": 70_000,
    }
    tf_coef     = {"TF individuel":1.20, "Délibération double signature":1.00,
                   "Délibération simple":0.85, "Attestation":0.75}
    viab_coef   = {"Eau + Électricité":1.15, "Électricité seulement":1.05,
                   "Eau seulement":1.05, "Non viabilisé":1.00}
    inond_coef  = {"Aucun":1.00, "Faible":0.93, "Modéré":0.85, "Élevé":0.72}
    zone_coef   = {"Lotissement officiel":1.20, "Zone habitée":1.10,
                   "Périphérie":1.00, "Zone rurale":0.80}

    noms = list(zones_coef.keys())
    poids = np.array(list(zones_coef.values()))
    proba = poids/poids.sum()

    zone       = rng.choice(noms, n, p=proba)
    titre_f    = rng.choice(list(tf_coef.keys()), n, p=[0.30,0.35,0.20,0.15])
    viabilise  = rng.choice(list(viab_coef.keys()), n, p=[0.50,0.15,0.10,0.25])
    inondation = rng.choice(list(inond_coef.keys()), n, p=[0.50,0.25,0.15,0.10])
    zone_type  = rng.choice(list(zone_coef.keys()), n, p=[0.30,0.30,0.25,0.15])
    superficie = np.clip(rng.exponential(250, n), 100, 2000)
    vue_mer    = rng.choice([0,1], n, p=[0.75,0.25])
    prox_route = rng.choice([0,1], n, p=[0.40,0.60])
    prox_aibd  = rng.choice([0,1], n, p=[0.80,0.20])
    dist_mer   = np.clip(rng.exponential(5, n), 0.1, 50)

    px_m2   = np.array([zones_coef[z] for z in zone])
    c_tf    = np.array([tf_coef[t] for t in titre_f])
    c_vi    = np.array([viab_coef[v] for v in viabilise])
    c_in    = np.array([inond_coef[i] for i in inondation])
    c_zt    = np.array([zone_coef[z] for z in zone_type])

    prix = (
        superficie * px_m2 * c_tf * c_vi * c_in * c_zt
        + vue_mer * 3_000_000
        + prox_route * 1_500_000
        + prox_aibd * 2_000_000
        - dist_mer * 100_000
        + rng.normal(0, 500_000 * np.sqrt(superficie/100), n)
    )
    prix = np.clip(prix, 1_000_000, None)

    df = pd.DataFrame({
        "Zone":zone, "Titre_foncier":titre_f, "Viabilisation":viabilise,
        "Inondation":inondation, "Zone_type":zone_type,
        "Superficie_m2":superficie.round(0),
        "Vue_mer":vue_mer, "Prox_route":prox_route,
        "Prox_aibd":prox_aibd, "Distance_mer_km":dist_mer.round(1),
        "Prix":prix.round(0),
    })

    enc_z  = LabelEncoder(); df["Zone_enc"]  = enc_z.fit_transform(df["Zone"])
    enc_tf = OrdinalEncoder(categories=[["Attestation","Délibération simple",
                                          "Délibération double signature","TF individuel"]])
    df["TF_enc"] = enc_tf.fit_transform(df[["Titre_foncier"]]).astype(int)
    enc_vi = OrdinalEncoder(categories=[["Non viabilisé","Eau seulement",
                                          "Électricité seulement","Eau + Électricité"]])
    df["Vi_enc"] = enc_vi.fit_transform(df[["Viabilisation"]]).astype(int)
    enc_in = OrdinalEncoder(categories=[["Élevé","Modéré","Faible","Aucun"]])
    df["In_enc"] = enc_in.fit_transform(df[["Inondation"]]).astype(int)
    enc_zt = LabelEncoder(); df["ZT_enc"] = enc_zt.fit_transform(df["Zone_type"])

    FEAT = ["Zone_enc","TF_enc","Vi_enc","In_enc","ZT_enc",
            "Superficie_m2","Vue_mer","Prox_route","Prox_aibd","Distance_mer_km"]
    LABS = ["Zone","Titre foncier","Viabilisation","Risque inondation","Type de zone",
            "Superficie","Vue mer","Prox. route","Prox. AIBD","Distance mer"]

    X=df[FEAT]; y=df["Prix"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    m=RandomForestRegressor(n_estimators=120,max_depth=12,random_state=42)
    m.fit(Xtr,ytr)

    return (m, enc_z, enc_tf, enc_vi, enc_in, enc_zt,
            sorted(df["Zone"].unique()), list(tf_coef.keys()),
            list(viab_coef.keys()), list(inond_coef.keys()), list(zone_coef.keys()),
            FEAT, LABS,
            r2_score(yte,m.predict(Xte)), mean_absolute_error(yte,m.predict(Xte)))


# ══════════════════════════════════════════════════════════════════
# MODÈLE BIEN CONSTRUIT
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def modele_construit():
    rng = np.random.default_rng(42)
    n = 1000

    quartiers_coef = {
        "Almadies":1.00,"Ngor":0.95,"Mermoz":0.85,"Sacré-Cœur":0.80,
        "Plateau":0.90,"Yoff":0.70,"Ouakam":0.75,"Liberté 6":0.65,
        "Parcelles Assainies":0.55,"Guédiawaye":0.40,"Pikine":0.38,
        "Diamniadio":0.50,"Rufisque":0.42,"Thiès":0.35,
        "Fatick":0.28,"Mbour":0.45,"Saly":0.60,
    }
    type_coef   = {"Villa":1.20,"Appartement":0.90,"Maison":1.00,"Studio":0.60}
    etat_coef   = {"Neuf":1.12,"Bon état":1.00,"À rénover":0.82}
    tf_coef     = {"TF plein":1.10,"Bail emphytéotique":1.00,"Attestation":0.88}
    inond_coef  = {"Aucun":1.00,"Faible":0.96,"Modéré":0.90,"Élevé":0.80}
    trans_coef  = {"Vente":1.00,"Location":0.0082}

    noms=list(quartiers_coef.keys()); poids=np.array(list(quartiers_coef.values()))
    proba=(1/poids); proba/=proba.sum()

    quartier  =rng.choice(noms,n,p=proba)
    type_bien =rng.choice(list(type_coef.keys()),n,p=[0.30,0.35,0.25,0.10])
    transaction=rng.choice(["Vente","Location"],n,p=[0.55,0.45])
    etat      =rng.choice(list(etat_coef.keys()),n,p=[0.25,0.55,0.20])
    titre_f   =rng.choice(list(tf_coef.keys()),n,p=[0.45,0.30,0.25])
    inondation=rng.choice(list(inond_coef.keys()),n,p=[0.50,0.25,0.15,0.10])

    superficie =np.clip(rng.normal(150,70,n),30,800)
    chambres   =rng.choice(range(1,11),n,p=[0.05,0.18,0.28,0.22,0.12,0.07,0.04,0.02,0.01,0.01])
    salles_bain=np.clip(chambres-rng.choice([0,1],n,p=[0.6,0.4]),1,None)
    age        =np.clip(rng.exponential(8,n).round(),0,60)
    parking    =rng.choice([0,1],n,p=[0.35,0.65])
    piscine    =rng.choice([0,1],n,p=[0.90,0.10])
    jardin     =rng.choice([0,1],n,p=[0.50,0.50])
    securite   =rng.choice([0,1],n,p=[0.45,0.55])
    clim       =rng.choice([0,1],n,p=[0.40,0.60])
    groupe     =rng.choice([0,1],n,p=[0.55,0.45])
    etage      =rng.choice([0,1,2,3,4,5],n,p=[0.40,0.22,0.15,0.10,0.08,0.05])
    dist_mer   =np.clip(rng.exponential(4,n),0.1,30)
    vue_mer    =rng.choice([0,1],n,p=[0.80,0.20])
    prox_ecole =rng.choice([0,1],n,p=[0.40,0.60])
    prox_marche=rng.choice([0,1],n,p=[0.35,0.65])
    meuble     =rng.choice([0,1],n,p=[0.65,0.35])

    coef_q =np.array([quartiers_coef[q] for q in quartier])
    coef_t =np.array([type_coef[t] for t in type_bien])
    coef_e =np.array([etat_coef[e] for e in etat])
    coef_tf=np.array([tf_coef[t] for t in titre_f])
    coef_in=np.array([inond_coef[i] for i in inondation])
    coef_tx=np.array([trans_coef[t] for t in transaction])

    valeur=(
        superficie*350000*coef_q*coef_t*coef_e*coef_tf*coef_in
        +chambres*1500000+salles_bain*800000-age*600000
        +parking*3000000+piscine*12000000+jardin*4000000
        +securite*2500000+clim*1800000+groupe*2200000
        -etage*300000-dist_mer*350000
        +vue_mer*8000000
        +prox_ecole*1200000+prox_marche*900000+meuble*2000000
        +rng.normal(0,5000000,n)
    )
    valeur=np.clip(valeur,3000000,None)
    prix=valeur*coef_tx

    df=pd.DataFrame({
        "Quartier":quartier,"Type_bien":type_bien,"Transaction":transaction,
        "Etat":etat,"Titre_foncier":titre_f,"Inondation":inondation,
        "Superficie_m2":superficie.round(1),"Chambres":chambres,
        "Salles_bain":salles_bain,"Age_annees":age.astype(int),
        "Parking":parking,"Piscine":piscine,"Jardin":jardin,
        "Securite":securite,"Climatisation":clim,"Groupe_electrogene":groupe,
        "Etage":etage,"Distance_mer_km":dist_mer.round(2),
        "Vue_mer":vue_mer,"Prox_ecole":prox_ecole,
        "Prox_marche":prox_marche,"Meuble":meuble,"Prix":prix.round(0),
    })

    enc_q =LabelEncoder(); df["Q_enc"] =enc_q.fit_transform(df["Quartier"])
    enc_t =LabelEncoder(); df["T_enc"] =enc_t.fit_transform(df["Type_bien"])
    enc_tx=LabelEncoder(); df["TX_enc"]=enc_tx.fit_transform(df["Transaction"])
    enc_e =OrdinalEncoder(categories=[["À rénover","Bon état","Neuf"]])
    df["E_enc"]=enc_e.fit_transform(df[["Etat"]]).astype(int)
    enc_tf=OrdinalEncoder(categories=[["Attestation","Bail emphytéotique","TF plein"]])
    df["TF_enc"]=enc_tf.fit_transform(df[["Titre_foncier"]]).astype(int)
    enc_in=OrdinalEncoder(categories=[["Élevé","Modéré","Faible","Aucun"]])
    df["IN_enc"]=enc_in.fit_transform(df[["Inondation"]]).astype(int)

    FEAT=["Q_enc","T_enc","TX_enc","E_enc","TF_enc","IN_enc",
          "Superficie_m2","Chambres","Salles_bain","Age_annees",
          "Parking","Piscine","Jardin","Securite","Climatisation",
          "Groupe_electrogene","Etage","Distance_mer_km","Vue_mer",
          "Prox_ecole","Prox_marche","Meuble"]
    LABS=["Quartier","Type de bien","Transaction","État","Titre foncier","Risque inondation",
          "Superficie","Chambres","Salles de bain","Âge",
          "Parking","Piscine","Jardin","Sécurité","Climatisation",
          "Groupe électrogène","Étage","Distance mer","Vue mer",
          "Prox. école","Prox. marché","Meublé"]

    X=df[FEAT].fillna(0); y=df["Prix"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    m=RandomForestRegressor(n_estimators=120,max_depth=14,random_state=42)
    m.fit(Xtr,ytr)

    return (m, enc_q, enc_t, enc_tx, enc_e, enc_tf, enc_in,
            sorted(df["Quartier"].unique()), list(type_coef.keys()),
            FEAT, LABS,
            r2_score(yte,m.predict(Xte)), mean_absolute_error(yte,m.predict(Xte)))


# Chargement des deux modèles
MT  = modele_terrain()
MC  = modele_construit()


# ══════════════════════════════════════════════════════════════════
# INTERFACE
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="bloc-titre">
  <h1>🏠 RNPI — Référentiel National des Prix Immobiliers · Sénégal</h1>
  <p>NekuImmo · Estimez le prix d'un terrain ou d'un bien construit · Prototype pédagogique</p>
</div>
""", unsafe_allow_html=True)

# ── Sélecteur de mode ──────────────────────────────────────────────
mode = st.radio(
    "Que voulez-vous estimer ?",
    ["🌿 Terrain nu / Parcelle", "🏘️ Bien construit (Villa, Appartement, Maison)"],
    horizontal=True
)
st.divider()

# ══════════════════════════════════════════════════════════════════
# MODE TERRAIN
# ══════════════════════════════════════════════════════════════════
if mode == "🌿 Terrain nu / Parcelle":
    (m_t, enc_z, enc_tf_t, enc_vi, enc_in_t, enc_zt,
     zones, titres_t, viabs, inonds_t, zone_types,
     FEAT_T, LABS_T, r2_t, mae_t) = MT

    st.markdown('<span class="mode-badge mode-terrain">🌿 Mode Terrain</span>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-agence">
      🎯 <strong>Adapté pour :</strong> Yenne Immo (Diamniadio/Daga), ventes de parcelles
      à Fatick, Mbour, Thiès · Coopératives d'habitat · Lotissements officiels
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_terrain"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("📍 Localisation")
            zone      = st.selectbox("Zone / Ville", sorted(zones))
            zone_type = st.selectbox("Type de zone", zone_types)
            superficie= st.number_input("Superficie du terrain (m²)", 50, 5000, 225)
            dist_mer  = st.number_input("Distance à la mer (km)", 0.0, 100.0, 5.0)

        with c2:
            st.subheader("📋 Statut juridique")
            titre_f   = st.selectbox("Type de papier / Titre", titres_t)
            viabilise = st.selectbox("Viabilisation", viabs)
            inondation= st.selectbox("Risque d'inondation", inonds_t)

        with c3:
            st.subheader("✨ Atouts")
            vue_mer   = st.checkbox("🌊 Vue mer")
            prox_route= st.checkbox("🛣️ Proximité route principale")
            prox_aibd = st.checkbox("✈️ Proximité AIBD (Aéroport Dakar)")

        submitted = st.form_submit_button(
            "🔮 Estimer le prix du terrain", type="primary", use_container_width=True
        )

    if submitted:
        z_enc  = enc_z.transform([zone])[0]
        tf_enc = int(enc_tf_t.transform(
            pd.DataFrame([[titre_f]], columns=["Titre_foncier"]))[0][0])
        vi_enc = int(enc_vi.transform(
            pd.DataFrame([[viabilise]], columns=["Viabilisation"]))[0][0])
        in_enc = int(enc_in_t.transform(
            pd.DataFrame([[inondation]], columns=["Inondation"]))[0][0])
        zt_enc = enc_zt.transform([zone_type])[0]

        X_pred = pd.DataFrame([{
            "Zone_enc":z_enc, "TF_enc":tf_enc, "Vi_enc":vi_enc,
            "In_enc":in_enc, "ZT_enc":zt_enc,
            "Superficie_m2":superficie,
            "Vue_mer":int(vue_mer), "Prox_route":int(prox_route),
            "Prox_aibd":int(prox_aibd), "Distance_mer_km":dist_mer,
        }])

        prix_estime = m_t.predict(X_pred)[0]
        marge = mae_t * 0.85
        prix_m2 = prix_estime / superficie

        col_prix, col_expl = st.columns([1,1])

        with col_prix:
            st.markdown(f"""
            <div class="prix-card">
              <div style="font-size:0.85rem;color:#D4A853;">
                🌿 Terrain · {zone} · {superficie:.0f} m²
              </div>
              <div style="margin:0.5rem 0 0.2rem;font-size:0.9rem;color:#8AB0A0;">
                Prix total estimé par IA
              </div>
              <div class="montant">{prix_estime:,.0f} FCFA</div>
              <div class="fourchette">
                Fourchette : {prix_estime-marge:,.0f} — {prix_estime+marge:,.0f} FCFA
              </div>
              <div style="margin-top:0.8rem;padding-top:0.8rem;
                          border-top:1px solid #2A3D30;">
                <span style="color:#D4A853;font-weight:700;font-size:1.1rem;">
                  {prix_m2:,.0f} FCFA / m²
                </span>
                <span style="color:#5A7A64;font-size:0.85rem;"> · Prix au m²</span>
              </div>
              <div style="margin-top:0.5rem;font-size:0.8rem;color:#5A7A64;">
                {titre_f} · {viabilise}
              </div>
            </div>
            """.replace(",", " "), unsafe_allow_html=True)

            # Alertes
            if inondation == "Élevé":
                st.markdown('<div class="alerte-rouge">🌊 <strong>Risque d\'inondation ÉLEVÉ</strong> — Ce terrain peut être inutilisable en hivernage. Vérifiez obligatoirement avant achat.</div>', unsafe_allow_html=True)
            elif inondation == "Modéré":
                st.markdown('<div class="alerte-orange">⚠️ <strong>Risque modéré</strong> — Renseignez-vous auprès de la mairie sur l\'historique d\'inondation de cette parcelle.</div>', unsafe_allow_html=True)

            if titre_f == "Attestation":
                st.markdown('<div class="alerte-orange">📋 <strong>Attestation uniquement</strong> — Titre le moins sécurisé. Faites vérifier par un notaire ou un géomètre avant de payer.</div>', unsafe_allow_html=True)
            elif titre_f == "Délibération simple":
                st.markdown('<div class="alerte-orange">📋 <strong>Délibération simple</strong> — Titre intermédiaire. Vérifiez la signature du maire et du sous-préfet.</div>', unsafe_allow_html=True)
            elif titre_f == "TF individuel":
                st.markdown('<div class="alerte-verte">✅ <strong>Titre Foncier individuel</strong> — Le titre le plus sécurisé au Sénégal. Transaction recommandable.</div>', unsafe_allow_html=True)

            if not viabilise == "Eau + Électricité":
                st.markdown('<div class="alerte-bleue">💡 <strong>Terrain non/partiellement viabilisé</strong> — Prévoyez un budget de raccordement eau/électricité (500k-2M FCFA selon la zone).</div>', unsafe_allow_html=True)

        with col_expl:
            st.subheader("🔍 Pourquoi ce prix ?")
            importances = m_t.feature_importances_
            top_idx = np.argsort(importances)[::-1][:7]
            for i in top_idx:
                pct = int(importances[i]*100)
                barre = "█"*max(1,pct//3)+"░"*(33-max(1,pct//3))
                st.markdown(f"""
                <div class="facteur-card">
                  <div style="display:flex;justify-content:space-between;font-size:0.88rem;">
                    <span><strong>{LABS_T[i]}</strong></span>
                    <span style="color:#C85A1A;font-weight:700;">{pct}%</span>
                  </div>
                  <div style="font-family:monospace;font-size:0.75rem;
                              color:#C85A1A;margin-top:3px;">{barre}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-agence" style="margin-top:1rem;">
              💡 <strong>Conseil investisseur :</strong> À Diamniadio et Yenne/Daga,
              les prix des terrains augmentent de 12-18%/an depuis l'ouverture du TER
              et le développement du Pôle Urbain. Une délibération double signature
              proche de l'AIBD est un actif stratégique à moyen terme.
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R² du modèle", f"{r2_t:.2f}")
        m2.metric("MAE terrain", f"{mae_t:,.0f} FCFA".replace(",", " "))
        m3.metric("Variables terrain", f"{len(FEAT_T)}")
        m4.metric("Données", "800 parcelles")


# ══════════════════════════════════════════════════════════════════
# MODE BIEN CONSTRUIT
# ══════════════════════════════════════════════════════════════════
else:
    (m_c, enc_q, enc_t, enc_tx, enc_e, enc_tf, enc_in,
     quartiers, types_bien, FEAT_C, LABS_C, r2_c, mae_c) = MC

    st.markdown('<span class="mode-badge mode-construit">🏘️ Mode Bien Construit</span>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-agence">
      🎯 <strong>Adapté pour :</strong> DST Immobilier (Saly/Mbour/Nianing),
      Djidjack Immo (Fatick), Immothiès-Dakar · Villas, appartements,
      maisons déjà habitables · Vente et location
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_construit"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("📍 Localisation & bien")
            quartier   = st.selectbox("Quartier / Ville", quartiers)
            type_bien  = st.selectbox("Type de bien", types_bien)
            transaction= st.selectbox("Transaction", ["Vente","Location"])
            superficie = st.number_input("Superficie (m²)", 20, 800, 150)
            chambres   = st.slider("Chambres", 1, 10, 3)
            salles_bain= st.slider("Salles de bain", 1, 6, 2)
            etage      = st.slider("Étage", 0, 10, 0)

        with c2:
            st.subheader("🏗️ État & statut juridique")
            etat      = st.selectbox("État du bien", ["Neuf","Bon état","À rénover"])
            titre_f   = st.selectbox("Titre foncier", ["TF plein","Bail emphytéotique","Attestation"])
            inondation= st.selectbox("Risque d'inondation", ["Aucun","Faible","Modéré","Élevé"])
            age       = st.number_input("Âge du bien (années)", 0, 60, 5)
            dist_mer  = st.number_input("Distance à la mer (km)", 0.0, 50.0, 3.0)

        with c3:
            st.subheader("✨ Équipements & environnement")
            vue_mer    = st.checkbox("🌊 Vue mer (atout Saly/Mbour/Fatick)")
            parking    = st.checkbox("🚗 Parking")
            piscine    = st.checkbox("🏊 Piscine")
            jardin     = st.checkbox("🌿 Jardin / Cour")
            securite   = st.checkbox("🔒 Résidence sécurisée")
            clim       = st.checkbox("❄️ Climatisation")
            groupe     = st.checkbox("⚡ Groupe électrogène")
            prox_ecole = st.checkbox("🏫 Proximité école")
            prox_marche= st.checkbox("🛒 Proximité marché")
            meuble     = st.checkbox("🛋️ Meublé")

        submitted = st.form_submit_button(
            "🔮 Estimer le prix", type="primary", use_container_width=True
        )

    if submitted:
        q_enc  = enc_q.transform([quartier])[0]
        t_enc  = enc_t.transform([type_bien])[0]
        tx_enc = enc_tx.transform([transaction])[0]
        e_enc  = int(enc_e.transform(pd.DataFrame([[etat]],columns=["Etat"]))[0][0])
        tf_enc = int(enc_tf.transform(pd.DataFrame([[titre_f]],columns=["Titre_foncier"]))[0][0])
        in_enc = int(enc_in.transform(pd.DataFrame([[inondation]],columns=["Inondation"]))[0][0])

        X_pred = pd.DataFrame([{
            "Q_enc":q_enc,"T_enc":t_enc,"TX_enc":tx_enc,
            "E_enc":e_enc,"TF_enc":tf_enc,"IN_enc":in_enc,
            "Superficie_m2":superficie,"Chambres":chambres,
            "Salles_bain":salles_bain,"Age_annees":age,
            "Parking":int(parking),"Piscine":int(piscine),
            "Jardin":int(jardin),"Securite":int(securite),
            "Climatisation":int(clim),"Groupe_electrogene":int(groupe),
            "Etage":etage,"Distance_mer_km":dist_mer,
            "Vue_mer":int(vue_mer),
            "Prox_ecole":int(prox_ecole),"Prox_marche":int(prox_marche),
            "Meuble":int(meuble),
        }])

        prix_estime = m_c.predict(X_pred)[0]
        marge = mae_c * 0.85
        unite = "FCFA / mois" if transaction=="Location" else "FCFA"

        col_prix, col_expl = st.columns([1,1])

        with col_prix:
            badge = "📅 Location" if transaction=="Location" else "🏷️ Vente"
            st.markdown(f"""
            <div class="prix-card">
              <div style="font-size:0.85rem;color:#8AB0A0;">
                {badge} · {type_bien} · {quartier}
              </div>
              <div style="margin:0.5rem 0 0.2rem;font-size:0.9rem;color:#8AB0A0;">
                Prix estimé par IA
              </div>
              <div class="montant">{prix_estime:,.0f} {unite}</div>
              <div class="fourchette">
                Fourchette : {prix_estime-marge:,.0f} — {prix_estime+marge:,.0f} {unite}
              </div>
              <div style="margin-top:0.8rem;font-size:0.8rem;color:#5A7A64;">
                {superficie:.0f} m² · {chambres} ch · {etat} · {titre_f}
              </div>
            </div>
            """.replace(",", " "), unsafe_allow_html=True)

            if inondation == "Élevé":
                st.markdown('<div class="alerte-rouge">🌊 <strong>Risque ÉLEVÉ</strong> — Zone inondable. Vérifiez impérativement avant transaction.</div>', unsafe_allow_html=True)
            elif inondation == "Modéré":
                st.markdown('<div class="alerte-orange">⚠️ <strong>Risque modéré</strong> — Renseignez-vous auprès de la mairie.</div>', unsafe_allow_html=True)
            if titre_f == "Attestation":
                st.markdown('<div class="alerte-orange">📋 <strong>Attestation</strong> — Faire vérifier par un notaire avant achat.</div>', unsafe_allow_html=True)
            elif titre_f == "TF plein":
                st.markdown('<div class="alerte-verte">✅ <strong>TF plein</strong> — Titre le plus sécurisé.</div>', unsafe_allow_html=True)
            if vue_mer and quartier in ["Saly","Mbour","Fatick","Ngor","Almadies","Yoff"]:
                st.markdown('<div class="alerte-bleue">🌊 <strong>Vue mer</strong> — Atout majeur dans cette zone. La vue mer représente +15-25% sur le prix du marché côtier.</div>', unsafe_allow_html=True)

        with col_expl:
            st.subheader("🔍 Pourquoi ce prix ?")
            importances = m_c.feature_importances_
            top_idx = np.argsort(importances)[::-1][:8]
            for i in top_idx:
                pct = int(importances[i]*100)
                barre = "█"*max(1,pct//3)+"░"*(33-max(1,pct//3))
                st.markdown(f"""
                <div class="facteur-card">
                  <div style="display:flex;justify-content:space-between;font-size:0.88rem;">
                    <span><strong>{LABS_C[i]}</strong></span>
                    <span style="color:#C85A1A;font-weight:700;">{pct}%</span>
                  </div>
                  <div style="font-family:monospace;font-size:0.75rem;
                              color:#C85A1A;margin-top:3px;">{barre}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R² du modèle", f"{r2_c:.2f}")
        m2.metric("MAE", f"{mae_c:,.0f} FCFA".replace(",", " "))
        m3.metric("Variables", f"{len(FEAT_C)}")
        m4.metric("Données", "1 000 biens")

st.divider()
st.caption(
    "⚠️ Données synthétiques — prototype pédagogique, ne pas utiliser pour une transaction réelle. "
    "Amade Gueye Ndiaye · UAM Diamniadio · NekuImmo — nekuimmo.sn"
)
