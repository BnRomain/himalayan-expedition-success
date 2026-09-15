# =============================================================
# 01 - Exploration des donnees
#
# Question metier : peut-on predire si une expedition himalayenne
# atteint le sommet (success) a partir d'infos connues AVANT le depart ?
#
# POINT DE DEPART : on part directement des ~20 features principales
# retenues par Muhammed Ali Yilmaz dans son notebook Kaggle (autorise :
# on a le droit d'utiliser la contribution des autres). On documente et
# on nettoie ces donnees deja reduites, plutot que les 65 colonnes brutes.
#
# Inspiration : "Himalayan Climb Prediction with ML/DL"
# https://www.kaggle.com/code/muhammedaliyilmazz/himalayan-climb-prediction-with-ml-dl
# =============================================================

import pandas as pd

# ----- Chargement du dataset brut (65 colonnes) -----
df_brut = pd.read_csv("DataBase/exped.csv", low_memory=False)
print("Dataset brut :", df_brut.shape[0], "lignes,", df_brut.shape[1], "colonnes")

# ----- Reduction aux 20 features de Muhammed + la cible -----
# Cle = nom reel dans le CSV, valeur = nom lisible (dictionnaire gemini-code.md).
# Le renommage est une etape de wrangling (Cours 2 : "Relabelling Columns").
correspondance = {
    "year": "year",
    "season": "season",
    "peakid": "peakid",
    "totmembers": "totmembers",
    "smtmembers": "smtmembers",
    "mdeaths": "mdeaths",
    "tothired": "tothired",
    "smthired": "smthired",
    "hdeaths": "hdeaths",
    "comrte": "comrte",
    "o2used": "o2used",
    "bcdate": "bcdate",
    "smtdate": "smtdate",
    "termdate": "termdate",
    "totdays": "totdays",
    "camps": "camps",
    "rope": "rope",
    "host": "host_country",
    "nation": "expedition_nation",
    "route1": "primary_route",
    "success1": "success",
}

df = df_brut[list(correspondance.keys())].rename(columns=correspondance)
print("Dataset reduit :", df.shape[0], "lignes,", df.shape[1], "colonnes")

# -------------------------------------------------------------
# ATTENTION - DATA LEAKAGE
# Parmi ces 20 features, certaines decrivent ce qui s'est passe PENDANT
# ou APRES l'expedition. Elles ne pourront PAS servir de variables
# predictives (on predirait le resultat avec le resultat). On les garde
# pour la description / le storytelling, mais elles sont exclues du modele :
#   smtmembers, mdeaths, smthired, hdeaths, smtdate, termdate, totdays
#
# Features utilisables comme predicteurs (connues avant le depart) :
#   year, season, peakid, totmembers, tothired, comrte, o2used,
#   bcdate, camps, rope, host_country, expedition_nation, primary_route
# -------------------------------------------------------------


# =============================================================
# 1. PREMIER REGARD
# =============================================================
print("\n===== 1. PREMIER REGARD =====")
print(df.head())
print("\nTypes des colonnes :")
print(df.dtypes)


# =============================================================
# 2. VALEURS MANQUANTES
# On regarde le % de NaN par colonne pour decider quoi nettoyer.
# Constat : les features qu'on va garder pour le modele (o2used, comrte,
# tothired, totmembers, camps, season, year, peakid) n'ont quasi aucun NaN.
# Les colonnes tres incompletes (totdays 26%, termdate 26%, bcdate 14%)
# font partie de celles qu'on ecarte de toute facon.
# =============================================================
print("\n===== 2. VALEURS MANQUANTES (%) =====")
print((df.isnull().mean() * 100).round(1).sort_values(ascending=False))


# =============================================================
# 3. ANALYSE DE LA CIBLE ET DES SIGNAUX
# C'est ici qu'on recolte les preuves pour justifier le choix des features.
# =============================================================
print("\n===== 3. CIBLE : success =====")
print(df["success"].value_counts(normalize=True).round(3))
# -> ~55% succes / 45% echec : cible EQUILIBREE, la regression logistique
#    fonctionnera sans reequilibrage, l'accuracy sera une metrique honnete.

print("\n-- Taux de succes par categorie --")
for c in ["season", "o2used", "comrte"]:
    print(f"\n[{c}]")
    print(df.groupby(c)["success"].agg(["mean", "count"]).round(3))

print("\n-- Moyenne des variables numeriques selon le succes --")
for c in ["year", "totmembers", "tothired", "camps", "rope"]:
    g = df.groupby("success")[c].mean().round(2)
    print(f"{c:12s} echec={g[False]:>8}  succes={g[True]:>8}")

print("\n-- Top 8 sommets les plus tentes : taux de succes --")
top8 = df["peakid"].value_counts().head(8).index
print(df[df["peakid"].isin(top8)].groupby("peakid")["success"]
      .agg(["mean", "count"]).round(3).sort_values("count", ascending=False))


# =============================================================
# 4. FEATURES RETENUES POUR LE MODELE (decision argumentee)
#
# Cible        : success (booleen, equilibre 55/45)
# Type modele  : regression logistique
#
# Features de base retenues (8, sous la limite de 10) :
#   o2used      -> signal le plus fort   : 80% succes avec O2 vs 45% sans
#   comrte      -> route commerciale      : 66% vs 44% (logistique encadree)
#   tothired    -> nb de sherpas          : moy 4.0 (succes) vs 2.1 (echec)
#   totmembers  -> taille de l'equipe     : 6.9 vs 5.2
#   camps       -> nb de camps d'altitude : 2.6 vs 1.8 (RESERVE : DATA LEAKAGE
#                  confirme. Sur l'Everest le succes saute de 22% (2 camps) a
#                  71% (3 camps) car l'assaut final part du camp 4 : "camps"
#                  mesure jusqu'ou on est monte, pas la preparation. A garder
#                  pour l'instant, on COMPARERA le modele avec/sans en etape 05.)
#   season      -> Printemps 57% / Automne 54% / Hiver 43% / Ete 38%
#   year        -> progres dans le temps  : 2005 (succes) vs 2001 (echec)
#   peakid      -> difficulte du sommet   : AmaDablam 71% vs Baruntse 38%
#                  (sera transforme en one-hot des 8 sommets + "Autre")
#
# Features ECARTEES (et pourquoi) :
#   primary_route     -> 1272 valeurs uniques : inexploitable en logistique
#   expedition_nation -> 100 valeurs, lien faible, redondant avec le reste
#   host_country      -> ecrasee par le Nepal (80% des lignes), apport faible
#   rope              -> aucun signal (166 vs 180), beaucoup de zeros
#   bcdate            -> date, 14% manquante, redondante avec season/year
#
# Handcrafted features (etape 04) - bilan apres tests :
#   ratio_hired = tothired/totmembers -> GARDEE (49% succes si ratio<0.5,
#                 64% si ratio 0.5-1 : la proportion de Sherpas compte).
#   o2used x season                   -> REJETEE : faux signal (confounding).
#                 L'effet "O2 selon saison" venait du sommet tente (Everest
#                 au printemps), pas de la saison. Cf preuve en 04.
#   rope_bool                         -> REJETEE : aucun signal (55% vs 54%).
# =============================================================
features_base = ["o2used", "comrte", "tothired", "totmembers",
                 "camps", "season", "year", "peakid"]

df_modele = df[features_base + ["success"]].copy()
print("\n===== 4. JEU DE DONNEES POUR LE MODELE =====")
print("Features de base :", features_base)
print("Shape :", df_modele.shape)