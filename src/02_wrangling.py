# =============================================================
# 02 - Data wrangling : du brut au dataset propre
#
# Ce script applique les decisions prises en exploration (01) et feature
# engineering (04). Il produit DEUX fichiers intermediaires :
#   DataBase/exped_depart.csv -> point de depart : les 21 colonnes de
#                                Muhammed, renommees, AVANT nettoyage.
#   DataBase/exped_clean.csv   -> dataset entierement nettoye et encode,
#                                pret pour la regression (etape 05).
#
# NOTE : on ne STANDARDISE pas ici. La standardisation (moyenne/ecart-type)
# doit etre calee sur le train uniquement, sinon le test "fuite" dans les
# parametres -> data leakage. Elle se fera donc en 05, apres le split.
# =============================================================

import pandas as pd

# =============================================================
# 1. CHARGEMENT + REDUCTION AUX 21 FEATURES (= point de depart)
# Cle = nom reel dans le CSV, valeur = nom lisible (cf gemini-code.md).
# =============================================================
df_brut = pd.read_csv("DataBase/exped.csv", low_memory=False)

correspondance = {
    "year": "year", "season": "season", "peakid": "peakid",
    "totmembers": "totmembers", "smtmembers": "smtmembers", "mdeaths": "mdeaths",
    "tothired": "tothired", "smthired": "smthired", "hdeaths": "hdeaths",
    "comrte": "comrte", "o2used": "o2used", "bcdate": "bcdate",
    "smtdate": "smtdate", "termdate": "termdate", "totdays": "totdays",
    "camps": "camps", "rope": "rope", "host": "host_country",
    "nation": "expedition_nation", "route1": "primary_route", "success1": "success",
}
df = df_brut[list(correspondance.keys())].rename(columns=correspondance)
df.to_csv("DataBase/exped_depart.csv", index=False)
print("exped_depart.csv :", df.shape[0], "lignes,", df.shape[1], "colonnes")


# =============================================================
# 2. NETTOYAGE DES LIGNES
#   - saison "Unknown" (2 lignes) : categorie inexploitable
#   - totmembers == 0 (63 lignes) : expedition sans membre = donnee douteuse,
#     et division par zero impossible pour ratio_hired
# =============================================================
avant = len(df)
df = df[df["season"] != "Unknown"]
df = df[df["totmembers"] > 0].copy()
print(f"Lignes retirees (nettoyage) : {avant - len(df)}  ->  reste {len(df)}")


# =============================================================
# 3. SELECTION DES FEATURES RETENUES (8 de base + cible)
# Tout le reste (data leakage + features ecartees, cf 01) est abandonne.
# =============================================================
features_base = ["o2used", "comrte", "tothired", "totmembers",
                 "camps", "season", "year", "peakid"]
df = df[features_base + ["success"]]


# =============================================================
# 4. ENCODAGE
# =============================================================
# 4a. Booleens -> 0/1 (le modele attend du numerique)
for col in ["o2used", "comrte", "success"]:
    df[col] = df[col].astype(int)

# 4b. Handcrafted feature retenue en 04 : ratio_hired = Sherpas / membres
#     (les autres, saison_o2 et rope_bool, ont ete rejetees)
df["ratio_hired"] = df["tothired"] / df["totmembers"]

# 4c. peakid : 416 valeurs uniques -> on ne garde que les 8 sommets les plus
#     tentes, le reste devient "Autre", puis one-hot (cf decision 01).
top8 = df["peakid"].value_counts().head(8).index
df["peakid"] = df["peakid"].where(df["peakid"].isin(top8), "Autre")

# 4d. One-hot de season et peakid. drop_first=True retire une categorie de
#     reference (evite la colinearite parfaite). References (1ere par ordre
#     alphabetique) : season = Autumn, peakid = AMAD (AmaDablam). Les
#     coefficients du modele s'interpreteront PAR RAPPORT a ces references
#     (ex : un coef positif sur peakid_EVER = "plus dur que l'AmaDablam").
df = pd.get_dummies(df, columns=["season", "peakid"], drop_first=True, dtype=int)


# =============================================================
# 5. SAUVEGARDE DU DATASET PROPRE
# =============================================================
df.to_csv("DataBase/exped_clean.csv", index=False)
print("exped_clean.csv  :", df.shape[0], "lignes,", df.shape[1], "colonnes")
print("\nColonnes finales :")
print(list(df.columns))
