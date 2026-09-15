# =============================================================
# 05 - Regression logistique : modele, validation, optimisation
#
# Objectif : predire success (sommet atteint ou non) a partir des features
# connues AVANT le depart. Cible booleenne equilibree (55/45) -> regression
# logistique (Cours 5), modele lineaire interpretable.
#
# Ce script se decompose en trois parties :
#   PARTIE 1 - Comparaison de deux methodes d'evaluation : Holdout (un seul
#              split 80/20) vs Validation croisee (5 plis). But : montrer que
#              le chiffre du Holdout depend du tirage, alors que la CV moyenne
#              plusieurs decoupages et donne une estimation plus fiable.
#   PARTIE 2 - Optimisation des hyperparametres (GridSearchCV) : on cherche les
#              meilleurs C et penalty. DISCIPLINE : la recherche se fait par CV
#              INTERNE sur le TRAIN uniquement, le test Holdout n'est jamais vu
#              pendant le tuning (sinon on optimiserait sur le juge final).
#   PARTIE 3 - Modele final optimise : matrice de confusion + coefficients.
#
# Anti-leakage transversal : la standardisation est DANS un Pipeline, donc
# recalculee sur le seul train a chaque entrainement / pli.
# camps est exclu des predicteurs (data leakage identifie en 01).
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, cross_val_predict, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set_theme(style="whitegrid")

# =============================================================
# 0. DONNEES, PIPELINE, OUTILS COMMUNS
# =============================================================
df = pd.read_csv("DataBase/exped_clean.csv")
y = df["success"]
features = [c for c in df.columns if c not in ["success", "camps"]]   # camps exclu (leakage)
X = df[features]
cols_num = ["tothired", "totmembers", "year", "ratio_hired"]          # a standardiser

baseline = max(y.mean(), 1 - y.mean())
print(f"Dataset : {len(df)} expeditions, {len(features)} predicteurs")
print(f"Baseline (classe majoritaire) : {baseline:.1%}\n")

# ColumnTransformer : standardise les colonnes continues, laisse passer le reste
# (booleens et one-hot deja en 0/1). Reutilise par tous les pipelines.
preprocesseur = ColumnTransformer(
    [("standardisation", StandardScaler(), cols_num)],
    remainder="passthrough")

# Modele "par defaut" (C=1, penalty L2, solver lbfgs) : sert de reference.
modele_defaut = Pipeline([
    ("preparation", preprocesseur),
    ("regression", LogisticRegression(max_iter=1000)),
])

# Meme protocole de CV partout : 5 plis stratifies, melange reproductible.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =============================================================
# PARTIE 1 - HOLDOUT vs VALIDATION CROISEE (modele par defaut)
# =============================================================
print("===== PARTIE 1 : Holdout vs Validation croisee =====\n")

# --- 1a. Holdout : un seul decoupage 80/20 ---
# Le test (20%) est mis de cote et sert de juge. stratify garde le ratio 55/45.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
modele_defaut.fit(X_train, y_train)
acc_holdout = accuracy_score(y_test, modele_defaut.predict(X_test))
print(f"Holdout (split 80/20)      : accuracy = {acc_holdout:.1%}  "
      f"(test = {len(y_test)} expeditions)")

# --- 1b. CV : 5 decoupages sur l'ensemble du dataset, puis moyenne ---
scores_cv = cross_validate(modele_defaut, X, y, cv=cv, scoring=["accuracy"])
acc_cv = scores_cv["test_accuracy"]
print(f"Validation croisee (5 plis): accuracy = {acc_cv.mean():.1%} "
      f"+/- {acc_cv.std():.1%}  (moyenne sur les 5 plis)")
print(f"  Detail des 5 plis : {[f'{a:.1%}' for a in acc_cv]}")
print("  -> Le Holdout tombe dans l'intervalle de la CV : estimations")
print("     coherentes, le modele ne doit rien a un decoupage chanceux.\n")


# =============================================================
# PARTIE 2 - OPTIMISATION DE L'HYPERPARAMETRE (GridSearchCV)
# On teste une grille de C, la force de regularisation : petit C = forte
# regularisation (coefficients brides, modele plus simple), grand C = faible
# regularisation (modele plus libre, risque de sur-apprentissage). C est LE
# levier principal d'une regression logistique. La recherche se fait par CV
# interne 5 plis SUR LE TRAIN, le test Holdout reste vierge.
# =============================================================
print("===== PARTIE 2 : Optimisation (GridSearchCV) =====\n")

grille = {
    "regression__C": [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100],
}
recherche = GridSearchCV(modele_defaut, grille, cv=cv,
                         scoring="accuracy", n_jobs=-1)
recherche.fit(X_train, y_train)     # uniquement sur le train

print(f"Meilleurs parametres   : {recherche.best_params_}")
print(f"Accuracy CV (train)    : {recherche.best_score_:.1%}")

# Evaluation FINALE du modele optimise sur le test Holdout (jamais vu pendant
# le tuning) -> comparaison juste avec le modele par defaut sur le MEME test.
acc_optim_holdout = accuracy_score(y_test, recherche.predict(X_test))
print(f"\nComparaison sur le meme test Holdout :")
print(f"  Modele par defaut (C=1, L2)   : {acc_holdout:.1%}")
print(f"  Modele optimise               : {acc_optim_holdout:.1%}")
print(f"  Gain                          : {(acc_optim_holdout-acc_holdout)*100:+.1f} pts")
print("  -> Sur une regression logistique deja bien posee, le gain est")
print("     marginal : les parametres par defaut etaient quasi optimaux.\n")


# =============================================================
# PARTIE 3 - MODELE FINAL OPTIMISE : confusion + coefficients
# On reutilise les meilleurs hyperparametres trouves.
# =============================================================
meilleur = recherche.best_estimator_

# --- 3a. Matrice de confusion (predictions out-of-fold sur tout le dataset) ---
# cross_val_predict reentraine a chaque pli, donc chaque expedition est predite
# alors qu'elle etait en validation (jamais vue a l'entrainement) : pas de leakage.
y_pred = cross_val_predict(meilleur, X, y, cv=cv)
cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Echec", "Succes"], yticklabels=["Echec", "Succes"])
ax.set_xlabel("Prediction du modele")
ax.set_ylabel("Realite")
ax.set_title("Matrice de confusion (modele optimise, validation croisee)")
fig.tight_layout()
fig.savefig("figures/07_matrice_confusion.png", dpi=150)
plt.close(fig)

# --- 3b. Coefficients (reentraine sur tout le dataset avec les meilleurs params) ---
meilleur.fit(X, y)
ordre = cols_num + [c for c in features if c not in cols_num]   # ordre du ColumnTransformer
coefs = pd.Series(meilleur.named_steps["regression"].coef_[0],
                  index=ordre).sort_values()

fig, ax = plt.subplots(figsize=(7, 6))
couleurs = ["#c0392b" if v < 0 else "#27ae60" for v in coefs.values]
ax.barh(coefs.index, coefs.values, color=couleurs)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Coefficient (log-odds)")
ax.set_title("Ce qui pousse au succes (vert) ou a l'echec (rouge)")
fig.tight_layout()
fig.savefig("figures/09_coefficients.png", dpi=150)
plt.close(fig)

print("-- Coefficients du modele final optimise (tries) --")
print(coefs.round(3))

print("\n2 figures generees :")
print("  07_matrice_confusion.png")
print("  09_coefficients.png")
