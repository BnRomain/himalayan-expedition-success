# =============================================================
# 03 - Visualisation : chercher des LIENS entre features
#
# Objectif (consigne) : on ne veut pas des variables isolees, on veut
# montrer des RELATIONS. Chaque figure = un message pour le rapport.
#
# METHODE pour "trouver" une relation (a expliquer dans le rapport) :
#   1. on croise une feature avec la cible (groupby + mean) -> y a-t-il un ecart ?
#   2. si oui, on croise DEUX features entre elles vs la cible -> l'effet de
#      l'une depend-il de l'autre ? (= interaction)
#   3. la heatmap de correlation donne la vue d'ensemble pour reperer
#      les liens forts ET les redondances (features qui disent la meme chose).
#
# Toutes les figures sont exportees dans figures/ via plt.savefig pour le rapport.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")  # grille legere, lisible en rapport

# ----- Chargement + reduction (memes etapes que 01) -----
df = pd.read_csv("DataBase/exped.csv", low_memory=False)
df = df.rename(columns={"success1": "success"})

# On enleve les 2 lignes "Unknown" en saison (donnee inexploitable, voir 01)
df = df[df["season"].isin(["Spring", "Summer", "Autumn", "Winter"])].copy()

# Ordre des saisons par taux de succes (plus lisible qu'alphabetique)
ordre_saisons = ["Spring", "Autumn", "Winter", "Summer"]


# =============================================================
# FIGURE 1 - L'effet de l'oxygene (le signal le plus fort)
# COMMENT on l'a trouve : groupby(o2used).success.mean() -> 80% vs 45%.
# C'est le plus gros ecart de tout le dataset, on commence par la.
# =============================================================
fig, ax = plt.subplots(figsize=(6, 4))
taux_o2 = df.groupby("o2used")["success"].mean()
ax.bar(["Sans oxygene", "Avec oxygene"], taux_o2.values,
       color=["#c0392b", "#27ae60"])
ax.set_ylim(0, 1)                       # axe Y a zero : regle de base, pas de biais visuel
ax.set_ylabel("Taux de succes")
ax.set_title("L'oxygene supplementaire double les chances de succes")
for i, v in enumerate(taux_o2.values):
    ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
fig.tight_layout()
fig.savefig("figures/01_effet_oxygene.png", dpi=150)
plt.close(fig)


# =============================================================
# FIGURE 2 - INTERACTION oxygene x saison (notre handcrafted feature)
# COMMENT on l'a trouve : on a vu que o2 ET season comptaient chacun.
# Question naturelle : l'effet de l'O2 est-il le meme a chaque saison ?
# On croise les DEUX -> groupby([season, o2used]).success.mean().
# Reponse : NON. +46 pts au printemps, ~0 en hiver. C'est une interaction,
# donc une nouvelle feature o2used x season a du sens (cf Cours 4, Pclass x Age).
# =============================================================
pivot = (df.groupby(["season", "o2used"])["success"]
         .mean().unstack().reindex(ordre_saisons))

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(ordre_saisons))
largeur = 0.38
ax.bar(x - largeur/2, pivot[False], largeur, label="Sans O2", color="#c0392b")
ax.bar(x + largeur/2, pivot[True],  largeur, label="Avec O2", color="#27ae60")
ax.set_xticks(x)
ax.set_xticklabels(["Printemps", "Automne", "Hiver", "Ete"])
ax.set_ylim(0, 1)
ax.set_ylabel("Taux de succes")
ax.set_title("L'effet de l'oxygene DEPEND de la saison (interaction)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/02_interaction_o2_saison.png", dpi=150)
plt.close(fig)


# =============================================================
# FIGURE 3 - Heatmap : taux de succes selon (taille equipe x nb Sherpas)
# COMMENT on l'a trouve : tothired et totmembers comptent tous les deux,
# mais sont correles (0.58 dans la heatmap). Sont-ils redondants ?
# On croise les DEUX en tranches et on colore par le taux de succes.
# LECTURE : on lit une COLONNE (taille d'equipe fixee) de bas en haut ;
# si la couleur s'eclaircit vers le vert quand on monte en Sherpas, c'est
# qu'a equipe egale les Sherpas font la difference -> le RATIO compte
# (handcrafted feature ratio_hired). Plus parlant qu'un nuage de 11k points.
# =============================================================
df["bin_membres"] = pd.cut(df["totmembers"], [0, 3, 6, 10, 1e9],
                           labels=["1-3", "4-6", "7-10", "11+"])
df["bin_sherpas"] = pd.cut(df["tothired"], [-1, 0, 2, 5, 1e9],
                           labels=["0", "1-2", "3-5", "6+"])

# Taux de succes ET effectif de chaque case
taux = df.pivot_table("success", "bin_sherpas", "bin_membres",
                      aggfunc="mean", observed=True)
effectif = df.pivot_table("success", "bin_sherpas", "bin_membres",
                          aggfunc="size", observed=True)
taux = taux.iloc[::-1]          # Sherpas: beaucoup en haut (lecture intuitive)
effectif = effectif.iloc[::-1]
taux_masque = taux.mask(effectif < 20)   # on cache les cases peu fiables (n<20)

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(taux_masque, annot=True, fmt=".0%", cmap="RdYlGn", center=0.5,
            vmin=0.2, vmax=0.8, linewidths=1, ax=ax,
            cbar_kws={"label": "Taux de succes"})
ax.set_xlabel("Nombre de membres")
ax.set_ylabel("Nombre de Sherpas engages")
ax.set_title("A equipe egale, le succes monte avec le nombre de Sherpas")
fig.tight_layout()
fig.savefig("figures/03_sherpas_vs_membres.png", dpi=150)
plt.close(fig)


# =============================================================
# FIGURE 4 - Heatmap de correlation (vue d'ensemble des liens)
# COMMENT on l'a trouve : c'est l'inverse, on part de la vue d'ensemble.
# On encode en numerique les booleens, et on regarde toutes les correlations
# d'un coup. Utile pour : (a) voir ce qui correle avec success, (b) reperer
# les REDONDANCES (deux features tres correlees disent la meme chose).
# =============================================================
df_num = df.copy()
for col in ["o2used", "comrte", "success"]:
    df_num[col] = df_num[col].astype(int)        # bool -> 0/1
cols_num = ["success", "o2used", "comrte", "tothired", "totmembers", "camps", "year"]
corr = df_num[cols_num].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("Correlations entre features et avec la cible (success)")
fig.tight_layout()
fig.savefig("figures/04_correlations.png", dpi=150)
plt.close(fig)


# =============================================================
# FIGURE 5 - Multidimensionnel EN COURBES : oxygene x sommet x succes
# On croise les TROIS dimensions demandees dans un seul graphe. En abscisse,
# les grands sommets classes par difficulte croissante (= taux de succes sans
# O2, du plus accessible au plus dur). Deux courbes : avec O2 / sans O2.
# LECTURE : la courbe "avec O2" reste haute et plate -> a l'oxygene, le sommet
# importe peu. La courbe "sans O2" s'effondre -> sans oxygene, la difficulte du
# sommet devient decisive. L'ECART entre les courbes (la valeur de l'oxygene)
# se creuse avec la difficulte. On ne garde que les sommets ou CHAQUE groupe
# (avec / sans O2) compte au moins 30 expeditions, pour des taux fiables.
# =============================================================
noms = {"EVER": "Everest", "KANG": "Kangchenjunga", "LHOT": "Lhotse",
        "MAKA": "Makalu", "CHOY": "Cho Oyu", "DHA1": "Dhaulagiri",
        "MANA": "Manaslu", "ANN1": "Annapurna"}

lignes = []
for pk in noms:
    d = df[df["peakid"] == pk]
    sans = d[~d["o2used"]]
    avec = d[d["o2used"]]
    if len(sans) >= 30 and len(avec) >= 30:
        lignes.append((pk, sans["success"].mean(), avec["success"].mean()))

lignes.sort(key=lambda t: t[1], reverse=True)   # du plus facile au plus dur (sans O2)
labels = [noms[pk] for pk, _, _ in lignes]
y_sans = [s for _, s, _ in lignes]
y_avec = [a for _, _, a in lignes]
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(9.5, 5))
ax.fill_between(x, y_sans, y_avec, color="#27ae60", alpha=0.10)  # ecart = valeur de l'O2
ax.plot(x, y_avec, "-o", color="#27ae60", lw=2.6, ms=7, label="Avec oxygene")
ax.plot(x, y_sans, "-o", color="#c0392b", lw=2.6, ms=7, label="Sans oxygene")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Taux de succes")
ax.set_xlabel("Sommets classes du plus accessible au plus difficile (sans oxygene)")
ax.set_title("Sans oxygene, le sommet fait tout ; avec oxygene, l'ecart s'efface")
ax.legend()
fig.tight_layout()
fig.savefig("figures/08_oxygene_par_sommet.png", dpi=150)
plt.close(fig)


print("5 figures generees dans figures/ :")
print("  01_effet_oxygene.png")
print("  02_interaction_o2_saison.png")
print("  03_sherpas_vs_membres.png")
print("  04_correlations.png")
print("  08_oxygene_par_sommet.png")
