# =============================================================
# 04 - Feature engineering : on BRICOLE de nouvelles features
#
# Principe (consigne + Cours 4) : a partir des colonnes brutes, on fabrique
# des features qui captent mieux la realite. REGLE qu'on s'impose : pour
# chaque feature creee, on visualise son lien avec success AVANT de decider
# de la garder. Une feature qui ne marche pas mais qu'on a testee proprement
# est une bonne histoire pour le rapport (demarche scientifique honnete).
#
# On fabrique 3 features :
#   A. ratio_hired = tothired / totmembers     -> GARDEE (signal clair)
#   B. saison_o2 (interaction o2 x saison)     -> TESTEE puis REJETEE (confounding)
#   C. rope_bool = corde fixe utilisee ou non  -> TESTEE puis REJETEE
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("DataBase/exped.csv", low_memory=False)
df = df.rename(columns={"success1": "success"})
df = df[df["season"].isin(["Spring", "Summer", "Autumn", "Winter"])].copy()


# =============================================================
# A. ratio_hired = nombre de Sherpas / nombre de membres
# IDEE : ce n'est pas le nombre brut de Sherpas qui compte, mais leur
# PROPORTION dans l'equipe. 4 Sherpas pour 4 membres (ratio 1) est un
# encadrement bien plus fort que 4 Sherpas pour 20 membres (ratio 0.2).
# =============================================================
df = df[df["totmembers"] > 0].copy()            # evite la division par zero
df["ratio_hired"] = df["tothired"] / df["totmembers"]

# On regarde le taux de succes par tranche de ratio (binning pour lisibilite)
df["tranche_ratio"] = pd.cut(df["ratio_hired"], [-0.01, 0.5, 1, 2, 100],
                             labels=["< 0.5", "0.5 - 1", "1 - 2", "> 2"])
taux = df.groupby("tranche_ratio", observed=True)["success"].mean()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(taux.index.astype(str), taux.values, color="#2980b9")
ax.set_ylim(0, 1)
ax.set_ylabel("Taux de succes")
ax.set_xlabel("Ratio Sherpas / membres")
ax.set_title("Plus la proportion de Sherpas est forte, plus le succes monte")
for i, v in enumerate(taux.values):
    ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
fig.tight_layout()
fig.savefig("figures/05_ratio_hired.png", dpi=150)
plt.close(fig)
# VERDICT : 49% (ratio<0.5) -> 64% (ratio 0.5-1). Signal net et monotone. GARDEE.


# =============================================================
# B. Interaction oxygene x saison -> HYPOTHESE REJETEE (confounding)
# IDEE de depart : la figure 02 montrait un effet de l'O2 tres different
# selon la saison (+46 pts au printemps, ~0 en hiver). On a cru a une
# interaction et voulu creer saison_o2 (Pclass x Age du Cours 4).
#
# MAIS deux objections cassent ce raisonnement :
#   1. echantillon : 2426 expes "Printemps+O2" mais seulement 40 "Hiver+O2"
#      et 9 "Ete+O2" -> on ne peut rien conclure pour hiver/ete.
#   2. physique : l'O2 aide partout en altitude, la saison ne change pas
#      la pression d'oxygene. Un effet qui depend de la saison est suspect.
#
# VRAIE explication = variable CONFONDANTE. On la prouve ci-dessous :
#   l'O2 est concentre sur l'Everest (76% des expes Everest), et l'Everest
#   se tente au printemps (88%). Donc "Printemps+O2" = surtout "Everest".
#   La chaine reelle est : saison -> quel sommet -> usage O2 ET succes.
#   Ce n'est pas la saison qui module l'O2, c'est le sommet tente.
# =============================================================
pct_o2_everest = df.loc[df["peakid"] == "EVER", "o2used"].mean()
part_ever_printemps = (df.loc[df["peakid"] == "EVER", "season"] == "Spring").mean()
masque_sp_o2 = (df["season"] == "Spring") & (df["o2used"])
part_ever_dans_sp_o2 = (df.loc[masque_sp_o2, "peakid"] == "EVER").mean()
print("\n-- Preuve du confounding (saison_o2) --")
print(f"  Part des expes Everest qui utilisent l'O2 : {pct_o2_everest:.0%}")
print(f"  Part des expes Everest faites au printemps : {part_ever_printemps:.0%}")
print(f"  Part d'Everest dans la categorie 'Printemps+O2' : {part_ever_dans_sp_o2:.0%}")
# VERDICT : REJETEE. peakid (deja dans le modele) capture deja la difficulte
# du sommet, qui est le vrai signal. saison_o2 ne serait qu'un faux signal.


# =============================================================
# C. rope_bool : la corde fixe a-t-elle ete utilisee ? (VOTRE idee)
# IDEE : transformer rope (longueur en metres, 84% de zeros) en simple
# booleen "corde fixe oui/non". Hypothese : une expe qui pose de la corde
# fixe serait mieux preparee, donc plus de succes.
# =============================================================
df["rope_bool"] = df["rope"] > 0
taux_rope = df.groupby("rope_bool")["success"].mean()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Sans corde fixe", "Avec corde fixe"], taux_rope.values,
       color=["#95a5a6", "#7f8c8d"])
ax.set_ylim(0, 1)
ax.set_ylabel("Taux de succes")
ax.set_title("Corde fixe : aucun effet sur le succes (hypothese rejetee)")
for i, v in enumerate(taux_rope.values):
    ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
fig.tight_layout()
fig.savefig("figures/06_rope_bool.png", dpi=150)
plt.close(fig)
# VERDICT : 55% (sans) vs 54% (avec) -> ecart nul, voire inverse. De plus
# 84% de zeros = sans doute "non renseigne" plutot que "zero metre", et la
# corde est posee PENDANT l'ascension (risque de data leakage).
# -> REJETEE. A raconter dans le rapport comme hypothese testee et refutee.


print("\nFeatures bricolees :")
print("  A. ratio_hired  -> GARDEE  (signal clair, fig 05)")
print("  B. saison_o2    -> REJETEE (confounding par le sommet, fig 02)")
print("  C. rope_bool    -> REJETEE (aucun signal, fig 06)")
