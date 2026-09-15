# Script oral — Soutenance (5 minutes)

Cible : ~5 min à deux. Format data storytelling. Ne pas lire mot à mot, c'est un fil.
Répartition : **Romain** ouvre et ferme (titre, 1, 2, 5), **Loriziano** porte le cœur technique (3, 4).

Minutage indicatif :
- Titre + Slide 1 (Romain) ............ ~1 min
- Slide 2 (Romain) .................... ~1 min  -> *passage à Loriziano*
- Slide 3 (Loriziano) ................ ~1 min
- Slide 4 (Loriziano) ................ ~1 min 15 -> *retour à Romain*
- Slide 5 (Romain) ................... ~45 s

---

## Slide titre — ROMAIN  (~20 s)

Bonjour à tous. Chaque année, des centaines d'expéditions tentent de gravir les
sommets de l'Himalaya, et toutes ne réussissent pas. Avec Loriziano, on s'est
posé une question simple : **peut-on prédire si une expédition atteindra le
sommet, en n'utilisant que ce qu'on sait avant même qu'elle parte ?**

---

## Slide 1 — Business goal — ROMAIN  (~45 s)

L'enjeu est concret : une agence d'expédition, un assureur ou un chef de cordée
voudraient estimer leurs chances en amont, quand il est encore temps d'ajuster
l'équipe, le matériel ou la stratégie.

Comme la réponse est « succès » ou « échec », c'est un problème de
**classification binaire**, qu'on traite par **régression logistique**.

Notre jeu de données, c'est *Himalayan Expeditions* sur Kaggle, tiré de la base
historique d'Elizabeth Hawley : plus de **11 000 expéditions**. Et un point
important : la cible est presque équilibrée, **55 % de succès contre 45 %
d'échec**. Ça tombe bien, parce que ça veut dire que l'accuracy sera une
métrique honnête, et que le minimum à battre, notre « baseline », c'est 55 %.

---

## Slide 2 — Data description — ROMAIN  (~1 min)

Côté données : on part de 65 colonnes brutes, qu'on réduit à la vingtaine de
variables retenues sur Kaggle, pour finir avec **8 prédicteurs** plus la cible.
Le nettoyage est minime, on perd 0,6 % des lignes : le jeu est déjà de bonne
qualité.

On a trois familles de variables : des **booléens** comme l'usage d'oxygène ou
la route commerciale, des **numériques** comme le nombre de Sherpas ou de
membres, et des **catégorielles** comme la saison et le sommet, qu'on encode en
one-hot.

Et surtout, le piège central de tout le projet : le **data leakage**. Beaucoup
de variables décrivent ce qui s'est passé *pendant* ou *après* l'expédition,
les décès, la date de sommet, la durée. Les utiliser, ce serait prédire le
résultat avec le résultat. On les a donc toutes écartées.

Sur la heatmap, on voit déjà ce qui corrèle avec le succès, l'oxygène en tête,
et une redondance entre nombre de Sherpas et taille d'équipe, qui va nous
inspirer une variable.

*(Je passe la parole à Loriziano.)*

---

## Slide 3 — Handcrafted features — LORIZIANO  (~1 min)

Merci Romain. Cette partie résume notre démarche : pour chaque variable qu'on
invente, on pose une hypothèse, on la teste, et on décide. Sur trois variables
bricolées, **une seule a survécu**, et c'est justement ce qui rend la démarche
crédible.

La gardée, c'est **ratio_hired**, la proportion de Sherpas par membre. À taille
d'équipe égale, le succès grimpe nettement : 49 % quand le ratio est faible,
64 % quand il monte.

La première rejetée : l'**interaction oxygène × saison**. On croyait tenir un
signal, l'oxygène semblait plus utile au printemps. Mais en creusant, on a
trouvé une **variable confondante** : l'oxygène est concentré sur l'Everest, qui
se tente au printemps. Ce n'était pas la saison, c'était le sommet. Corrélation
n'est pas causalité.

La deuxième rejetée : **rope_bool**, la corde fixe. Aucun écart, 55 contre 54 %,
et en plus elle est posée pendant l'ascension, donc du leakage. On l'écarte.

Le message : on n'a rien gardé « au feeling », et accepter de rejeter ses
propres idées, c'est le cœur de la démarche scientifique.

---

## Slide 4 — Régression — LORIZIANO  (~1 min 15)

Pourquoi la régression logistique ? Parce que la cible est binaire, ce qui
exclut la linéaire, et qu'avec seulement deux classes la softmax n'apporte rien.
En prime, ses coefficients sont **interprétables**, ce qui est exactement notre
but.

Tout est dans un pipeline **anti-leakage** : la standardisation est recalculée
sur le seul jeu d'entraînement à chaque pli. On a validé de deux façons,
validation croisée et Holdout, et les deux concordent.

Le résultat : **69,3 % en validation croisée**, contre 55 % pour la baseline.
Avec une précision de 75 %.

Sur les coefficients, deux variables écrasent tout : l'**oxygène** à +2,03 et la
**route commerciale** à +1,39. Les sommets s'interprètent par rapport à
l'AmaDablam : l'Everest et le Manaslu ressortent comme les plus durs.

Et un twist honnête : notre fameux **ratio_hired s'effondre** à +0,02. Seul il
était utile, mais une fois l'oxygène et la route commerciale présents, ils
absorbent son signal. C'est un classique.

Deux précisions : on a **exclu camps** parce que c'est du leakage, sur l'Everest
le succès passe de 22 à 71 % selon le nombre de camps, ce qui mesure jusqu'où on
est monté. Et l'optimisation des hyperparamètres n'apporte rien : sur un modèle
déjà bien posé, optimiser ne crée pas un signal qui n'existe pas.

*(Je rends la parole à Romain.)*

---

## Slide 5 — Conclusion — ROMAIN  (~45 s)

Pour conclure, notre analyse dégage une **hiérarchie claire** : d'abord
l'oxygène, ensuite la route commerciale, donc l'encadrement, et enfin la
difficulté du sommet. Le modèle atteint 69,3 %, loin devant 55 %, et c'est
cohérent avec le bon sens alpin, ce qui nous rassure.

Mais au fond, le vrai apport du projet, ce sont les **deux pièges** qu'on a su
éviter : le data leakage avec la variable camps, et la confusion entre
corrélation et causalité avec la fausse interaction. Dans les deux cas, c'est le
raisonnement, pas la statistique brute, qui a tranché.

Peut-on faire mieux ? Oui : en ajoutant l'**altitude du sommet** comme variable
continue, ou en testant un modèle non linéaire, tout en restant vigilant sur le
leakage.

Merci de votre attention, on est prêts pour vos questions.

---

## Réponses préparées (Q&A)

- **Pourquoi pas un modèle plus puissant (random forest, XGBoost) ?**
  Notre objectif est d'**expliquer** les facteurs, pas seulement de prédire. La
  logistique donne des coefficients lisibles. On l'a noté comme piste, mais avec
  prudence sur le leakage.
- **69 %, c'est moyen, non ?**
  C'est +14 points sur la baseline, sur de l'humain (météo, forme physique,
  décisions du jour J ne sont pas dans les données). On préfère un 69 % honnête
  qu'un 90 % gonflé par du leakage.
- **Pourquoi exclure camps si ça aide tant ?**
  Justement parce que ça « aide » trop : c'est une conséquence de la montée, pas
  une info connue avant le départ. La garder, ce serait tricher.
- **Comment êtes-vous sûrs que l'interaction O2 × saison est fausse ?**
  Effectifs minuscules (9 expéditions « été + O2 »), argument physique (la saison
  ne change pas la pression d'oxygène), et la variable confondante trouvée : la
  catégorie « printemps + O2 » est composée à 68 % d'Everest.
