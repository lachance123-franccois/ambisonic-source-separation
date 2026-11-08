# Extraction de locuteur par direction d'arrivÃ©e sur signaux ambisoniques


## Sommaire

1. [RÃ©sumÃ© des rÃ©sultats](#1-rÃ©sumÃ©-des-rÃ©sultats)
2. [Question de recherche](#2-question-de-recherche)
3. [Principe](#3-principe)
4. [Structure du dÃ©pÃ´t](#4-structure-du-dÃ©pÃ´t)
5. [Installation](#5-installation)
6. [DonnÃ©es](#6-donnÃ©es)
7. [Protocole expÃ©rimental](#7-protocole-expÃ©rimental)
8. [Mesures](#8-mesures)
9. [RÃ©sultats](#9-rÃ©sultats)
10. [Discussion](#10-discussion)
11. [Points d'attention](#11-points-dattention)
12. [Limites](#12-limites)
13. [Pistes d'extension](#13-pistes-dextension)
14. [DÃ©pannage (Windows / Spyder)](#14-dÃ©pannage-windows--spyder)
15. [RÃ©fÃ©rences](#15-rÃ©fÃ©rences)

---

## 1. RÃ©sumÃ© des rÃ©sultats

Les scÃ¨nes contiennent deux locuteurs Ã  0 dB. Les mesures portent sur 200 scÃ¨nes d'Ã©valuation, les mÃªmes pour toutes les mÃ©thodes. Il n'y a qu'un seul entraÃ®nement par configuration.

1. **La conclusion dÃ©pend de la statistique choisie.** Le beamformer cardioÃ¯de est **excellent sur une minoritÃ© de scÃ¨nes** : sur 28 scÃ¨nes sur 200, il dÃ©passe 20 dB, et jusqu'Ã  53 dB. Ce sont les scÃ¨nes oÃ¹ l'interfÃ©rent tombe dans son zÃ©ro arriÃ¨re. Sur la moitiÃ© des scÃ¨nes, en revanche, il reste **sous 3,5 dB**. L'autoencodeur est **rÃ©gulier** : son Ã©cart-type est d'environ 3 dB, contre 10,7 dB pour le cardioÃ¯de.
   - En **moyenne**, le cardioÃ¯de gagne : 8,38 dB contre 7,93 dB pour le modÃ¨le sans bruit, Ã  0Â° d'erreur.
   - En **mÃ©diane**, le modÃ¨le gagne nettement : 7,65 dB contre 3,46 dB.
   - Le modÃ¨le bat le cardioÃ¯de sur **64 Ã  68 % des scÃ¨nes**, quel que soit le bruit d'entraÃ®nement.
2. **Le bruit d'azimut Ã  l'entraÃ®nement fait apparaÃ®tre le compromis directivitÃ© / robustesse.** Plus le bruit est large, plus le modÃ¨le tolÃ¨re l'erreur de visÃ©e, et plus il perd Ã  erreur nulle.
   - La tolÃ©rance (erreur avant de perdre 3 dB en moyenne, pire des deux cÃ´tÃ©s) passe de **7,3Â°** sans bruit Ã  **17,4Â°** avec N(0, 10Â²), puis Ã  **plus de 30Â°** avec U(Â±30Â°).
   - En contrepartie, le SI-SDR Ã  0Â° baisse de **0,9 dB** avec N(0, 10Â²) et de **1,45 dB** avec U(Â±30Â°).
   - Avec U(Â±30Â°), la courbe est presque plate : de 6,3 Ã  6,5 dB entre â10Â° et +15Â°. Sa mÃ©diane bat celle des deux beamformers de â30Â° Ã  +20Â°.
3. **Ã variance Ã©gale, la forme du bruit (uniforme ou gaussien) ne change rien de mesurable** pour les erreurs positives. C'est l'**amplitude** qui compte. Attention : les modÃ¨les comparÃ©s n'ont pas tous le mÃªme nombre d'Ã©poques (Â§9.4).
4. **Avec une direction de cible fixe et connue**, le modÃ¨le `fixed` bat le cardioÃ¯de : moyenne de 9,80 dB contre 8,34 dB, mÃ©diane de 9,48 dB contre 3,42 dB, et **75 % des scÃ¨nes gagnÃ©es**.
5. **L'hypercardioÃ¯de n'est pas une base plus exigeante que le cardioÃ¯de.** Son lobe arriÃ¨re capte les interfÃ©rents placÃ©s derriÃ¨re. Sa moyenne est plus basse (5,96 dB contre 8,38 dB), mais sa mÃ©diane est plus haute (5,72 dB contre 3,46 dB).

---

## 2. Question de recherche

1. Un autoencodeur peut-il apprendre Ã  extraire une voix selon sa direction ? Fait-il mieux qu'un microphone virtuel (beamformer) ?
2. Qu'apporte la **parcimonie** du latent ?
3. Qu'apporte le **conditionnement** par la direction visÃ©e (FiLM) ?
4. Le modÃ¨le suit-il des **sources mobiles** ?
5. Quelle est sa **tolÃ©rance Ã  une erreur de visÃ©e** ? Peut-on l'amÃ©liorer en injectant du **bruit sur l'azimut pendant l'entraÃ®nement** ? Est-ce l'**amplitude** ou la **forme** (uniforme / gaussienne) du bruit qui compte ?

---

## 3. Principe

### 3.1 Modèle physique de la scène

On se place en **champ libre** (pas de salle, donc pas de réverbération) et en **champ lointain** : chaque locuteur est une source ponctuelle assez éloignée pour que son onde arrive sur le microphone comme une **onde plane**. Une onde plane est entièrement décrite par sa direction d'arrivée : l'azimut θ, mesuré dans le plan horizontal, et l'élévation φ.

**Convention d'angle** : θ = 0° correspond à l'avant, θ = +90° à la gauche, et θ croît dans le sens trigonométrique. Ici, toutes les sources sont dans le plan horizontal : φ = 0.

### 3.2 Ambisonie du premier ordre (FOA)

L'ambisonie décompose le champ sonore autour d'un point sur la base des **harmoniques sphériques**. Au premier ordre, on garde 4 composantes, qui correspondent à 4 microphones virtuels placés au même point :

| Canal | Équivalent physique | Directivité | Gain pour une onde venant de (θ, φ) |
|---|---|---|---|
| W | pression (micro omnidirectionnel) | identique dans toutes les directions | $1$ |
| X | gradient de pression avant / arrière (micro en « 8 ») | maximale devant, nulle sur les côtés | $\cos\theta\cos\varphi$ |
| Y | gradient gauche / droite | maximale à gauche, nulle devant | $\sin\theta\cos\varphi$ |
| Z | gradient haut / bas | maximale en haut | $\sin\varphi$ |

Une source $s(t)$ venant de la direction (θ, φ) donne donc le signal à 4 canaux

$$
\mathbf{x}(t) = \mathbf{y}(\theta, \varphi)\, s(t),
\qquad
\mathbf{y}(\theta, \varphi) = \begin{bmatrix} 1 & \cos\theta\cos\varphi & \sin\theta\cos\varphi & \sin\varphi \end{bmatrix}^{\top}.
$$

Le vecteur $\mathbf{y}$ est le **vecteur d'encodage**. La normalisation est **SN3D** : W a un gain de 1, et les canaux du premier ordre ont un gain maximal de 1. Les canaux sont rangés dans l'ordre W, X, Y, Z. Attention : le format AmbiX utilise aussi SN3D, mais range les canaux dans l'ordre W, Y, Z, X.

Comme φ = 0 ici, Z = 0 : ce canal n'apporte aucune information et n'est pas utilisé.

**Scène à deux locuteurs.** La propagation est linéaire, donc les champs s'additionnent. La cible $s_1$ arrive de $\theta_1(t)$ et l'interférent $s_2$ de $\theta_2(t)$ :

$$
\mathbf{x}(t) = \mathbf{y}(\theta_1(t))\, s_1(t) + \mathbf{y}(\theta_2(t))\, s_2(t).
$$

Pour une **source mobile**, $\theta(t)$ varie linéairement pendant le clip, et le gain d'encodage est recalculé à chaque échantillon. Avec un déplacement d'au plus 90° en 2 s, la source tourne d'au plus 1,4° pendant une fenêtre STFT de 32 ms : dans une trame, on peut la considérer comme immobile.

**Écart à la réalité.** Les scènes sont simulées directement au format ambisonique (format B). Un vrai micro ambisonique, avec 4 capsules en tétraèdre, a une directivité qui se dégrade aux hautes fréquences (repliement spatial au-delà de quelques kHz). Il capte aussi la réverbération de la salle. Aucun de ces deux effets n'est simulé ici.

### 3.3 Représentation temps-fréquence (STFT)

Chaque canal $c$ est découpé en trames de $N = 512$ échantillons (32 ms), avec un pas de $H = 128$ échantillons (8 ms, soit 75 % de recouvrement) et une fenêtre de Hann $w$ :

$$
X_c(k, n) = \sum_{t} x_c(t)\, w(t - nH)\, e^{-j 2\pi k t / N},
\qquad k = 0, \dots, 256.
$$

Cela donne $F = 257$ fréquences, espacées de 31,25 Hz, et 251 trames pour un clip de 2 s.

La STFT est linéaire. Dans chaque case $(k, n)$, en considérant les sources immobiles pendant la trame, on a donc :

$$
W = S_1 + S_2,
\qquad
X = \cos\theta_1\, S_1 + \cos\theta_2\, S_2,
\qquad
Y = \sin\theta_1\, S_1 + \sin\theta_2\, S_2.
$$

### 3.4 Descripteurs spatiaux (`features.py`)

**Idée.** En acoustique, la direction d'une onde se lit dans l'**intensité active** : le produit de la pression (canal W) et de la vitesse particulaire (canaux X, Y). On en calcule une version normalisée, dans chaque case temps-fréquence :

$$
i_x(k, n) = \frac{\mathrm{Re}\{W^{*} X\}}{|W|^2 + \epsilon},
\qquad
i_y(k, n) = \frac{\mathrm{Re}\{W^{*} Y\}}{|W|^2 + \epsilon},
\qquad
\ell(k, n) = \log\big(1 + |W|\big).
$$

On ajoute $\epsilon = 10^{-6}$ pour éviter de diviser par zéro.

**Une seule source dans la case.** Avec $W = S$ et $X = \cos\theta\, S$, on obtient $\mathrm{Re}\{W^* X\} = \cos\theta\, |S|^2$, donc :

$$
i_x = \cos\theta, \qquad i_y = \sin\theta.
$$

Le couple $(i_x, i_y)$ est un point du cercle unité qui indique **exactement** la direction de la source, quel que soit son spectre. On pourrait d'ailleurs estimer directement la direction par $\hat\theta = \mathrm{atan2}(i_y, i_x)$. C'est le principe des méthodes classiques de localisation par vecteur d'intensité, comme DirAC.

**Deux sources dans la case.** Avec $c_m = \cos\theta_m$ :

$$
i_x = \frac{c_1 |S_1|^2 + c_2 |S_2|^2 + (c_1 + c_2)\, \mathrm{Re}\{S_1^{*} S_2\}}{|S_1 + S_2|^2}.
$$

- Si **une source domine** la case ($|S_1| \gg |S_2|$), on retrouve la direction de cette source. La parole est **parcimonieuse** en temps-fréquence : à deux locuteurs, la plupart des cases sont dominées par l'un des deux (hypothèse de *W-disjoint orthogonality*). Les descripteurs indiquent donc, case par case, **qui parle**.
- Si les deux sources ont une énergie comparable, on obtient une moyenne pondérée des deux directions, perturbée par le terme croisé $\mathrm{Re}\{S_1^* S_2\}$. Quand $S_1$ et $S_2$ s'annulent partiellement, $|W|^2$ devient petit et $i_x$ peut sortir de $[-1, 1]$. C'est pourquoi les valeurs sont **bornées à $[-2, 2]$**.
- Dans les cases **silencieuses**, $\epsilon$ domine et $i_x, i_y \to 0$ : la case ne porte aucune information de direction.

**Le niveau** $\ell$ indique au modèle quelles cases contiennent de l'énergie. Il lui donne aussi une information spectrale, comme les formants et les harmoniques de la voix.

**Vecteur d'entrée.** Pour chaque trame $n$, on concatène les trois descripteurs sur les 257 fréquences :

$$
\mathbf{f}_n = \big[\, \ell(\cdot, n),\; i_x(\cdot, n),\; i_y(\cdot, n) \,\big] \in \mathbb{R}^{771}.
$$

**Ce que les descripteurs ne contiennent pas** : la phase absolue, le canal Z, et le contexte des trames voisines. Le modèle décide trame par trame.

### 3.5 Modèle (`models.py`)

**Masque.** On cherche un masque réel $M(k, n) \in [0, 1]$ qu'on applique au canal omnidirectionnel :

$$
\hat S_1(k, n) = M(k, n)\, W(k, n),
\qquad
\hat s_1 = \mathrm{iSTFT}\big(\hat S_1\big).
$$

La phase de $\hat S_1$ est celle du mélange $W$.

**Réseau.** Pour chaque trame :

$$
\mathbf{z} = \mathrm{topk}_8\Big(\mathrm{ReLU}\big(A_2\, \mathrm{ReLU}(A_1 \mathbf{f}_n + \mathbf{a}_1) + \mathbf{a}_2\big)\Big) \in \mathbb{R}^{64}_{+}
$$

$$
[\boldsymbol\gamma, \boldsymbol\beta] = g\big(\cos\theta_{\text{visé}}, \sin\theta_{\text{visé}}\big),
\qquad
\mathbf{h} = (1 + \boldsymbol\gamma) \odot \mathbf{z} + \boldsymbol\beta
\quad \text{(FiLM, mode conditionné)}
$$

$$
\mathbf{M}_n = \sigma\big(B_2\, \mathrm{ReLU}(B_1 \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2\big) \in [0, 1]^{257}
$$

- **Top-k** : on garde les 8 plus grandes valeurs du latent et on met les 56 autres à zéro. Le gradient ne passe que par les 8 dimensions gardées. Chaque trame est donc décrite par peu de « détecteurs » actifs.
- **FiLM** : le petit réseau $g$ (2 → 64 → 128) transforme la direction visée en un gain $\boldsymbol\gamma$ et un décalage $\boldsymbol\beta$ appliqués au latent. Le même encodeur sert pour toutes les directions, et c'est le FiLM qui « oriente » le décodeur. On lui donne $(\cos\theta, \sin\theta)$ plutôt que θ, pour que −180° et +180° donnent la même entrée.
- **Taille** : environ 297 000 paramètres sans conditionnement et 305 000 avec.

**Perte.** C'est une L1 entre la magnitude estimée et la magnitude de la vraie voix, en moyenne sur les cases :

$$
\mathcal{L} = \frac{1}{F N} \sum_{k, n} \Big|\, M(k, n)\, |W(k, n)| - |S_1(k, n)| \,\Big|
\;\; \big(+\, \lambda\, \overline{|\mathbf{z}|} \text{ avec } \texttt{--sparse l1}\big).
$$

Le masque idéal pour cette perte est $M^\star = \min\big(1, |S_1| / |W|\big)$. Quand les deux voix s'annulent partiellement ($|S_1| > |W|$), même le masque idéal ne peut pas reconstruire $|S_1|$. Et comme la phase reste celle du mélange, **la qualité atteignable est plafonnée**. Autre point : la perte porte sur les magnitudes, alors que la métrique finale (SI-SDR) porte sur le signal temporel. Les deux objectifs ne coïncident pas exactement.

### 3.6 Beamformer de référence (`beamformer.py`)

**Micro virtuel.** En combinant les canaux, on fabrique un micro de premier ordre orienté vers $\theta_s$ :

$$
b(t) = p\, W(t) + (1 - p)\big(X(t) \cos\theta_s + Y(t) \sin\theta_s\big).
$$

Pour une onde venant de θ, ce micro a le gain (sa **directivité**) :

$$
G(\Delta) = p + (1 - p) \cos\Delta,
\qquad
\Delta = \theta - \theta_s.
$$

| Directivité | $p$ | Gain sur le côté, $G(90°)$ | Zéros | Gain à l'arrière, $G(180°)$ | Indice de directivité (3D) |
|---|---|---|---|---|---|
| Omnidirectionnel (= W) | 1 | 1 | aucun | 1 | 0 dB |
| **Cardioïde** | 0,5 | 0,5 (−6 dB) | **180°** | **0** | 4,8 dB |
| Supercardioïde | 0,37 | 0,37 (−8,7 dB) | ±125° | −0,27 (−11,4 dB) | 5,7 dB |
| **Hypercardioïde** | 0,25 | 0,25 (−12 dB) | ±109,5° | −0,5 (−6 dB) | 6,0 dB (maximum au 1er ordre) |

**Ce que le beamformer obtient, en théorie.** Prenons une scène immobile, le beamformer visé exactement sur la cible, et deux voix décorrélées de même puissance. La sortie vaut $b = s_1 + G(\Delta)\, s_2$, où Δ est l'écart angulaire entre l'interférent et la cible. Son SI-SDR (§8.1) vaut alors :

$$
\mathrm{SI\text{-}SDR} \approx -20 \log_{10} |G(\Delta)|.
$$

| Interférent à Δ = | 30° | 60° | 90° | 120° | 150° | 180° |
|---|---|---|---|---|---|---|
| Cardioïde | 0,6 dB | 2,5 dB | 6,0 dB | 12,0 dB | 23,4 dB | **∞** |
| Hypercardioïde | 0,8 dB | 4,1 dB | 12,0 dB | 30,1 dB | 13,6 dB | 6,0 dB |

**Conséquences, visibles dans les résultats :**

- **Le cardioïde est « tout ou rien ».** Il est faible tant que l'interférent est à moins de 90° de la cible, et presque parfait quand l'interférent est proche de l'arrière. Avec la distribution d'interférents de ces scènes (§6), la théorie prédit une **médiane de 3,2 dB** et **14 % de scènes au-dessus de 20 dB**. On mesure **3,46 dB et 14 %** (28 scènes sur 200) : c'est ce qui sépare la moyenne de la médiane (§9.1).
- **L'hypercardioïde** rejette mieux les interférents sur les côtés, mais son lobe arrière plafonne le rejet à 6 dB pour un interférent derrière. Médiane prédite : 5,4 dB, mesurée : 5,72 dB.
- **Robustesse à l'erreur de visée.** Avec une erreur $e$, le gain sur la cible devient $G(e)$. Pour le cardioïde, $G(30°) = 0{,}93$, soit seulement −0,6 dB : son lobe est si large qu'une erreur de 30° ne change presque rien. C'est pourquoi le beamformer est presque insensible à l'erreur de visée (§9.2).
- Le beamformer est **fixe** : il ne sait pas où est l'interférent. Un beamformer **adaptatif** (MVDR) placerait son zéro sur l'interférent. Ce serait une base plus exigeante.

### 3.7 Les modes d'entraînement

| Mode | Tâche | Scènes |
|---|---|---|
| `fixed` | Extraire la voix située **devant (0°)**, sans conditionnement | sources immobiles, cible à 0° |
| `conditioned` | Extraire la voix à l'**azimut demandé** (FiLM) | sources mobiles, azimut quelconque |
| `conditioned` + bruit | Idem, avec une consigne volontairement bruitée à l'entraînement | sources mobiles |

### 3.8 Bruit sur l'azimut visé (`sample_az_noise` dans `train.py`)

À l'entraînement, la consigne donnée au FiLM est $\theta_{\text{visé}} = \theta_{\text{vrai}} + \varepsilon$ :

| Type | Tirage | Écart-type de ε | Part des tirages au-delà de 2 écarts-types |
|---|---|---|---|
| uniforme | $\varepsilon \sim \mathcal{U}(-a, +a)$ | $a / \sqrt{3}$ | 0 % (borné à ±a) |
| gaussien | $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ | $\sigma$ | 4,6 % |

Le modèle voit ainsi des consignes imparfaites. Il apprend à garder les cases dont la direction est **proche** de la consigne, et pas seulement **égale** à la consigne : son masque s'élargit en angle.

Le bruit est tiré indépendamment pour chaque échantillon, puis sous-échantillonné à une valeur par trame. Le modèle traite chaque trame séparément, donc ce choix équivaut, en espérance, à un décalage constant par clip. Ce ne serait plus vrai avec un modèle à contexte temporel. Le bruit n'est appliqué **qu'à l'entraînement** : la validation reçoit l'azimut exact.

---
## 4. Structure du dÃ©pÃ´t

```
ambisonics.py       encodage FOA (W, X, Y, Z)
scenes.py           scÃ¨ne Ã  deux locuteurs + Dataset gÃ©nÃ©rÃ© Ã  la volÃ©e
features.py         STFT / iSTFT, descripteurs spatiaux, azimut par trame
beamformer.py       micros virtuels cardioÃ¯de et hypercardioÃ¯de
models.py           autoencodeur Ã  masque (dense / sparse / FiLM)
metrics.py          SI-SDR
train.py            entraÃ®ne les expÃ©riences de la liste EXPERIENCES + courbes de perte
evaluate.py         SI-SDR de chaque scÃ¨ne pour chaque .pt de checkpoints/ (option miroir)
plot_results.py     rÃ©sumÃ© (moyenne, mÃ©diane, % gagnÃ©, IC), tolÃ©rances, asymÃ©trie, figures
analyze_latent.py   carte Â« dimensions du latent Ã azimut Â» pour chaque .pt
config.py           constantes (signal, paramÃ¨tres des scÃ¨nes, dossiers de donnÃ©es)
conftest.py         permet Ã  pytest de trouver les modules depuis tests/
tests/              tests unitaires (pytest)
checkpoints/        modÃ¨les sauvegardÃ©s (.pt)
results/            CSV et figures produits par les scripts
```

---

## 5. Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

### Dossiers de donnÃ©es

Les chemins sont dÃ©finis dans `config.py`, **relativement au dossier du projet** :

| Constante | Dossier par dÃ©faut | Variable d'environnement pour le changer |
|---|---|---|
| `TRAIN_DIR` | `LibriSpeech/` | `AMBI_TRAIN_DIR` |
| `VAL_DIR` | `TRAIN/LibriSpeech/` | `AMBI_VAL_DIR` |
| `TEST_DIR` | `TEST/LibriSpeech/` | `AMBI_TEST_DIR` |

Si `TEST/LibriSpeech/` n'existe pas, `evaluate.py` et `analyze_latent.py` se rabattent sur le dossier de validation et affichent un avertissement.

---

## 6. DonnÃ©es

Il faut un dossier de fichiers audio de parole (`.flac` ou `.wav`, mono). Les scÃ¨nes sont fabriquÃ©es Ã  la volÃ©e par `scenes.py` :

- deux locuteurs **diffÃ©rents** (si le dossier suit l'organisation `locuteur/chapitre/fichier` de LibriSpeech) ;
- des clips de 2 s, au mÃªme niveau RMS (SIR = 0 dB) ;
- un rÃ©Ã©chantillonnage automatique Ã  16 kHz si besoin ;
- un interfÃ©rent Ã  `az2 = az1 + U(MIN_SEP_DEG, 360 â MIN_SEP_DEG)`, avec **`MIN_SEP_DEG = â90`** dans `config.py`. C'est la valeur de tous les rÃ©sultats ci-dessous. Elle ne garantit **aucun Ã©cart minimal** entre les deux locuteurs. De plus, l'intervalle couvre 540Â° : les interfÃ©rents situÃ©s Ã  moins de 90Â° de la cible sont **deux fois plus probables** que les autres ;
- en mode mobile, chaque locuteur se dÃ©place linÃ©airement d'au plus **90Â°** pendant le clip (`MAX_MOVE_DEG = 90`).

Pour des scÃ¨nes avec un Ã©cart minimal de 30Â°, on met `MIN_SEP_DEG = 30`. Il faut alors **tout rÃ©-entraÃ®ner et rÃ©-Ã©valuer**.

Il faut des dossiers **sÃ©parÃ©s** pour l'entraÃ®nement, la validation et le test, avec des **locuteurs diffÃ©rents** :

| Usage | Corpus conseillÃ© (Mini LibriSpeech, OpenSLR SLR31) | Taille |
|---|---|---|
| EntraÃ®nement | `train-clean-5` | ~332 Mo |
| Validation | `dev-clean-2` | ~126 Mo |
| Test | `test-clean` (SLR12) ou une partie de `dev-clean-2` avec d'autres locuteurs | |

---

## 7. Protocole expÃ©rimental

Les valeurs par dÃ©faut sont lues dans `config.py`. On peut donc lancer les scripts sans argument, avec le bouton Play de Spyder.

### 7.1 EntraÃ®nement

```bash
python train.py                          # toutes les expÃ©riences dont le .pt manque
python train.py --only model_cond_gauss10 model_cond_unif30
python train.py --epochs 60              # nombre d'Ã©poques
python train.py --seed 1                 # autre graine -> checkpoints suffixÃ©s _s1
python train.py --retrain                # rÃ©-entraÃ®ne mÃªme si le .pt existe
```

`train.py` parcourt la liste `EXPERIENCES` et **ignore les checkpoints dÃ©jÃ  prÃ©sents**. RÃ©glages communs : latent top-k (k = 8 sur 64), taux d'apprentissage 1e-3, 2000 scÃ¨nes par Ã©poque, graine 0.

| Checkpoint | Mode | Bruit d'azimut | Ãcart-type | Ãpoques (campagne actuelle) | Meilleure val. (L1) |
|---|---|---|---|---|---|
| `model_fixed_none` | `fixed` | â | â | 100 | 0,155 |
| `model_cond_none` | `conditioned` | aucun | 0Â° | 100 | 0,205 |
| `model_cond_Uniform` | `conditioned` | U(Â±5Â°) | 2,9Â° | 100 | 0,206 |
| `model_cond_gauss2p9` | `conditioned` | N(0, 2,89Â²) | 2,9Â° | **60** | 0,212 (Ã©p. 56) |
| `model_cond_gaussian` | `conditioned` | N(0, 5Â²) | 5Â° | 100 | 0,209 |
| `model_cond_unif8p7` | `conditioned` | U(Â±8,66Â°) | 5Â° | **60** | 0,215 (Ã©p. 59) |
| `model_cond_gauss10` | `conditioned` | N(0, 10Â²) | 10Â° | **60** | 0,227 (Ã©p. 59) |
| `model_cond_unif30` | `conditioned` | U(Â±30Â°) | 17,3Â° | **60** | 0,246 (Ã©p. 57) |

Les paires `Uniform` / `gauss2p9` et `gaussian` / `unif8p7` ont le **mÃªme Ã©cart-type** : elles servent Ã  isoler l'effet de la forme du bruit.

### 7.2 Ãvaluation

```bash
python evaluate.py                 # scÃ¨nes mobiles, erreurs de -30Â° Ã  +30Â°
python evaluate.py --mirror        # + chaque scÃ¨ne en miroir gauche / droite (temps x2)
python evaluate.py --static        # modÃ¨les conditionnÃ©s sur scÃ¨nes immobiles
```

Tous les `.pt` du dossier `checkpoints/` sont Ã©valuÃ©s sur les **mÃªmes scÃ¨nes** (`--seed 999`). Les modÃ¨les `fixed` sont toujours testÃ©s sur des scÃ¨nes immobiles, cible Ã  0Â°. Les deux beamformers reÃ§oivent la mÃªme direction fautive que le modÃ¨le. `evaluate.py` enregistre le SI-SDR **de chaque scÃ¨ne**, puis appelle `plot_results.py`.

Pour refaire le rÃ©sumÃ© et les figures sans rÃ©-Ã©valuer :

```bash
python plot_results.py --per_scene results/eval_par_scene.csv
```

| Sortie | Contenu |
|---|---|
| `eval_par_scene.csv` | SI-SDR de chaque scÃ¨ne, pour chaque modÃ¨le et chaque erreur (+ colonne `mirror`) |
| `eval_comparatif.csv` | par (modÃ¨le, erreur) : moyenne, mÃ©diane, moyenne plafonnÃ©e Ã  30 dB, % de scÃ¨nes gagnÃ©es contre chaque beamformer, IC 95 % |
| `tolerance.csv` | tolÃ©rance `e*`, sur la moyenne et sur la mÃ©diane, pour les erreurs positives et nÃ©gatives |
| `asymetrie.csv` | SI-SDR(âe) â SI-SDR(+e), appariÃ© par scÃ¨ne, avec IC 95 % |
| `robustesse_amplitude.png` | SI-SDR selon l'erreur (moyenne et mÃ©diane), pour la sÃ©rie d'amplitudes de bruit |
| `forme_variance_egale.png` | Ã©cart appariÃ© gaussien â uniforme Ã  variance Ã©gale |
| `compromis.png` | SI-SDR Ã  0Â° en fonction de la tolÃ©rance |
| `victoires.png` | % de scÃ¨nes oÃ¹ le modÃ¨le bat le cardioÃ¯de |
| `distribution_0deg.png` | distribution des SI-SDR par scÃ¨ne Ã  0Â° |
| `fixe_vs_beamformer.png` | modÃ¨le `fixed` contre cardioÃ¯de, sur scÃ¨nes immobiles |

### 7.3 Analyse du latent

```bash
python analyze_latent.py --n_clips 20 --step 30
```

Pour chaque modÃ¨le, le script encode 20 clips **d'une seule source** Ã  chaque azimut, de â180Â° Ã  +150Â° par pas de 30Â°. Il moyenne ensuite le latent **avant FiLM**. Sortie : `results/latent_map_<nom>.png`.

---


## 8. Mesures

### 8.1 SI-SDR (*scale-invariant signal-to-distortion ratio*)

Soient $\hat{\mathbf{s}}$ le signal estimé et $\mathbf{s}$ la vraie voix, tous deux centrés (moyenne retirée). On projette l'estimation sur la référence :

$$
\alpha = \frac{\langle \hat{\mathbf{s}}, \mathbf{s} \rangle}{\|\mathbf{s}\|^2},
\qquad
\mathbf{e}_{\text{cible}} = \alpha\, \mathbf{s},
\qquad
\mathbf{e}_{\text{résidu}} = \hat{\mathbf{s}} - \alpha\, \mathbf{s},
$$

$$
\mathrm{SI\text{-}SDR} = 10 \log_{10} \frac{\|\mathbf{e}_{\text{cible}}\|^2}{\|\mathbf{e}_{\text{résidu}}\|^2} \quad \text{(dB)}.
$$

- **Invariance au gain** : multiplier $\hat{\mathbf{s}}$ par une constante ne change pas le score. On mesure donc la **forme** du signal, pas son volume. C'est indispensable ici, car le beamformer et le masque ne conservent pas le niveau de la cible.
- **Le résidu contient tout ce qui n'est pas la cible** : l'interférent qui reste, les distorsions de la voix dues au masque, les artefacts de phase. Le SI-SDR les pénalise de la même façon, sans les distinguer.
- **Repères** : 0 dB signifie autant d'énergie dans le résidu que dans la cible. Chaque +10 dB divise l'énergie du résidu par 10. Le mélange brut, W = $s_1 + s_2$ avec deux voix de même puissance, est à **0 dB** (mesuré : −0,01 dB).
- **Amélioration** : $\Delta\mathrm{SI\text{-}SDR} = \mathrm{SI\text{-}SDR}(\hat{\mathbf{s}}) - \mathrm{SI\text{-}SDR}(W)$, c'est-à-dire ce qu'apporte la méthode par rapport au mélange brut.
- **Ce qu'il ne mesure pas** : la qualité perçue et l'intelligibilité. Pour cela, il faudrait des mesures perceptives comme PESQ ou STOI (§13).

Dans le code (`metrics.py`), un petit $\epsilon = 10^{-8}$ évite la division par zéro. Une estimation parfaite donne donc une valeur très grande mais finie.

### 8.2 Statistiques sur les scènes

On calcule un SI-SDR $x_i$ par scène $i = 1, \dots, n$ ($n = 200$), pour chaque méthode et chaque erreur de visée.

| Statistique | Définition | Ce qu'elle dit |
|---|---|---|
| Moyenne | $\bar x = \frac{1}{n} \sum_i x_i$ | statistique habituelle, mais très sensible aux scènes extrêmes (cardioïde > 20 dB) |
| Médiane | valeur qui sépare les scènes en deux moitiés | ce qu'obtient une scène typique ; insensible aux scènes extrêmes |
| Moyenne plafonnée | $\frac{1}{n} \sum_i \min(x_i, 30\ \text{dB})$ | moyenne où une scène ne peut pas compter pour plus de 30 dB |
| % de scènes gagnées | $\frac{1}{n} \sum_i \mathbf{1}\big[x_i^{\text{modèle}} > x_i^{\text{beamformer}}\big]$ | sur combien de scènes le modèle fait mieux |

**Pourquoi plusieurs statistiques ?** Les distributions sont très différentes. Le cardioïde est « tout ou rien » (§3.6), avec un écart-type de 10,7 dB. Le modèle est régulier, avec un écart-type d'environ 3 dB. Une seule statistique peut alors donner une conclusion trompeuse : il faut regarder la moyenne **et** la médiane **et** le pourcentage de scènes gagnées.

### 8.3 Intervalles de confiance (bootstrap apparié)

Pour savoir si un écart est réel ou dû au hasard du choix des 200 scènes :

1. On tire 200 scènes **avec remise** parmi les 200, et on calcule la moyenne sur ce tirage.
2. On répète $B = 2000$ fois.
3. L'intervalle de confiance à 95 % va du 2,5ᵉ au 97,5ᵉ centile de ces 2000 moyennes.

Pour comparer deux méthodes A et B, on calcule l'écart **scène par scène**, $d_i = x_i^A - x_i^B$, puis on applique le bootstrap aux $d_i$. Les deux méthodes étant testées sur les mêmes scènes, la difficulté propre à chaque scène s'annule dans l'écart : l'intervalle est beaucoup plus étroit qu'en comparant deux moyennes séparées. **Si l'intervalle ne contient pas 0, l'écart est significatif.**

Ce bootstrap ne mesure que la variabilité due **aux scènes**. La variabilité due **à l'entraînement** (initialisation, ordre des données) demande de répéter avec plusieurs graines.

### 8.4 Tolérance à l'erreur de visée

On note $m(e)$ le SI-SDR (moyen ou médian) quand on vise avec une erreur $e$. La **tolérance** $e^\star$ est la plus grande erreur qui coûte au plus 3 dB :

$$
e^\star = \max\big\{\, e \ge 0 \;:\; m(e) \ge m(0) - 3\ \text{dB} \,\big\},
$$

obtenue par interpolation linéaire entre les erreurs testées. Elle est calculée séparément pour $e > 0$ et $e < 0$. « Pire des deux côtés » désigne la plus petite des deux. Si la courbe ne perd jamais 3 dB dans la plage testée, on note « > 30° ».

### 8.5 Asymétrie

$$
a(e) = \frac{1}{n} \sum_i \big[\, x_i(-e) - x_i(+e) \,\big].
$$

Avec $a(e) < 0$, la méthode fait moins bien quand on vise trop à droite que trop à gauche. Avec `--mirror`, chaque scène est aussi évaluée en miroir gauche / droite ($Y \to -Y$, $\theta \to -\theta$). Le jeu de scènes devient alors parfaitement symétrique : toute asymétrie restante vient du modèle lui-même.

### 8.6 Carte du latent

Pour chaque azimut θ, de −180° à +150° par pas de 30°, on encode 20 clips d'une seule source et on moyenne le latent $\mathbf{z}$, **avant FiLM**, sur les clips et les trames. On obtient une activation moyenne $\bar z_d(\theta)$ pour chaque dimension $d$. La **fraction active** est la part des couples (dimension, azimut) avec $\bar z_d(\theta) > 10^{-6}$. Sur la figure, les dimensions sont triées par azimut préféré. La forme en diagonale vient donc en partie de ce tri : ce qui est informatif, c'est le nombre de dimensions utilisées et la largeur en azimut de chacune.

### 8.7 Courbes d'apprentissage

On trace la perte L1 (§3.5) sur l'entraînement et sur la validation, à chaque époque. Le checkpoint gardé est celui de la **meilleure perte de validation**, calculée avec l'azimut exact.
---

## 9. RÃ©sultats

Campagne actuelle : 8 modÃ¨les, 200 scÃ¨nes d'Ã©valuation (graine 999), sources mobiles pour les modÃ¨les conditionnÃ©s. Fichiers : `results/eval_par_scene.csv`, `eval_comparatif.csv`, `tolerance.csv`, `asymetrie.csv`.

> **Jeu de test** : le cardioÃ¯de donne exactement les mÃªmes valeurs que lors de la campagne prÃ©cÃ©dente. L'Ã©valuation s'est donc trÃ¨s probablement faite sur le dossier de **validation**, faute de dossier `TEST/LibriSpeech/`. Les locuteurs testÃ©s sont alors ceux qui ont servi Ã  choisir les checkpoints. Ce biais est lÃ©ger, et identique pour tous les modÃ¨les, mais il reste Ã  corriger.

### 9.1 Moyenne contre mÃ©diane : le beamformer est Â« tout ou rien Â»

![Distribution des SI-SDR par scÃ¨ne](results/distribution_0deg.png)

Ã 0Â° d'erreur (sources mobiles) :

| MÃ©thode | Moyenne | MÃ©diane | Moyenne plafonnÃ©e Ã  30 dB | ScÃ¨nes gagnÃ©es par le modÃ¨le |
|---|---|---|---|---|
| CardioÃ¯de | **8,38** | 3,46 | 7,80 | â |
| HypercardioÃ¯de | 5,96 | 5,72 | 5,96 | â |
| AE sans bruit | 7,93 | 7,65 | 7,93 | 66 % contre le cardioÃ¯de, 74 % contre l'hypercardioÃ¯de |
| AE U(Â±5Â°) | 7,92 | **7,97** | 7,92 | 68 % / 74 % |
| AE N(0, 5Â²) | 7,82 | 7,77 | 7,82 | 68 % / 74 % |
| AE N(0, 10Â²) | 7,00 | 7,12 | 7,00 | 65 % / 71 % |
| AE U(Â±30Â°) | 6,47 | 6,74 | 6,47 | 64 % / 64 % |

- Le cardioÃ¯de a une distribution trÃ¨s Ã©talÃ©e : un Ã©cart-type de **10,7 dB**, 28 scÃ¨nes au-dessus de 20 dB et un maximum Ã  53 dB. Ce sont les scÃ¨nes oÃ¹ l'interfÃ©rent passe dans son **zÃ©ro arriÃ¨re**, ce qui donne un rejet presque parfait. Sur la moitiÃ© des scÃ¨nes, il reste **sous 3,5 dB**.
- Le modÃ¨le est **rÃ©gulier** : son Ã©cart-type est d'environ 3 dB. Le modÃ¨le sans bruit ne dÃ©passe pas 21 dB, et aucun modÃ¨le ne dÃ©passe 26 dB. Son masque est plafonnÃ©, donc il ne profite jamais de ces cas parfaits, mais il n'a presque pas de scÃ¨nes ratÃ©es.
- **Plafonner Ã  30 dB ne suffit pas Ã  inverser la moyenne** (7,80 dB pour le cardioÃ¯de contre 7,93 dB pour le modÃ¨le sans bruit) : beaucoup de scÃ¨nes du cardioÃ¯de se situent entre 20 et 30 dB.
- **La bonne lecture** : le modÃ¨le est **meilleur sur deux scÃ¨nes sur trois**, et le cardioÃ¯de l'emporte en moyenne grÃ¢ce Ã  une minoritÃ© de scÃ¨nes gÃ©omÃ©triquement favorables. Dans une vraie salle, la rÃ©verbÃ©ration empÃªcherait ces rejets parfaits Ã  30 dB ou plus. **Il faut donc rapporter la moyenne ET la mÃ©diane.**

### 9.2 Robustesse Ã  l'erreur de visÃ©e, selon l'amplitude du bruit

![Robustesse selon l'amplitude du bruit](results/robustesse_amplitude.png)

**SI-SDR moyen (dB)**

| Erreur | sans bruit | U(Â±5Â°) | N(0, 5Â²) | N(0, 10Â²) | U(Â±30Â°) | CardioÃ¯de | HypercardioÃ¯de |
|---|---|---|---|---|---|---|---|
| â30Â° | â3,37 | â2,69 | â2,18 | 0,30 | 3,56 | **7,31** | 5,26 |
| â10Â° | 3,48 | 5,77 | 6,20 | 6,17 | 6,26 | **8,11** | 5,92 |
| 0Â° | 7,93 | 7,92 | 7,82 | 7,00 | 6,47 | **8,38** | 5,96 |
| +10Â° | 4,19 | 5,25 | 6,04 | 6,88 | 6,46 | **8,54** | 6,17 |
| +20Â° | 0,72 | 1,03 | 1,75 | 5,45 | 5,99 | **8,05** | 6,13 |
| +30Â° | â1,88 | â1,79 | â1,35 | 2,52 | 4,27 | **7,72** | 6,08 |

**SI-SDR mÃ©dian (dB)**

| Erreur | sans bruit | U(Â±5Â°) | N(0, 5Â²) | N(0, 10Â²) | U(Â±30Â°) | CardioÃ¯de | HypercardioÃ¯de |
|---|---|---|---|---|---|---|---|
| â30Â° | â1,58 | â0,56 | 0,11 | 1,78 | **5,13** | 2,89 | 4,74 |
| â10Â° | 4,84 | 6,41 | **6,84** | 6,64 | 6,66 | 3,40 | 5,32 |
| 0Â° | 7,65 | **7,97** | 7,77 | 7,12 | 6,74 | 3,46 | 5,72 |
| +10Â° | 4,90 | 5,90 | 6,46 | **7,03** | 6,66 | 3,75 | 5,90 |
| +20Â° | 1,82 | 2,22 | 2,86 | 6,17 | **6,41** | 4,02 | 5,83 |
| +30Â° | â0,08 | â0,19 | 0,28 | 4,04 | 5,34 | 4,13 | **5,77** |

![ScÃ¨nes gagnÃ©es contre le cardioÃ¯de](results/victoires.png)

**Part des scÃ¨nes oÃ¹ le modÃ¨le bat le cardioÃ¯de (%)**

| Erreur | sans bruit | U(Â±5Â°) | N(0, 5Â²) | N(0, 10Â²) | U(Â±30Â°) |
|---|---|---|---|---|---|
| â30Â° | 8 | 11 | 10 | 18 | 32 |
| â10Â° | 39 | 53 | 57 | 61 | 64 |
| 0Â° | 66 | 68 | 68 | 65 | 64 |
| +10Â° | 42 | 50 | 57 | 66 | 66 |
| +20Â° | 16 | 16 | 20 | 52 | 57 |
| +30Â° | 8 | 8 | 8 | 22 | 34 |

**Ãcarts appariÃ©s par rapport au modÃ¨le sans bruit (dB, moyenne [IC 95 %])**

| Erreur | N(0, 10Â²) â sans bruit | U(Â±30Â°) â sans bruit | U(Â±30Â°) â N(0, 10Â²) |
|---|---|---|---|
| â30Â° | +3,67 [+3,03 ; +4,37] | +6,94 [+6,00 ; +7,97] | +3,27 [+2,62 ; +3,93] |
| 0Â° | **â0,93** [â1,15 ; â0,73] | **â1,45** [â1,76 ; â1,15] | â0,53 [â0,74 ; â0,33] |
| +10Â° | +2,69 [+2,18 ; +3,19] | +2,27 [+1,80 ; +2,77] | â0,42 [â0,58 ; â0,25] |
| +20Â° | +4,73 [+4,06 ; +5,43] | +5,27 [+4,51 ; +6,05] | +0,54 [+0,32 ; +0,78] |
| +30Â° | +4,40 [+3,70 ; +5,17] | +6,15 [+5,26 ; +7,08] | +1,75 [+1,39 ; +2,12] |

**Lecture**

1. **Sans bruit, le modÃ¨le est trÃ¨s directif.** Il perd 3,7 dB en moyenne Ã  +10Â°, et passe sous le SI-SDR du mÃ©lange brut au-delÃ  de 20Â°.
2. **Un bruit faible (Ã©cart-type â¤ 5Â°) aide peu**, et seulement dans la plage vue Ã  l'entraÃ®nement : de +1 Ã  +2 dB autour de 10Â°, rien au-delÃ  de 20Â°.
3. **Un bruit large change la forme de la courbe.**
   - Avec N(0, 10Â²), le modÃ¨le reste au-dessus de 6,8 dB de 0Â° Ã  +10Â°, et Ã  5,45 dB Ã  +20Â°.
   - Avec U(Â±30Â°), la courbe est presque plate : de 6,3 Ã  6,5 dB entre â10Â° et +15Â°, puis 6,0 dB Ã  +20Â°.
4. **Le coÃ»t est rÃ©el et significatif** : â0,9 dB Ã  0Â° pour N(0, 10Â²) et â1,45 dB pour U(Â±30Â°). C'est le compromis attendu (Â§10).
5. **Contre le cardioÃ¯de** :
   - en moyenne, aucun modÃ¨le ne le bat, quelle que soit l'erreur ;
   - en mÃ©diane et en part de scÃ¨nes gagnÃ©es, les modÃ¨les Ã  bruit large (N(0, 10Â²) et U(Â±30Â°)) le battent **de â10Â° Ã  +20Â°**. Leur mÃ©diane bat aussi l'hypercardioÃ¯de dans cette plage.

### 9.3 Le compromis directivitÃ© / robustesse

![Compromis](results/compromis.png)

| ModÃ¨le | Ãcart-type du bruit | Moyenne Ã  0Â° | MÃ©diane Ã  0Â° | `e*` moyenne (+ / â) | `e*` mÃ©diane (+ / â) |
|---|---|---|---|---|---|
| sans bruit | 0Â° | **7,93** | 7,65 | 8,6Â° / 7,3Â° | 10,7Â° / 10,6Â° |
| U(Â±5Â°) | 2,9Â° | 7,92 | **7,97** | 10,7Â° / 12,0Â° | 12,2Â° / 14,1Â° |
| N(0, 2,9Â²)* | 2,9Â° | 7,41 | 7,14 | 11,8Â° / 10,2Â° | 13,7Â° / 13,1Â° |
| N(0, 5Â²) | 5Â° | 7,82 | 7,77 | 12,7Â° / 13,3Â° | 14,4Â° / 16,2Â° |
| U(Â±8,7Â°)* | 5Â° | 7,34 | 7,28 | 13,3Â° / 14,4Â° | 15,4Â° / 16,6Â° |
| N(0, 10Â²)* | 10Â° | 7,00 | 7,12 | 24,9Â° / 17,4Â° | 29,6Â° / 20,4Â° |
| U(Â±30Â°)* | 17,3Â° | 6,47 | 6,74 | **> 30Â° / > 30Â°** | **> 30Â° / > 30Â°** |

\* 60 Ã©poques au lieu de 100 (Â§9.4).

Les points se rangent le long d'une **frontiÃ¨re** : quand la tolÃ©rance augmente, le SI-SDR Ã  0Â° diminue. Les modÃ¨les Ã  60 Ã©poques sont un peu sous cette frontiÃ¨re, ce qui est cohÃ©rent avec un entraÃ®nement plus court. **Le choix du bruit d'entraÃ®nement revient Ã  choisir un point sur cette frontiÃ¨re**, selon la prÃ©cision du systÃ¨me de suivi qui fournira l'azimut.

### 9.4 Forme du bruit : uniforme ou gaussien Ã  variance Ã©gale

![Forme du bruit Ã  variance Ã©gale](results/forme_variance_egale.png)

| Erreur | N(0, 2,9Â²) â U(Â±5Â°) | U(Â±8,7Â°) â N(0, 5Â²) |
|---|---|---|
| â30Â° | â0,93 [â1,29 ; â0,57] | +0,26 [â0,26 ; +0,88] |
| â10Â° | â1,27 [â1,56 ; â1,00] | â0,09 [â0,41 ; +0,23] |
| â5Â° | â0,93 [â1,14 ; â0,72] | â0,40 [â0,59 ; â0,19] |
| 0Â° | â0,51 [â0,69 ; â0,32] | â0,48 [â0,63 ; â0,34] |
| +10Â° | +0,02 [â0,34 ; +0,39] | â0,19 [â0,44 ; +0,06] |
| +20Â° | +0,06 [â0,31 ; +0,41] | â0,18 [â0,51 ; +0,15] |
| +30Â° | â0,02 [â0,41 ; +0,37] | +0,07 [â0,30 ; +0,43] |

**Lecture**

- **Pour les erreurs positives (+10Â° Ã  +30Â°), les Ã©carts sont nuls dans la marge d'incertitude**, pour les deux variances.
- **Ã 0Â°, les deux modÃ¨les rÃ©cents perdent environ 0,5 dB**, que leur bruit soit gaussien ou uniforme. Ils ont Ã©tÃ© entraÃ®nÃ©s sur **60 Ã©poques au lieu de 100**. Leur meilleure perte de validation (0,212 et 0,215) correspond Ã  celle des modÃ¨les Ã  100 Ã©poques vers l'Ã©poque 60. Ce dÃ©ficit vient donc probablement de l'entraÃ®nement plus court, et non de la forme du bruit.
- Le seul Ã©cart net est **N(0, 2,9Â²) sous U(Â±5Â°) pour les erreurs nÃ©gatives**, de â0,9 Ã  â1,3 dB. Il ne se retrouve pas dans l'autre paire et peut venir de l'asymÃ©trie propre Ã  chaque modÃ¨le (Â§9.6).
- **Conclusion : Ã  variance Ã©gale, aucun effet de la forme n'est dÃ©montrÃ©. L'amplitude du bruit, c'est-Ã -dire son Ã©cart-type, explique l'essentiel des diffÃ©rences.** Pour conclure proprement, il faut que tous les modÃ¨les aient le mÃªme nombre d'Ã©poques, puis rÃ©pÃ©ter avec plusieurs graines.

### 9.5 ModÃ¨le `fixed` contre beamformer (cible Ã  0Â°, sources immobiles)

![ModÃ¨le fixed contre beamformer](results/fixe_vs_beamformer.png)

| MÃ©thode | Moyenne | MÃ©diane | Moyenne plafonnÃ©e Ã  30 dB |
|---|---|---|---|
| CardioÃ¯de visÃ© Ã  0Â° | 8,34 | 3,42 | 7,23 |
| AE sparse top-k `fixed` | **9,80** | **9,48** | **9,80** |

Le modÃ¨le `fixed` gagne sur **toutes** les statistiques, et sur **75 % des scÃ¨nes**. L'Ã©cart moyen est de +1,46 dB [â0,35 ; +3,10]. Son intervalle de confiance touche 0, Ã  cause de la dispersion du cardioÃ¯de. Le modÃ¨le n'utilise pas de consigne d'azimut : sa courbe plate n'est **pas** une mesure de robustesse.

### 9.6 AsymÃ©trie entre erreurs positives et nÃ©gatives

`asymetrie.csv`, SI-SDR(âe) â SI-SDR(+e) en dB, sans scÃ¨nes miroir :

| ModÃ¨le | Â±5Â° | Â±10Â° | Â±30Â° |
|---|---|---|---|
| sans bruit | â0,73 [â1,15 ; â0,33] | â0,70 [â1,18 ; â0,21] | â1,50 [â2,51 ; â0,49] |
| U(Â±5Â°) | +0,39 [+0,08 ; +0,70] | +0,52 [â0,04 ; +1,10] | â0,90 [â1,96 ; +0,12] |
| N(0, 2,9Â²) | â0,30 [â0,57 ; â0,01] | â0,77 [â1,25 ; â0,28] | â1,81 [â2,83 ; â0,83] |
| N(0, 10Â²) | â0,28 [â0,48 ; â0,09] | â0,70 [â1,08 ; â0,34] | â2,23 [â3,33 ; â1,09] |
| U(Â±30Â°) | â0,10 [â0,25 ; +0,05] | â0,21 [â0,49 ; +0,08] | â0,71 [â1,61 ; +0,25] |
| CardioÃ¯de | â0,40 [â1,11 ; +0,19] | â0,43 [â1,32 ; +0,37] | â0,41 [â1,92 ; +1,01] |

La plupart des modÃ¨les font moins bien aux erreurs nÃ©gatives, mais pas tous : U(Â±5Â°) fait l'inverse Ã  Â±5Â°. Un bug de signe dans le code affecterait tous les modÃ¨les de la mÃªme faÃ§on. Ce n'est pas le cas : il s'agit donc d'un mÃ©lange d'effet du jeu de 200 scÃ¨nes (le cardioÃ¯de penche lui aussi du mÃªme cÃ´tÃ©, sans que ce soit significatif) et d'asymÃ©trie apprise par chaque modÃ¨le. **Ã vÃ©rifier avec `python evaluate.py --mirror`** : si l'asymÃ©trie disparaÃ®t avec les scÃ¨nes miroir, elle venait des scÃ¨nes.

### 9.7 Structure du latent

Cartes `results/latent_map_<nom>.png` : 20 clips, une seule source, un azimut tous les 30Â°, latent **avant FiLM**. Les dimensions sont triÃ©es par azimut prÃ©fÃ©rÃ©, donc **la diagonale est en partie produite par le tri**.

| ![](results/latent_map_model_fixed_none.png) | ![](results/latent_map_model_cond_none.png) |
|---|---|
| ![](results/latent_map_model_cond_gauss10.png) | ![](results/latent_map_model_cond_unif30.png) |

| ModÃ¨le | Fraction active | ParticularitÃ©s |
|---|---|---|
| `fixed_none` | 0,20 | deux blocs (gauche / droite), **presque rien autour de 0Â°** |
| `cond_none` | 0,25 | diagonale + raie isolÃ©e vers la dimension 50 (arriÃ¨re, Â±150Â° Ã  180Â°) |
| `cond_Uniform` | 0,26 | diagonale, raie vers 26 Ã  l'arriÃ¨re |
| `cond_gauss2p9` | 0,25 | diagonale + **forte raie vers 38** Ã  l'arriÃ¨re |
| `cond_gaussian` | 0,26 | diagonale compacte (dimensions 0 Ã  23) |
| `cond_unif8p7` | 0,24 | diagonale, raies faibles vers 52 et 60 |
| `cond_gauss10` | 0,28 | diagonale + raies vers 52 et 61 Ã  l'arriÃ¨re |
| `cond_unif30` | **0,31** | activitÃ© **Ã©talÃ©e sur 0 Ã  31**, maximum plus faible (â 50), pas de raie isolÃ©e |

**Observations**

- **Le modÃ¨le `fixed` code Â« ce qui n'est pas devant Â»**. Quand la source est vers 0Â°, le latent est presque nul et le dÃ©codeur produit un masque par dÃ©faut qui laisse passer le signal. Ce modÃ¨le n'a pas besoin de coder l'azimut finement, d'oÃ¹ sa fraction active plus faible.
- **Les modÃ¨les conditionnÃ©s codent l'azimut de faÃ§on continue et circulaire** : les mÃªmes dimensions sont actives Ã  â180Â° et Ã  +150Â°.
- **Correction d'une affirmation prÃ©cÃ©dente** : nous avions Ã©crit que le bruit d'azimut faisait disparaÃ®tre les raies isolÃ©es Ã  haut indice. **C'est faux.** `gauss2p9`, `gauss10` et `unif8p7` en ont aussi. Ces raies codent la rÃ©gion arriÃ¨re et dÃ©pendent de l'initialisation.
- **Le seul effet net du bruit sur le latent se voit avec U(Â±30Â°)** : plus de dimensions utilisÃ©es, une activitÃ© plus Ã©talÃ©e et moins intense. C'est cohÃ©rent avec un modÃ¨le qui doit produire des masques plus larges.
- Les fractions actives des modÃ¨les dÃ©jÃ  analysÃ©s ont baissÃ© de 0,02 Ã  0,03 depuis la campagne prÃ©cÃ©dente, alors que les modÃ¨les n'ont pas changÃ©. Les clips d'analyse viennent maintenant du dossier de validation, et non plus de celui d'entraÃ®nement. **Des Ã©carts de cet ordre ne sont donc pas interprÃ©tables.**

### 9.8 Courbes d'apprentissage

| ![](results/loss_model_cond_none.png) | ![](results/loss_model_cond_gauss10.png) |
|---|---|
| ![](results/loss_model_cond_unif30.png) | ![](results/loss_model_fixed_none.png) |

- Les modÃ¨les Ã  60 Ã©poques atteignent leur meilleure validation aux Ã©poques 56 Ã  59 : ils n'ont **pas fini de converger**.
- Avec un bruit large (`gauss10`, `unif30`), la perte d'entraÃ®nement reste **au-dessus** de la validation. C'est normal : l'entraÃ®nement se fait avec une consigne fausse, et la validation avec la consigne exacte. Leur perte de validation est plus haute (0,227 et 0,246) : c'est la traduction, sur la perte, du coÃ»t de directivitÃ© vu Ã  0Â°.
- Sans bruit, un lÃ©ger Ã©cart entre entraÃ®nement et validation apparaÃ®t aprÃ¨s environ 60 Ã©poques. Le bruit joue aussi un rÃ´le de **rÃ©gularisation**.

---

## 10. Discussion

**Le modÃ¨le et le beamformer n'Ã©chouent pas de la mÃªme faÃ§on.** Le cardioÃ¯de dÃ©pend de la gÃ©omÃ©trie : il est presque parfait quand l'interfÃ©rent est derriÃ¨re, et mÃ©diocre quand l'interfÃ©rent est Ã  moins de 90Â° de la cible, ce qui est frÃ©quent dans ces scÃ¨nes (Â§6). L'autoencodeur est plafonnÃ© par son masque, mais rÃ©gulier. Selon l'usage :

- si l'on veut le meilleur rÃ©sultat **en moyenne** sur des scÃ¨nes simulÃ©es sans rÃ©verbÃ©ration, le cardioÃ¯de reste devant ;
- si l'on veut **un bon rÃ©sultat sur la plupart des scÃ¨nes**, l'autoencodeur est prÃ©fÃ©rable : il gagne sur deux scÃ¨nes sur trois Ã  consigne exacte.

**Le compromis directivitÃ© / robustesse est confirmÃ©, et il est rÃ©glable.** L'Ã©cart-type du bruit d'entraÃ®nement fixe le point sur la frontiÃ¨re (Â§9.3) :

| PrÃ©cision du systÃ¨me de suivi | Bruit conseillÃ© | Ce que l'on obtient |
|---|---|---|
| quelques degrÃ©s | aucun, ou Ã©cart-type â¤ 5Â° | meilleur SI-SDR Ã  0Â° ; s'effondre au-delÃ  de 10Â° |
| environ 10Â° | N(0, 10Â²) | â0,9 dB Ã  0Â°, reste au-dessus de 6,8 dB jusqu'Ã  +10Â° |
| 20Â° Ã  30Â° ou inconnue | U(Â±30Â°) | â1,45 dB Ã  0Â°, courbe presque plate jusqu'Ã  Â±20Â° |

**La forme du bruit compte peu, son amplitude compte beaucoup.** Ã variance Ã©gale, aucune diffÃ©rence n'est dÃ©montrÃ©e. Pour choisir un bruit d'entraÃ®nement, il suffit donc de viser l'Ã©cart-type de l'erreur attendue du systÃ¨me de suivi, voire un peu plus.

**L'hypercardioÃ¯de n'est pas une base plus exigeante dans ce cadre.** Son lobe arriÃ¨re lui coÃ»te cher en moyenne, puisque la moitiÃ© des interfÃ©rents arrivent de l'arriÃ¨re. Une base vraiment plus forte serait un beamformer **adaptatif** (MVDR), qui place son zÃ©ro sur l'interfÃ©rent.

---

## 11. Points d'attention

| Point | ConsÃ©quence | Ãtat |
|---|---|---|
| `MIN_SEP_DEG = â90` (`config.py`) | aucun Ã©cart minimal entre locuteurs ; interfÃ©rent deux fois plus probable dans Â±90Â° de la cible | gardÃ© pour la comparabilitÃ© ; passer Ã  30 demande de tout rÃ©-entraÃ®ner |
| pas de dossier `TEST/LibriSpeech/` | Ã©valuation probablement sur les locuteurs de validation | **Ã  corriger** : ajouter un jeu de test |
| 4 modÃ¨les Ã  60 Ã©poques, 4 Ã  100 | comparaisons directes biaisÃ©es d'environ 0,5 dB Ã  0Â° | **Ã  corriger** : rÃ©-entraÃ®ner avec le mÃªme nombre d'Ã©poques |
| une seule graine par configuration | variabilitÃ© d'entraÃ®nement inconnue | `python train.py --seed 1`, puis `--seed 2` |
| moyenne seule dans les anciennes versions | conclusion biaisÃ©e en faveur du cardioÃ¯de | **corrigÃ©** : mÃ©diane, moyenne plafonnÃ©e, % gagnÃ© |
| asymÃ©trie Â± non expliquÃ©e | biais possible du jeu de scÃ¨nes | **outil ajoutÃ©** : `evaluate.py --mirror` |
| chemins Windows absolus, `--moving` impossible Ã  dÃ©sactiver, `pandas` manquant | â | **corrigÃ©s** |

---

## 12. Limites

- Les scÃ¨nes sont **simulÃ©es** : sources ponctuelles, deux locuteurs, Ã©lÃ©vation nulle, **sans salle**. L'absence de rÃ©verbÃ©ration favorise le beamformer, puisqu'elle permet des rejets parfaits dans son zÃ©ro.
- Le modÃ¨le travaille trame par trame, sans contexte temporel.
- Le masque rÃ©el appliquÃ© sur W avec la phase du mÃ©lange plafonne la qualitÃ© atteignable.
- Une graine par expÃ©rience, 200 scÃ¨nes d'Ã©valuation, et un nombre d'Ã©poques qui n'est pas le mÃªme pour tous les modÃ¨les.
- L'Ã©valuation porte probablement sur les locuteurs de validation (Â§9).

---

## 13. Pistes d'extension

**PrioritÃ© 1, pour consolider les conclusions :**
- rÃ©-entraÃ®ner tous les modÃ¨les conditionnÃ©s avec le **mÃªme nombre d'Ã©poques** ;
- ajouter un **jeu de test sÃ©parÃ©** ;
- utiliser **3 graines** par configuration ;
- Ã©valuer avec `--mirror`.

**PrioritÃ© 2, pour amÃ©liorer le modÃ¨le :**
- ajouter un contexte temporel (GRU ou convolutions) ;
- estimer un masque complexe ou un filtre multicanal, pour ne plus Ãªtre plafonnÃ© par le masque rÃ©el sur W ;
- utiliser une perte directement sur le SI-SDR ;
- appliquer un **curriculum** d'amplitude de bruit.

**PrioritÃ© 3, pour des scÃ¨nes plus rÃ©alistes et des bases plus fortes :**
- ajouter de la rÃ©verbÃ©ration (mÃ©thode des sources-images) ; on s'attend Ã  ce que l'avantage moyen du cardioÃ¯de disparaisse ;
- faire varier les niveaux entre locuteurs ;
- ajouter un bruit de fond ;
- comparer Ã  un beamformer adaptatif MVDR ;
- tester un Ã©cart minimal de 30Â° entre locuteurs.

---

## 14. DÃ©pannage (Windows / Spyder)

| SymptÃ´me | Cause | Solution |
|---|---|---|
| `AssertionError: Aucun fichier .pt trouvÃ©` | dossier `checkpoints/` vide | lancer `train.py` d'abord |
| `AssertionError: Aucun fichier audio dans ...` | dossier de donnÃ©es absent ou mal nommÃ© | vÃ©rifier `config.py`, ou dÃ©finir `AMBI_TRAIN_DIR` / `AMBI_VAL_DIR` |
| `[AVERTISSEMENT] Jeu de test introuvable` | pas de dossier `TEST/LibriSpeech/` | ajouter un jeu de test ou dÃ©finir `AMBI_TEST_DIR` |
| `train.py` ne fait rien | tous les checkpoints existent | `--retrain`, ou `--only <nom>` |
| `PermissionError: ... results/*.csv` | fichier ouvert dans Excel | fermer le fichier, relancer |
| `OMP: Error #15 ... libiomp5md.dll`, puis le kernel s'arrÃªte | deux copies d'OpenMP | `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` avant l'import de `torch` (dÃ©jÃ  en place) |
| entraÃ®nement lent | lecture et dÃ©codage des `.flac` Ã  chaque scÃ¨ne | mettre `load_audio` en cache (`functools.lru_cache`), prÃ©parer la validation une seule fois |

---

## 15. RÃ©fÃ©rences

- MarkoviÄ et al. â filtrage spatial implicite pour la sÃ©paration multicanale.
- Paek et al. â autoencodeurs sparse dans les espaces latents audio.
- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer* (2018).
- Panayotov et al., *LibriSpeech: an ASR corpus based on public domain audio books* (2015).
- Le Roux et al., *SDR â half-baked or well done?* (2019) â dÃ©finition du SI-SDR.