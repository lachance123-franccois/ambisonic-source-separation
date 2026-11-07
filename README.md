# Extraction de locuteur par direction d'arrivée sur signaux ambisoniques

Projet pédagogique (PyTorch) : à partir d'une scène sonore enregistrée en **ambisonie du premier ordre** (FOA, 4 canaux) contenant deux locuteurs, extraire la voix qui vient d'une **direction donnée**, avec un **autoencodeur à masque dont l'espace latent est parcimonieux et conditionné par la direction visée**.

Le projet compare cette approche à un **beamformer classique** (microphone virtuel cardioïde) et étudie en particulier sa **robustesse à l'erreur de direction**, y compris lorsque le modèle est entraîné avec un **bruit sur l'azimut visé** (uniforme ou gaussien).

---

## Sommaire

1. [Question de recherche](#1-question-de-recherche)
2. [Principe](#2-principe)
3. [Structure du dépôt](#3-structure-du-dépôt)
4. [Installation](#4-installation)
5. [Données](#5-données)
6. [Protocole expérimental](#6-protocole-expérimental)
7. [Mesures](#7-mesures)
8. [Résultats](#8-résultats)
9. [Discussion : le compromis directivité / robustesse](#9-discussion--le-compromis-directivité--robustesse)
10. [Limites](#10-limites)
11. [Pistes d'extension](#11-pistes-dextension)
12. [Dépannage (Windows / Spyder)](#12-dépannage-windows--spyder)
13. [Références](#13-références)

---

## 1. Question de recherche

1. Un autoencodeur peut-il apprendre à extraire une voix selon sa direction, et fait-il mieux qu'un simple microphone virtuel (beamformer) ?
2. Qu'apporte la **parcimonie** du latent ?
3. Qu'apporte le **conditionnement** par la direction visée (FiLM) ?
4. Le modèle suit-il des **sources mobiles** ?
5. Quelle est sa **tolérance à une erreur de visée**, et peut-on l'améliorer en injectant du **bruit sur l'azimut pendant l'entraînement** ? Le type de bruit (uniforme ou gaussien) et son amplitude changent-ils la courbe de robustesse ?

---

## 2. Principe

### Ambisonie du 1er ordre (convention SN3D)

Une source `s(t)` à l'azimut θ et à l'élévation φ est encodée en 4 canaux :

| Canal | Formule |
|---|---|
| W | `s` |
| X | `s · cos θ · cos φ` |
| Y | `s · sin θ · cos φ` |
| Z | `s · sin φ` |

θ = 0° : devant ; θ = 90° : à gauche. Une scène à deux locuteurs est la **somme** des deux encodages.

### Descripteurs spatiaux

Pour une seule source dominante dans une case temps-fréquence, `Re(conj(W)·X)/|W|² ≈ cos θ` et `Re(conj(W)·Y)/|W|² ≈ sin θ`. Le modèle reçoit donc, pour chaque trame STFT, le niveau `log|W|` et ces deux rapports, soit 3 × 257 = 771 valeurs.

### Modèle (`models.py`)

```
descripteurs d'une trame (771)
        │  encodeur : Linear → ReLU → Linear → ReLU
        ▼
   latent z (64) ──[option sparse : L1 ou top-k]
        │
        ├──[option conditionné : FiLM]  h = (1 + γ(θ_visé)) · z + β(θ_visé)
        ▼
   décodeur : Linear → ReLU → Linear → Sigmoïde
        ▼
   masque (257)  ×  STFT du canal W  →  iSTFT  →  voix estimée
```

La perte est une L1 entre la magnitude estimée et la magnitude de la vraie voix (+ une pénalité L1 sur le latent pour `--sparse l1`). Le conditionnement FiLM reçoit `(cos θ_visé, sin θ_visé)`, ce qui évite la discontinuité entre −180° et +180°.

### Les modes d'entraînement

| Mode | Tâche | Étape |
|---|---|---|
| `fixed` | Extraire la voix située **devant (0°)**, sans conditionnement | 2 (dense) et 3 (sparse) |
| `conditioned` | Extraire la voix à l'**azimut demandé** (FiLM) | 4 (statique) et 5 (`--moving`) |
| `conditioned` + bruit d'azimut | Idem, avec une consigne volontairement bruitée à l'entraînement | 6 |

### Bruit sur l'azimut visé (augmentation de données)

À l'entraînement, la consigne donnée à FiLM est `θ_visé = θ_vrai + ε`, où `ε` est tiré au hasard :

| Type | Tirage | Effet attendu |
|---|---|---|
| Uniforme | `ε ~ U(−a, +a)` | Robustesse égale sur toute la plage `[−a, +a]`, frontière nette au-delà |
| Gaussien | `ε ~ N(0, σ²)` | Robustesse surtout pour les petites erreurs, queue de plus en plus rare (modèle plus réaliste d'un capteur ou d'un tracker) |

Règle des trois sigmas : avec `σ = 3°`, 99,7 % des erreurs vues à l'entraînement sont dans `±9°`. Une erreur de 30° n'est quasiment jamais vue, et on ne doit donc pas s'attendre à une robustesse à 30°. Pour couvrir 30°, il faut `σ ≈ 10°` ou un bruit uniforme d'amplitude ≥ 30°.

---

## 3. Structure du dépôt

```
ambisonics.py       encodage FOA (W, X, Y, Z)
scenes.py           scène à deux locuteurs + Dataset (partie « simulation », à personnaliser)
features.py         STFT / iSTFT, descripteurs spatiaux
beamformer.py       base de référence classique (micro virtuel cardioïde)
models.py           autoencodeur à masque (dense / sparse / FiLM)
metrics.py          SI-SDR
train.py            entraînement (courbes train / validation, option de bruit d'azimut)
evaluate.py         SI-SDR avant / beamformer / modèle, robustesse à l'erreur de direction
analyze_latent.py   carte « dimensions du latent × azimut »
config.py           constantes globales (SR = 16 kHz, clip de 2 s, N_FFT = 512, HOP = 128)
tests/              tests unitaires (pytest)
checkpoints/        modèles sauvegardés (.pt)
results/            CSV et figures produits par les scripts
```

---

## 4. Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pytest            # vérifie l'encodage, le beamformer, le SI-SDR, la STFT et le modèle
```

Dépendances : `torch`, `numpy`, `scipy`, `soundfile`, `matplotlib`, `pandas` (pour tracer les courbes d'évaluation), `pytest`.

---

## 5. Données

Il faut un dossier de fichiers audio de parole (`.flac` ou `.wav`, mono). Les scènes sont fabriquées à la volée par `scenes.py` : deux locuteurs **différents**, clips de 2 s, même niveau (SIR = 0 dB), directions séparées d'au moins 30°. Si le dossier est organisé en `locuteur/chapitre/fichier` (comme LibriSpeech), le code évite de mélanger deux fois le même locuteur. Les fichiers qui ne sont pas à 16 kHz sont rééchantillonnés automatiquement.

Utiliser des dossiers **séparés** pour l'entraînement et la validation/test, avec des **locuteurs différents** :

| Usage | Corpus conseillé (Mini LibriSpeech, OpenSLR SLR31) | Taille |
|---|---|---|
| Entraînement | `train-clean-5` | ~332 Mo |
| Validation / test | `dev-clean-2` | ~126 Mo |

Pour un travail plus solide : `train-clean-100` pour entraîner et `test-clean` pour évaluer (SLR12).

> Si `--val_dir` est omis, la validation réutilise les fichiers d'entraînement : les scènes diffèrent mais les voix restent les mêmes, ce qui biaise l'erreur de validation.

---

## 6. Protocole expérimental

Remplacer `DATA_TRAIN` et `DATA_TEST` par les dossiers voulus. Les scripts utilisent des valeurs par défaut modifiables en tête de fichier, ce qui permet de les lancer sans terminal (bouton Play de Spyder).

### Étape 1 — Base de référence (sans apprentissage)

Le beamformer est évalué à chaque appel de `evaluate.py` (colonne `sisdr_beamformer`).

### Étape 2 — Autoencodeur dense

```bash
python train.py --data_dir DATA_TRAIN --val_dir DATA_VAL --mode fixed --sparse none --name dense
python evaluate.py --data_dir DATA_TEST --ckpt checkpoints/dense.pt --out results/dense.csv
```

### Étape 3 — Autoencodeur sparse

```bash
python train.py --data_dir DATA_TRAIN --val_dir DATA_VAL --mode fixed --sparse topk --topk 8 --name sparse_topk
python train.py --data_dir DATA_TRAIN --val_dir DATA_VAL --mode fixed --sparse l1 --l1_weight 0.01 --name sparse_l1
python evaluate.py --data_dir DATA_TEST --ckpt checkpoints/sparse_topk.pt --out results/sparse_topk.csv
python analyze_latent.py --data_dir DATA_TEST --ckpt checkpoints/sparse_topk.pt
python analyze_latent.py --data_dir DATA_TEST --ckpt checkpoints/dense.pt --out results/latent_dense.png
```

### Étape 4 — Conditionnement par la direction

```bash
python train.py --data_dir DATA_TRAIN --val_dir DATA_VAL --mode conditioned --sparse topk --name cond
python evaluate.py --data_dir DATA_TEST --ckpt checkpoints/cond.pt \
                   --az_errors 0 10 20 45 90 --out results/cond.csv
```

`--az_errors` ajoute une erreur à la direction demandée : l'extraction marche-t-elle encore quand on vise à côté ? Le beamformer reçoit la même direction fautive, la comparaison reste équitable.

### Étape 5 — Sources mobiles

```bash
python train.py --data_dir DATA_TRAIN --val_dir DATA_VAL --mode conditioned --sparse topk --moving --name cond_moving
python evaluate.py --data_dir DATA_TEST --ckpt checkpoints/cond_moving.pt --moving \
                   --az_errors 0 2 5 10 15 20 30 --out results/cond_moving.csv
```

Chaque locuteur se déplace linéairement d'au plus 60° pendant le clip (`max_move_deg` dans `scenes.py`).

### Étape 6 — Robustesse : bruit sur l'azimut à l'entraînement

On entraîne plusieurs variantes de l'étape 5 qui ne diffèrent que par le bruit injecté sur la consigne d'azimut, puis on les évalue avec **la même** liste `--az_errors`.

| Nom du modèle | Bruit à l'entraînement | Paramètre |
|---|---|---|
| `cond_moving` | aucun (référence de l'étape 5) | – |
| `cond_moving_unif5` | uniforme | `a = 5°` |
| `cond_moving_unif15` | uniforme | `a = 15°` |
| `cond_moving_unif30` | uniforme | `a = 30°` |
| `cond_moving_gauss3` | gaussien | `σ = 3°` |
| `cond_moving_gauss10` | gaussien | `σ = 10°` |

Extrait du code d'entraînement (`compute_loss` dans `train.py`). **Un seul décalage par clip**, constant sur la durée, pour simuler une erreur de tracker plutôt qu'un bruit blanc sur l'angle :

```python
if model.conditioned and az_noise != "none":
    shape = (az.shape[0],) + (1,) * (az.dim() - 1)       # une erreur par clip
    if az_noise == "uniform":
        eps = (torch.rand(shape, device=az.device) * 2 - 1) * az_noise_scale   # U(-a, +a)
    else:  # "gauss"
        eps = torch.randn(shape, device=az.device) * az_noise_scale            # N(0, sigma^2)
    az_in = az + eps
else:
    az_in = az
az_frames = az_to_frames(az_in, X.shape[-1]) if model.conditioned else None
```

Points de protocole importants :

- **Le bruit ne s'applique qu'à l'entraînement.** La validation (`run_epoch` avec `optimizer=None`) doit utiliser l'azimut exact, sinon l'erreur de validation n'est plus comparable d'un modèle à l'autre ni à celle du modèle de référence.
- **Même graine, mêmes données, même nombre d'époques** pour toutes les variantes : seule la distribution du bruit change.
- **Même jeu d'évaluation** (`--seed` identique, `--moving` activé, mêmes `--az_errors`) pour toutes les variantes.
- Évaluer aussi avec des erreurs **négatives** (par exemple −30, −10) : un biais de signe éventuel serait invisible sinon.

```bash
python evaluate.py --data_dir DATA_TEST --ckpt checkpoints/cond_moving_gauss10.pt --moving \
                   --az_errors 0 2 5 10 15 20 30 --out results/cond_moving_gauss10.csv
```

---

## 7. Mesures

- **SI-SDR** (dB) avant extraction (canal W brut) et après : `sisdr_input`, `sisdr_beamformer`, `sisdr_model`.
- **Amélioration** : `improvement_beamformer`, `improvement_model` (SI-SDR après − avant).
- **Erreur de direction** : SI-SDR en fonction de `az_error_deg`.
- **Largeur de tolérance** : plus grande erreur `e*` telle que `sisdr_model(e*) ≥ sisdr_model(0) − 3 dB`.
- **Dense vs sparse**, **modèle appris vs beamformer**, **avec vs sans bruit d'azimut**, **uniforme vs gaussien**.
- **Carte du latent** : dimensions actives selon l'azimut (`analyze_latent.py`), avec la fraction de (dimension, azimut) actifs.
- **Courbes d'apprentissage** : erreur L1 d'entraînement et de validation par époque (`results/courbe_apprentissage_<nom>.png`).

### Cas d'échec à examiner

Directions voisines (`min_sep_deg` dans `scenes.py`), forte réverbération (à ajouter dans `scenes.py`), locuteurs trop proches, trajectoires qui se croisent.

---

## 8. Résultats

### 8.1 Résultats déjà obtenus (sans bruit d'azimut)

Corpus : `train-clean-5` (entraînement), `dev-clean-2` (validation), mélanges à 0 dB, évaluation sur 200 scènes.

**Courbes d'apprentissage (perte L1 sur la magnitude)**

| Modèle | Époques | Erreur train finale | Meilleure erreur validation |
|---|---|---|---|
| AE sparse top-k, `fixed` | 50 | 0,130 | 0,138 |
| AE sparse top-k, conditionné (FiLM), sources mobiles | 100 | 0,197 | 0,204 |

La tâche conditionnée et mobile est plus difficile, ce qui explique une erreur plus élevée. L'écart train / validation reste faible dans les deux cas : pas de surapprentissage marqué.

**SI-SDR (dB), erreur de visée nulle**

| Méthode | Scène | SI-SDR entrée | SI-SDR sortie | Amélioration |
|---|---|---|---|---|
| Beamformer cardioïde | sources fixes, cible à 0° | −0,01 | 14,26 | +14,26 |
| AE sparse top-k (`fixed`) | sources fixes, cible à 0° | −0,01 | 11,15 | +11,16 |
| Beamformer cardioïde | sources fixes, azimut quelconque | −0,01 | 9,62 | +9,62 |
| AE conditionné (FiLM) | sources fixes, azimut quelconque | −0,01 | 8,07 | +8,08 |
| Beamformer cardioïde | sources mobiles | −0,01 | 8,38 | +8,38 |
| AE conditionné (FiLM) | sources mobiles | −0,01 | 7,96 | +7,97 |

**Robustesse à l'erreur de visée, sources mobiles, modèle sans bruit d'azimut**

| Erreur d'azimut (°) | 0 | 2 | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|
| Beamformer (dB) | 8,38 | 8,44 | 8,65 | 8,54 | 8,28 | 8,05 | 7,72 |
| AE conditionné (dB) | 7,96 | 7,81 | 6,61 | 4,04 | 2,32 | 0,96 | −1,64 |

Figure : `results/courbe_robustesse.png`. Le beamformer cardioïde, dont le lobe est large, est presque insensible à l'erreur de visée ; l'autoencodeur est très directif et perd environ 1,3 dB à 5°, puis environ 4 dB à 10°.

**Carte du latent** (`results/latent_map.png`) : la parcimonie imposée (fraction active ≈ 0,28) est visible, et l'activation de certaines dimensions se déplace avec l'azimut (structure en « diagonale »), ce qui suggère un codage de la direction par des dimensions spécialisées.

### 8.2 Structure du latent avec bruit d'azimut (modèle entraîné avec bruit uniforme)

Carte obtenue avec `analyze_latent.py` sur le modèle entraîné avec un bruit **uniforme ±5°** sur l'azimut visé (checkpoint `cond_moving_unif5`, enregistré sous `modelUniform.pt` dans la version actuelle de `train.py`). Figure : `results/latent_map_unif5.png`, à comparer à `results/latent_map.png` (modèle sans bruit).

**Observations (à vérifier sur la figure)**

- **Activations plus étalées en azimut.** Les dimensions basses (environ 4 à 20) forment une diagonale par blocs plus épaisse et plus diffuse que sans bruit : chaque dimension répond sur une plage d'angles plus large.
- **Deux dimensions quasi constantes autour des indices 25 et 26**, fortement actives sur une grande partie de l'axe des azimuts (de −180° à environ −75°, puis de +120° à +150°) et éteintes dans la zone centrale et à droite. Elles semblent coder une information « hors champ » grossière.
- **Parcimonie conservée.** Les dimensions situées au-delà d'environ 28 restent à zéro. La contrainte `topk = 8` limite de toute façon le nombre de dimensions actives par trame ; la fraction active moyenne reste à mesurer et à reporter (`Fraction de (dimension, azimut) actifs`).

**Interprétation (hypothèse)** : un bruit sur la consigne FiLM oblige le décodeur à produire un masque correct pour des consignes légèrement décalées, ce qui peut se traduire par des réponses latentes moins étroites en azimut. Le latent est calculé **avant** FiLM, donc cette carte reflète surtout ce que l'encodeur a appris des descripteurs spatiaux, et non directement la consigne bruitée. Le lien de cause à effet est donc à établir.

**Ce que cette figure ne prouve pas** : elle ne mesure ni la qualité audio ni la robustesse. Affirmer que le modèle est « plus tolérant » ou qu'il perd « peu » de directivité demande les courbes SI-SDR du tableau 8.3 (modèle bruité vs modèle sans bruit, mêmes scènes, mêmes `--az_errors`).

À renseigner pour ce modèle : nombre d'époques réellement effectuées (la valeur par défaut de `--epochs` dans `train.py` était passée à 1 pour un essai), fraction active, meilleure erreur de validation.

### 8.3 Effet du bruit d'azimut sur le SI-SDR (à compléter)

SI-SDR (dB) en fonction de l'erreur de visée, sources mobiles, mêmes scènes d'évaluation pour toutes les lignes.

| Modèle | Bruit | 0° | 2° | 5° | 10° | 15° | 20° | 30° | Tolérance `e*` (−3 dB) |
|---|---|---|---|---|---|---|---|---|---|
| Beamformer cardioïde | – | 8,38 | 8,44 | 8,65 | 8,54 | 8,28 | 8,05 | 7,72 | |
| `cond_moving` | aucun | 7,96 | 7,81 | 6,61 | 4,04 | 2,32 | 0,96 | −1,64 | |
| `cond_moving_unif5` | U(±5°) | | | | | | | | |
| `cond_moving_unif15` | U(±15°) | | | | | | | | |
| `cond_moving_unif30` | U(±30°) | | | | | | | | |
| `cond_moving_gauss3` | N(0, 3°) | | | | | | | | |
| `cond_moving_gauss10` | N(0, 10°) | | | | | | | | |

Erreur L1 de validation (azimut exact) pour contrôler que le bruit ne dégrade pas l'ajustement :

| Modèle | Meilleure erreur validation | Époque |
|---|---|---|
| `cond_moving` | 0,204 | |
| `cond_moving_unif5` | | |
| `cond_moving_unif15` | | |
| `cond_moving_unif30` | | |
| `cond_moving_gauss3` | | |
| `cond_moving_gauss10` | | |

### 8.4 Tableau de synthèse du protocole (à compléter)

| Méthode | SI-SDR entrée | SI-SDR sortie | Amélioration |
|---|---|---|---|
| Beamformer cardioïde | | | |
| AE dense (fixed) | | | |
| AE sparse top-k (fixed) | | | |
| AE conditionné (FiLM) | | | |
| AE conditionné, sources mobiles | | | |
| AE conditionné, mobiles, bruit uniforme | | | |
| AE conditionné, mobiles, bruit gaussien | | | |

### 8.5 Figures à insérer

- SI-SDR vs erreur de direction : beamformer, modèle sans bruit, modèles uniforme et gaussien sur le même graphique.
- Courbes d'apprentissage de chaque modèle.
- Carte du latent : dense vs sparse, et sans bruit vs avec bruit d'azimut (uniforme, gaussien).

---

## 9. Discussion : le compromis directivité / robustesse

C'est le compromis fondamental de ce projet : **plus on veut que le modèle soit robuste aux grandes erreurs de visée, plus il faut élargir son masque à l'entraînement, et plus il perd en directivité pure.**

- **Sans bruit d'azimut**, le modèle FiLM est très sélectif : excellent si la direction demandée est exacte, mais il rejette la vraie voix dès que la consigne est décalée de quelques degrés. Il suppose donc un système de suivi (tracking) précis.
- **Avec bruit d'azimut**, on lui montre des consignes imparfaites : il apprend à ouvrir son masque. Premier indice qualitatif : avec bruit uniforme ±5°, la carte du latent est plus étalée en azimut (§8.2), ce qui reste à confirmer par les courbes SI-SDR. On s'attend à une courbe plus plate, au prix d'un SI-SDR potentiellement plus faible à erreur nulle (modèle « moins chirurgical »).
- **Uniforme vs gaussien** : l'uniforme garantit une robustesse homogène jusqu'à sa borne, le gaussien privilégie les petites erreurs réalistes et couvre mal les grandes. Le choix dépend du système de suivi visé.
- **Limite de la comparaison** : le beamformer cardioïde est large par construction, donc peu sensible à l'erreur de visée mais aussi peu sélectif. Une baseline plus directive (hypercardioïde, ordre supérieur) donnerait une comparaison plus exigeante.

Question à trancher dans les conclusions : garder le modèle ultra-directif (en précisant qu'il nécessite un tracking précis) ou adopter un modèle entraîné avec bruit large, plus tolérant mais moins directif. Le tableau 8.3 permet de répondre quantitativement.

*(Compléter cette section avec les observations réelles après les expériences de l'étape 6.)*

---

## 10. Limites

- Les scènes sont **simulées** (sources ponctuelles, deux locuteurs, pas de salle par défaut) : les conclusions valent pour ce cadre, pas en général.
- Le modèle travaille trame par trame (pas de contexte temporel) : choix de simplicité.
- Le masque est appliqué sur le canal W avec la phase du mélange, ce qui limite la qualité atteignable.
- Une seule graine par expérience : pour des conclusions solides, répéter avec plusieurs graines et donner la moyenne et l'écart-type.
- Le bruit d'azimut est modélisé comme un décalage constant par clip ; un vrai tracker produit aussi des erreurs variables dans le temps.
- Le beamformer de référence est un cardioïde simple : le comparer à une baseline plus directive serait plus rigoureux.

---

## 11. Pistes d'extension

- Réverbération (sources-images) dans `scenes.py`, niveaux différents entre locuteurs, bruit de fond.
- Contexte temporel (GRU / convolutions) et perte directement sur le SI-SDR.
- Bruit d'azimut **variable dans le temps**, ou **curriculum** (amplitude du bruit croissante au fil des époques).
- Mélange de bruits (gaussien + queue large) et balayage systématique de `a` et `σ`.
- Baselines plus directives (hypercardioïde, MVDR).

---

## 12. Dépannage (Windows / Spyder)

| Symptôme | Cause | Solution |
|---|---|---|
| `error: the following arguments are required: --data_dir` ou `--ckpt` | argument obligatoire non fourni | passer l'argument, ou définir une valeur `default=` dans le script |
| `PermissionError: ... checkpoints` | un **dossier** est passé à `--ckpt` | donner le chemin complet du fichier `.pt` |
| `PermissionError: ... results/eval.csv` | fichier ouvert dans Excel | fermer le fichier, relancer |
| `OMP: Error #15 ... libiomp5md.dll` puis kernel qui meurt | deux copies d'OpenMP | `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` **avant** l'import de `torch` (contournement non officiel) |
| `UserWarning: FigureCanvasAgg is non-interactive` | backend `Agg` actif | sans gravité, la figure est bien sauvegardée |
| `ValueError` dans le `DataLoader` avec `--moving` | forme de l'azimut ou fichier invalide, à diagnostiquer dans la trace complète | lire la fin de la trace, vérifier `val_dir`, tester sans `--moving` |

---

## 13. Références

- Marković et al. — filtrage spatial implicite pour la séparation multicanale.
- Paek et al. — autoencodeurs sparse dans les espaces latents audio.
- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer* (2018).
- Panayotov et al., *LibriSpeech: an ASR corpus based on public domain audio books* (2015).