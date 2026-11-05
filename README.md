# Extraction de locuteur par direction d'arrivée sur signaux ambisoniques

Projet pédagogique (PyTorch) : à partir d'une scène sonore enregistrée en **ambisonie du premier ordre**
(4 canaux) contenant deux locuteurs, extraire la voix qui vient d'une **direction donnée**,
avec un **autoencodeur dont l'espace latent est parcimonieux et conditionné par la position**.

> Ce dépôt contient le code et le protocole. Les résultats sont à produire en lançant les expériences
> ci-dessous (section « Résultats » à compléter).

---

## 1. Question de recherche

Un autoencodeur peut-il apprendre à extraire une voix selon sa direction, et fait-il mieux qu'un
simple microphone virtuel (beamformer) ? Qu'apporte la parcimonie du latent, et qu'apporte
le conditionnement par la direction visée ?

## 2. Principe

### Ambisonie du 1er ordre (convention SN3D)
Une source `s(t)` à l'azimut θ et l'élévation φ est encodée en 4 canaux :

| Canal | Formule |
|---|---|
| W | `s` |
| X | `s · cos θ · cos φ` |
| Y | `s · sin θ · cos φ` |
| Z | `s · sin φ` |

θ = 0° : devant ; θ = 90° : à gauche. Une scène à deux locuteurs est la **somme** des deux encodages.

### Descripteurs spatiaux
Pour une seule source dominante dans une case temps-fréquence, `Re(conj(W)·X)/|W|² ≈ cos θ`
et `Re(conj(W)·Y)/|W|² ≈ sin θ`. Le modèle reçoit donc, pour chaque trame STFT :
le niveau `log|W|` et ces deux rapports (3 × 257 valeurs).

### Modèle (`models.py`)
```
descripteurs d'une trame (771)
        │ encodeur : Linear → ReLU → Linear → ReLU
        ▼
   latent z (64)  ──[option sparse : L1 ou top-k]
        │
        ├──[option conditionné : FiLM]  h = (1 + γ(θ_visé)) · z + β(θ_visé)
        ▼
   décodeur : Linear → ReLU → Linear → Sigmoïde
        ▼
   masque (257)  ×  STFT du canal W  →  iSTFT  →  voix estimée
```
La perte est une L1 entre la magnitude estimée et la magnitude de la vraie voix
(+ une pénalité L1 sur le latent pour `--sparse l1`).

### Les deux modes d'entraînement
| Mode | Tâche | Étape |
|---|---|---|
| `fixed` | Extraire la voix située **devant (0°)**, sans conditionnement | 2 (dense) et 3 (sparse) |
| `conditioned` | Extraire la voix à l'**azimut demandé** (FiLM) | 4 (statique) et 5 (`--moving`) |

## 3. Structure du dépôt

```
ambisonics.py       encodage FOA (W, X, Y, Z)
scenes.py           scène à deux locuteurs + Dataset (partie « simulation », à personnaliser)
features.py         STFT / iSTFT, descripteurs spatiaux
beamformer.py       base de référence classique (micro virtuel cardioïde)
models.py           autoencodeur à masque (dense / sparse / FiLM)
metrics.py          SI-SDR
train.py            entraînement
evaluate.py         SI-SDR avant / beamformer / modèle, robustesse à l'erreur de direction
analyze_latent.py   carte « dimensions du latent × azimut »
tests/              tests unitaires (pytest)
```

## 4. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest            # vérifie l'encodage, le beamformer, le SI-SDR, la STFT et le modèle
```

## 5. Données

Il faut un dossier de fichiers audio de parole (`.flac` ou `.wav`, mono).
Les scènes sont fabriquées à la volée par `scenes.py` : deux locuteurs **différents**,
clips de 2 s, même niveau (SIR = 0 dB), directions séparées d'au moins 30°.
Si l'organisation du dossier est `locuteur/chapitre/fichier` (comme LibriSpeech),
le code évite de mélanger deux fois le même locuteur. Les fichiers qui ne sont pas à 16 kHz
sont rééchantillonnés automatiquement.

Utilise des dossiers **séparés** pour l'entraînement et le test (par exemple `train-clean-100`
pour entraîner, `test-clean` pour évaluer).

## 6. Protocole expérimental

Remplace `DATA_TRAIN` et `DATA_TEST` par tes dossiers.

### Étape 1 — Base de référence (sans apprentissage)
Le beamformer est évalué à chaque appel de `evaluate.py` (colonne `sisdr_beamformer`).

### Étape 2 — Autoencodeur dense
```bash
python train.py --data_dir DATA_TRAIN --mode fixed --sparse none --name dense
python evaluate.py --data_dir DATA_TEST --ckpt checkpoints/dense.pt --out results/dense.csv
```

### Étape 3 — Autoencodeur sparse
```bash
python train.py --data_dir DATA_TRAIN --mode fixed --sparse topk --topk 8 --name sparse_topk
python train.py --data_dir DATA_TRAIN --mode fixed --sparse l1 --l1_weight 0.01 --name sparse_l1
python evaluate.py --data_dir DATA_TEST --ckpt checkpoints/sparse_topk.pt --out results/sparse_topk.csv
python analyze_latent.py --data_dir DATA_TEST --ckpt checkpoints/sparse_topk.pt
python analyze_latent.py --data_dir DATA_TEST --ckpt checkpoints/dense.pt --out results/latent_dense.png
```

### Étape 4 — Conditionnement par la direction
```bash
python train.py --data_dir DATA_TRAIN --mode conditioned --sparse topk --name cond
python evaluate.py --data_dir DATA_TEST --ckpt checkpoints/cond.pt \
                   --az_errors 0 10 20 45 90 --out results/cond.csv
```
`--az_errors` ajoute une erreur à la direction demandée : l'extraction marche-t-elle encore
quand on vise à côté ? (le beamformer reçoit la même direction fautive, la comparaison reste équitable.)

### Étape 5 — Sources mobiles
```bash
python train.py --data_dir DATA_TRAIN --mode conditioned --sparse topk --moving --name cond_moving
python evaluate.py --data_dir DATA_TEST --ckpt checkpoints/cond_moving.pt --moving \
                   --az_errors 0 20 --out results/cond_moving.csv
```
Chaque locuteur se déplace linéairement d'au plus 60° pendant le clip (`max_move_deg` dans `scenes.py`).

## 7. Mesures

- **SI-SDR** (dB) avant extraction (canal W brut) et après : `sisdr_input`, `sisdr_beamformer`, `sisdr_model`.
- **Amélioration** : `improvement_beamformer`, `improvement_model` (SI-SDR après − avant).
- **Erreur de direction** : SI-SDR en fonction de `az_error_deg`.
- **Dense vs sparse**, **modèle appris vs beamformer**.
- **Carte du latent** : dimensions actives selon l'azimut (`analyze_latent.py`).

### Cas d'échec à examiner
Directions voisines (`min_sep_deg` dans `scenes.py`), forte réverbération (à ajouter dans `scenes.py`),
locuteurs trop proches, trajectoires qui se croisent.

## 8. Résultats

*À compléter après les expériences.*

| Méthode | SI-SDR entrée | SI-SDR sortie | Amélioration |
|---|---|---|---|
| Beamformer cardioïde | | | |
| AE dense (fixed) | | | |
| AE sparse top-k (fixed) | | | |
| AE conditionné (FiLM) | | | |
| AE conditionné, sources mobiles | | | |

Figures à insérer : SI-SDR vs erreur de direction ; carte du latent (dense vs sparse).

## 9. Limites

- Les scènes sont **simulées** (sources ponctuelles, deux locuteurs, pas de salle par défaut) :
  les conclusions valent pour ce cadre, pas en général.
- Le modèle travaille trame par trame (pas de contexte temporel) : choix de simplicité.
- Le masque est appliqué sur le canal W avec la phase du mélange, ce qui limite la qualité atteignable.
- Une seule graine par expérience : pour des conclusions solides, répéter avec plusieurs graines.

## 10. Pistes d'extension

Réverbération (sources-images) dans `scenes.py`, niveaux différents entre locuteurs,
bruit de fond, contexte temporel (GRU / convolutions), perte directement sur le SI-SDR.

## 11. Références

- Marković et al. — filtrage spatial implicite pour la séparation multicanale.
- Paek et al. — autoencodeurs sparse dans les espaces latents audio.
- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer* (2018).
