"""Paramètres globaux. Tout est regroupé ici pour être facile à modifier."""
import os
from pathlib import Path

# --- Signal -----------------------------------------------------------------
SR = 16000            # fréquence d'échantillonnage (Hz)
CLIP_SECONDS = 2.0    # durée d'une scène
N_SAMPLES = int(SR * CLIP_SECONDS)

N_FFT = 512           # taille de la fenêtre STFT
HOP = 128             # pas entre deux fenêtres
N_FREQ = N_FFT // 2 + 1   # nombre de fréquences = 257

TARGET_RMS = 0.2      # niveau de chaque locuteur (donc rapport signal/interférent = 0 dB)

# --- Scènes -----------------------------------------------------------------
# Écart angulaire minimal entre cible et interférent au départ du clip.
#   -90 : AUCUN écart garanti (sources superposables) et interférent 2x plus
#         probable à moins de 90° de la cible. C'est la valeur qui a produit les
#         résultats actuels du README (campagne à 4 modèles).
#    30 : écart minimal de 30°, interférent uniforme sur le reste du cercle.
# Changer cette valeur change la difficulté des scènes : ré-entraîner ET ré-évaluer.
MIN_SEP_DEG = -90
MAX_MOVE_DEG = 90     # déplacement maximal d'un locuteur sur la durée du clip (mode mobile)

# --- Dossiers ---------------------------------------------------------------
# Chemins relatifs au dossier du projet (là où se trouve ce fichier), donc
# valables sous Windows comme sous Linux. On peut les surcharger avec des
# variables d'environnement (AMBI_TRAIN_DIR, AMBI_VAL_DIR, AMBI_TEST_DIR).
ROOT = Path(__file__).resolve().parent

TRAIN_DIR = Path(os.environ.get("AMBI_TRAIN_DIR", ROOT / "LibriSpeech"))
VAL_DIR = Path(os.environ.get("AMBI_VAL_DIR", ROOT / "TRAIN" / "LibriSpeech"))
# Jeu de test : locuteurs distincts de ceux de la validation (ex. test-clean).
# S'il n'existe pas, evaluate.py se rabat sur VAL_DIR en affichant un avertissement.
TEST_DIR = Path(os.environ.get("AMBI_TEST_DIR", ROOT / "TEST" / "LibriSpeech"))

CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"