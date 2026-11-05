"""Paramètres globaux. Tout est regroupé ici pour être facile à modifier."""

SR = 16000            # fréquence d'échantillonnage (Hz)
CLIP_SECONDS = 2.0    # durée d'une scène
N_SAMPLES = int(SR * CLIP_SECONDS)

N_FFT = 512           # taille de la fenêtre STFT
HOP = 128             # pas entre deux fenêtres
N_FREQ = N_FFT // 2 + 1   # nombre de fréquences = 257

TARGET_RMS = 0.2      # niveau de chaque locuteur (donc rapport signal/interférent = 0 dB)
