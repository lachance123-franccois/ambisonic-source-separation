"""On place UN seul locuteur à différents azimuts, on calcule le latent moyen,
et on trace une carte (dimensions du latent x azimut). Les dimensions sont triées
par azimut préféré pour que la structure éventuelle soit visible."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ambisonics import encode_foa
from config import N_SAMPLES
from evaluate import load_model
from features import stft, spatial_features
from scenes import list_audio_files, load_audio, random_clip, normalize_rms

model_dir=r"C:\Users\AWOUNANG\Downloads\ambi-speaker-extraction\ambi-speaker-extraction\ambisonic-source-separation\checkpoints\model.pt"
data_dir=r"C:\Users\AWOUNANG\Downloads\ambi-speaker-extraction\ambi-speaker-extraction\ambisonic-source-separation\LibriSpeech"
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=data_dir)
    p.add_argument("--ckpt", default=model_dir)
    p.add_argument("--n_clips", type=int, default=20)
    p.add_argument("--step", type=int, default=30, help="pas en degrés entre deux azimuts")
    p.add_argument("--out", default="results/latent_map.png")
    args = p.parse_args()

    model, _ = load_model(args.ckpt)
    rng = np.random.default_rng(0)
    files = list_audio_files(args.data_dir)
    clips = [normalize_rms(random_clip(load_audio(files[rng.integers(len(files))]), N_SAMPLES, rng))
             for _ in range(args.n_clips)]

    azimuths = list(range(-180, 180, args.step))
    activity = []                                    # une ligne par azimut
    with torch.no_grad():
        for az in azimuths:
            foa = torch.from_numpy(np.stack([encode_foa(c, az) for c in clips]))   # (n_clips, 4, T)
            z = model.encode(spatial_features(stft(foa)))                           # (n_clips, N, latent)
            activity.append(z.mean(dim=(0, 1)).numpy())
    activity = np.stack(activity)                    # (n_az, latent)

    order = activity.argmax(axis=0).argsort()        # tri des dimensions par azimut préféré
    active_fraction = (activity > 1e-6).mean()
    print(f"Fraction de (dimension, azimut) actifs : {active_fraction:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.imshow(activity[:, order].T, aspect="auto", cmap="viridis",
               extent=[azimuths[0], azimuths[-1], 0, activity.shape[1]])
    plt.xlabel("Azimut du locuteur (°)")
    plt.ylabel("Dimension du latent (triée)")
    plt.colorbar(label="Activation moyenne")
    plt.title("Activation du latent selon la direction")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Figure sauvegardée dans {args.out}")


if __name__ == "__main__":
    main()
