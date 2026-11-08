import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ambisonics import encode_foa
from config import N_SAMPLES, TEST_DIR, CKPT_DIR, RESULTS_DIR
from evaluate import load_model, resolve_test_dir
from features import spatial_features, stft
from scenes import list_audio_files, load_audio, normalize_rms, random_clip

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(TEST_DIR))
    p.add_argument("--ckpt_dir", default=str(CKPT_DIR))
    p.add_argument("--n_clips", type=int, default=20)
    p.add_argument("--step", type=int, default=30, help="pas en degrés entre deux azimuts")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    files = list_audio_files(resolve_test_dir(args.data_dir))
    clips = [normalize_rms(random_clip(load_audio(files[rng.integers(len(files))]), N_SAMPLES, rng))
             for _ in range(args.n_clips)]

    azimuths = list(range(-180, 180, args.step))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Recherche tous les modèles .pt dans le dossier checkpoints
    model_paths = sorted(Path(args.ckpt_dir).glob("*.pt"))

    for path in model_paths:
        nom_modele = path.stem  # Récupère le nom sans l'extension .pt
        print(f"Analyse de l'espace latent pour : {nom_modele}...")
        model, _ = load_model(path)

        activity = []
        with torch.no_grad():
            for az in azimuths:
                foa = torch.from_numpy(np.stack([encode_foa(c, az) for c in clips]))
                z = model.encode(spatial_features(stft(foa)))
                activity.append(z.mean(dim=(0, 1)).numpy())
        activity = np.stack(activity)

        order = activity.argmax(axis=0).argsort()
        active_fraction = (activity > 1e-6).mean()
        print(f"   -> Fraction active ({nom_modele}) : {active_fraction:.2f}")

        out_path = RESULTS_DIR / f"latent_map_{nom_modele}.png"
        plt.figure(figsize=(8, 5))
        plt.imshow(activity[:, order].T, aspect="auto", cmap="viridis",
                   extent=[azimuths[0], azimuths[-1], 0, activity.shape[1]])
        plt.xlabel("Azimut du locuteur (°)")
        plt.ylabel("Dimension du latent (triée)")
        plt.colorbar(label="Activation moyenne")
        plt.title(f"Activation du latent - {nom_modele} (Actif: {active_fraction:.2f})")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"   -> Figure sauvegardée dans {out_path}")


if __name__ == "__main__":
    main()