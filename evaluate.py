import argparse
import csv
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt 
import torch
import os
from torch.utils.data import DataLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from beamformer import cardioid_beamformer
from metrics import si_sdr
from models import MaskAutoencoder, extract_voice
from scenes import SceneDataset, list_audio_files


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = MaskAutoencoder(**ckpt["cfg"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt["mode"]

val_dir=r"C:\Users\AWOUNANG\Downloads\ambi-speaker-extraction\ambi-speaker-extraction\ambisonic-source-separation\TRAIN\LibriSpeech"
model_dir=r"C:\Users\AWOUNANG\Downloads\ambi-speaker-extraction\ambi-speaker-extraction\ambisonic-source-separation\checkpoints\modelUniform.pt"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=val_dir)
    p.add_argument("--ckpt", default=model_dir)
    p.add_argument("--n_items", type=int, default=200)
    p.add_argument("--moving", action="store_true",default=True)
    p.add_argument("--az_errors", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0],
                   help="erreurs (en degres) ajoutees à la direction visee")
    p.add_argument("--out", default="results/eval.csv")
    p.add_argument("--seed", type=int, default=999)
    args = p.parse_args()

    model, mode = load_model(args.ckpt)
    fixed_az = 0.0 if mode == "fixed" else None
    ds = SceneDataset(list_audio_files(args.data_dir), args.n_items, seed=args.seed,
                      moving=args.moving, fixed_target_az=fixed_az)
    loader = DataLoader(ds, batch_size=16)

    rows = []
    for err in args.az_errors:
        before, bf, ml = [], [], []
        with torch.no_grad():
            for batch in loader:
                foa, target = batch["foa"], batch["target"]
                az_asked = batch["az"] + err                       # direction demandée (avec erreur)
                before.append(si_sdr(foa[:, 0], target))           # canal W brut = mélange
                bf.append(si_sdr(cardioid_beamformer(foa, az_asked), target))
                ml.append(si_sdr(extract_voice(model, foa, az_asked), target))
        before, bf, ml = torch.cat(before), torch.cat(bf), torch.cat(ml)
        row = {"az_error_deg": err,
               "sisdr_input": before.mean().item(),
               "sisdr_beamformer": bf.mean().item(),
               "sisdr_model": ml.mean().item(),
               "improvement_beamformer": (bf - before).mean().item(),
               "improvement_model": (ml - before).mean().item()}
        rows.append(row)
        print({k: round(v, 2) for k, v in row.items()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Résultats écrits dans {args.out}")
    
    df = pd.read_csv(args.out)

# 2. Créer le graphique
    plt.figure(figsize=(7, 5))
    plt.plot(df["az_error_deg"], df["sisdr_beamformer"], label="Baseline : Beamformer", marker="o", color="red", linestyle="--")
    plt.plot(df["az_error_deg"], df["sisdr_model"], label="Votre IA : Autoencodeur FiLM", marker="s", color="green", lw=2)
    
    plt.xlabel("Erreur de visee de l'azimut ")
    plt.ylabel("Qualité audio finale (SI-SDR en dB)")
    plt.title("Robustesse du système face aux erreurs de direction")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # 3. Sauvegarder la figure
    plt.savefig("results/courbe_robustesseUniforme.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
