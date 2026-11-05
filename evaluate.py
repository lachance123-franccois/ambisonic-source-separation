# Evaluation : SI-SDR avant extraction, après beamformer, après modèle.
#On peut aussi tester la robustesse à une erreur sur la direction visée (--az_errors).

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_items", type=int, default=200)
    p.add_argument("--moving", action="store_true")
    p.add_argument("--az_errors", type=float, nargs="+", default=[0.0],
                   help="erreurs (en degrés) ajoutées à la direction visée")
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


if __name__ == "__main__":
    main()
