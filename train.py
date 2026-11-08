import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import TRAIN_DIR, VAL_DIR, CKPT_DIR, RESULTS_DIR
from features import az_to_frames, spatial_features, stft
from models import MaskAutoencoder
from scenes import SceneDataset, list_audio_files



EXPERIENCES = [

    {"nom": "model_fixed_none",    "mode": "fixed",       "moving": False, "noise": None},
    {"nom": "model_cond_none",     "mode": "conditioned", "moving": True,  "noise": None},
    {"nom": "model_cond_Uniform",  "mode": "conditioned", "moving": True,  "noise": ("uniform", 5.0)},
    {"nom": "model_cond_gaussian", "mode": "conditioned", "moving": True,  "noise": ("gaussian", 5.0)},
    
    # U(±8,66°) a le même écart-type (5°) que N(0, 5²)
    {"nom": "model_cond_unif8p7",  "mode": "conditioned", "moving": True,  "noise": ("uniform", 8.66)},
    # N(0, 2,89²) a le même écart-type que U(±5°)
    {"nom": "model_cond_gauss2p9", "mode": "conditioned", "moving": True,  "noise": ("gaussian", 2.89)},
    # --- grandes amplitude
    {"nom": "model_cond_unif30",   "mode": "conditioned", "moving": True,  "noise": ("uniform", 30.0)},
    {"nom": "model_cond_gauss10",  "mode": "conditioned", "moving": True,  "noise": ("gaussian", 10.0)},
]

"""Bruit sur la consigne d'azimut, tiré indépendamment pour chaque échantillon.

Le modèle traite chaque trame séparément et la perte est une moyenne sur
les trames : un bruit indépendant par trame donne en espérance la même
perte qu'un décalage constant par clip.
"""
def sample_az_noise(az, noise):
  
    if noise is None:
        return az
    kind, scale = noise
    if kind == "uniform":
        return az + (torch.rand_like(az) * 2 - 1) * scale
    if kind == "gaussian":
        return az + torch.randn_like(az) * scale
    raise ValueError(f"type de bruit inconnu : {kind}")


def compute_loss(model, batch, l1_weight, device, noise=None):
    foa, target, az = batch["foa"].to(device), batch["target"].to(device), batch["az"].to(device)
    X = stft(foa)
    feats = spatial_features(X)

    az_frames = az_to_frames(sample_az_noise(az, noise), X.shape[-1]) if model.conditioned else None

    mask, z = model(feats, az_frames)
    est_mag = mask.transpose(1, 2) * X[:, 0].abs()
    target_mag = stft(target).abs()
    loss = F.l1_loss(est_mag, target_mag)
    if model.sparse == "l1":
        loss = loss + l1_weight * z.abs().mean()
    return loss


def run_epoch(model, loader, optimizer, l1_weight, device, noise=None):
    # noise n'est utilisé qu'à l'entraînement : la validation reçoit l'azimut exact.
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            loss = compute_loss(model, batch, l1_weight, device, noise=noise if train else None)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(TRAIN_DIR))
    p.add_argument("--val_dir", default=str(VAL_DIR))
    p.add_argument("--sparse", choices=["none", "l1", "topk"], default="topk")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--l1_weight", type=float, default=1e-2)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--items_per_epoch", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0,
                   help="graine ; si != 0, le nom du checkpoint reçoit le suffixe _s<seed>")
    p.add_argument("--only", nargs="+", default=None,
                   help="noms des expériences à lancer (par défaut : toutes)")
    p.add_argument("--retrain", action="store_true",
                   help="ré-entraîne même si le checkpoint existe déjà")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_files = list_audio_files(args.data_dir)
    val_files = list_audio_files(args.val_dir)
    assert train_files, f"Aucun fichier audio dans {args.data_dir}"
    assert val_files, f"Aucun fichier audio dans {args.val_dir}"

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    experiences = EXPERIENCES
    if args.only:
        inconnus = set(args.only) - {e["nom"] for e in EXPERIENCES}
        assert not inconnus, f"Expériences inconnues : {inconnus}"
        experiences = [e for e in EXPERIENCES if e["nom"] in args.only]

    for exp in experiences:
        nom = exp["nom"] + (f"_s{args.seed}" if args.seed != 0 else "")
        ckpt_path = CKPT_DIR / f"{nom}.pt"
        if ckpt_path.exists() and not args.retrain:
            print(f"[{nom}] checkpoint déjà présent, ignoré (--retrain pour forcer).")
            continue

        print(f"\n{'='*60}\nLANCEMENT DE L'EXPÉRIENCE : {nom}  (bruit : {exp['noise']})\n{'='*60}")

        torch.manual_seed(args.seed)
        fixed_az = 0.0 if exp["mode"] == "fixed" else None

        train_ds = SceneDataset(train_files, args.items_per_epoch, seed=None,
                                moving=exp["moving"], fixed_target_az=fixed_az)
        val_ds = SceneDataset(val_files, 200, seed=12345,
                              moving=exp["moving"], fixed_target_az=fixed_az)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = list(DataLoader(val_ds, batch_size=args.batch_size))  # scènes générées une seule fois

        cfg = dict(hidden=256, latent=args.latent, sparse=args.sparse, topk=args.topk,
                   conditioned=(exp["mode"] == "conditioned"))
        model = MaskAutoencoder(**cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best, best_epoch = float("inf"), 0
        train_losses, val_losses = [], []

        for epoch in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, optimizer, args.l1_weight, device, noise=exp["noise"])
            va = run_epoch(model, val_loader, None, args.l1_weight, device)
            train_losses.append(tr)
            val_losses.append(va)
            print(f"[{nom}] epoch {epoch:3d} | train {tr:.5f} | val {va:.5f}")

            if va < best:
                best, best_epoch = va, epoch
                torch.save({"state": model.state_dict(), "cfg": cfg, "mode": exp["mode"],
                            "noise": exp["noise"], "seed": args.seed,
                            "best_val": best, "best_epoch": best_epoch},
                           ckpt_path)

        plt.figure(figsize=(9, 5))
        plt.plot(range(1, args.epochs + 1), train_losses, label="Entraînement (consigne bruitée si bruit)", lw=2)
        plt.plot(range(1, args.epochs + 1), val_losses, label="Validation (azimut exact)", lw=2)
        plt.xlabel("Epoque")
        plt.ylabel("Perte L1")
        plt.title(f"Courbe d'apprentissage - {nom} (meilleure val : {best:.5f}, époque {best_epoch})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"loss_{nom}.png", dpi=150)
        plt.close()

    print("\nTOUS LES ENTRAÎNEMENTS SONT TERMINES.")


if __name__ == "__main__":
    main()