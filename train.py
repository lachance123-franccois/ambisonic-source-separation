"""Entraînement.

Deux modes :
  --mode fixed        : le modèle extrait toujours la voix située DEVANT (0°). Pas de conditionnement.
                        (étapes 2 et 3 : autoencodeur dense, puis sparse)
  --mode conditioned  : le modèle reçoit l'azimut visé (FiLM) et extrait la voix à cette direction.
                        (étapes 4 et 5, avec --moving pour les sources mobiles)

Exemples :
  python train.py --data_dir LibriSpeech/train-clean-100 --mode fixed --sparse none --name dense
  python train.py --data_dir LibriSpeech/train-clean-100 --mode fixed --sparse topk --name sparse
  python train.py --data_dir LibriSpeech/train-clean-100 --mode conditioned --sparse topk --name cond
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from features import stft
from models import MaskAutoencoder
from scenes import SceneDataset, list_audio_files
from features import spatial_features, az_to_frames


def compute_loss(model, batch, l1_weight, device):
    foa, target, az = batch["foa"].to(device), batch["target"].to(device), batch["az"].to(device)
    X = stft(foa)                                        # (B, 4, F, N)
    feats = spatial_features(X)                          # (B, N, 3F)
    az_frames = az_to_frames(az, X.shape[-1]) if model.conditioned else None
    mask, z = model(feats, az_frames)                    # (B, N, F), (B, N, latent)

    est_mag = mask.transpose(1, 2) * X[:, 0].abs()       # magnitude estimée (B, F, N)
    target_mag = stft(target).abs()                      # magnitude de la vraie voix
    loss = F.l1_loss(est_mag, target_mag)
    if model.sparse == "l1":
        loss = loss + l1_weight * z.abs().mean()         # pénalité de parcimonie
    return loss


def run_epoch(model, loader, optimizer, l1_weight, device):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            loss = compute_loss(model, batch, l1_weight, device)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="dossier d'entraînement (audio)")
    p.add_argument("--val_dir", default=None, help="dossier de validation (sinon = data_dir)")
    p.add_argument("--mode", choices=["fixed", "conditioned"], default="fixed")
    p.add_argument("--sparse", choices=["none", "l1", "topk"], default="none")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--l1_weight", type=float, default=1e-2)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--moving", action="store_true", help="locuteurs mobiles (mode conditioned)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--items_per_epoch", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--name", default="model")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.moving and args.mode == "fixed":
        p.error("--moving n'a de sens qu'avec --mode conditioned")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed_az = 0.0 if args.mode == "fixed" else None

    train_files = list_audio_files(args.data_dir)
    val_files = list_audio_files(args.val_dir or args.data_dir)
    train_ds = SceneDataset(train_files, args.items_per_epoch, seed=None,
                            moving=args.moving, fixed_target_az=fixed_az)
    val_ds = SceneDataset(val_files, 200, seed=12345,
                          moving=args.moving, fixed_target_az=fixed_az)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    cfg = dict(hidden=256, latent=args.latent, sparse=args.sparse, topk=args.topk,
               conditioned=(args.mode == "conditioned"))
    model = MaskAutoencoder(**cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path("checkpoints").mkdir(exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, args.l1_weight, device)
        va = run_epoch(model, val_loader, None, args.l1_weight, device)
        print(f"epoch {epoch:3d} | train {tr:.5f} | val {va:.5f}")
        if va < best:
            best = va
            torch.save({"state": model.state_dict(), "cfg": cfg, "mode": args.mode},
                       f"checkpoints/{args.name}.pt")
    print(f"Meilleur modèle (val {best:.5f}) sauvegardé dans checkpoints/{args.name}.pt")


if __name__ == "__main__":
    main()
