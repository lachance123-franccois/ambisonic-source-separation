import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Évite le crash OpenMP sous Windows

from features import stft
from models import MaskAutoencoder
from scenes import SceneDataset, list_audio_files
from features import spatial_features, az_to_frames


def compute_loss(model, batch, l1_weight, device):
    foa, target, az = batch["foa"].to(device), batch["target"].to(device), batch["az"].to(device)
    X = stft(foa)                                        # (B, 4, F, N)
    feats = spatial_features(X)   
#1)  Premier model                       # (B, N, 3F)
    az_frames = az_to_frames(az, X.shape[-1]) if model.conditioned else None
    
    #2 ) INTRODUCTION DU BRUUT UNIFORME/ GAUSSIEN dans les position a l'entrainement.
    if model.conditioned:
        bruit_angle = (torch.rand_like(az) * 10.0) - 5.0  # Génère une erreur aléatoire entre -5° et +5°
        az_bruite = az + bruit_angle
        az_frames = az_to_frames(az_bruite, X.shape[-1])
    else:
        az_frames = None
    mask, z = model(feats, az_frames)                    # (B, N, F), (B, N, latent)

    est_mag = mask.transpose(1, 2) * X[:, 0].abs()       # magnitude estimee (B, F, N)
    target_mag = stft(target).abs()                      # magnitude de la vraie voix
    loss = F.l1_loss(est_mag, target_mag)
    if model.sparse == "l1":
        loss = loss + l1_weight * z.abs().mean()         # pénalite de parcimonie
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

data_dir=r"C:\Users\AWOUNANG\Downloads\ambi-speaker-extraction\ambi-speaker-extraction\ambisonic-source-separation\LibriSpeech"
val_dir=r"C:\Users\AWOUNANG\Downloads\ambi-speaker-extraction\ambi-speaker-extraction\ambisonic-source-separation\TRAIN\LibriSpeech" 


best = float("inf")
train_losses = []
val_losses = []

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=data_dir, help="dossier d'entraînement (audio)")
    p.add_argument("--val_dir", default=val_dir, help="dossier de validation (sinon = data_dir)")
    p.add_argument("--mode", choices=["fixed", "conditioned"], default="conditioned")
    p.add_argument("--sparse", choices=["none", "l1", "topk"], default="topk")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--l1_weight", type=float, default=1e-2)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--moving", action="store_true",default=True, help="locuteurs mobiles (mode conditioned)")
    p.add_argument("--epochs", type=int, default=100)
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
    fixed_az = 0 if args.mode == "fixed" else None

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
        train_losses.append(tr)
        val_losses.append(va)
        print(f"epoch {epoch:3d} | train {tr:.5f} | val {va:.5f}")
        if va < best:
            best = va
            torch.save({"state": model.state_dict(), "cfg": cfg, "mode": args.mode},
                       f"checkpoints/{args.name}Uniform.pt")
    print(f"Meilleur modèle (val {best:.5f}) sauvegardé dans checkpoints/{args.name}.pt")
    

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Erreur d'Entraînement", color="blue", lw=2)
    plt.plot(range(1, args.epochs + 1), val_losses, label="Erreur de Validation", color="orange", lw=2)
    plt.xlabel("Epochs")
    plt.ylabel("Perte L1 (Magnitude spectrale)")
    plt.title(f"Courbe d'apprentissage - Modèle {args.name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    chart_path = f"results/courbe_apprentissage_{args.name}.png"
    plt.savefig(chart_path, dpi=150)
    print(f"\nEntraînement terminé ! Meilleure validation: {best:.5f}")
    print(f"Le graphique des erreurs a été sauvegardé dans : {chart_path}")


if __name__ == "__main__":
    main()
