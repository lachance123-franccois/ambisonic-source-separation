"""Autoencodeur à masque : trames spatiales -> latent -> masque spectral.

    entrée  : descripteurs spatiaux d'une trame (3 x 257 valeurs)
    encodeur: Linear -> ReLU -> Linear -> ReLU  => latent z (dimension 64)
    [option sparse] : on force z à être parcimonieux (L1 ou top-k)
    [option FiLM]   : on module z avec l'azimut visé : h = (1 + gamma) * z + beta
    décodeur: Linear -> ReLU -> Linear -> Sigmoïde => masque (257 valeurs entre 0 et 1)

La voix estimée = masque x STFT du canal W (omnidirectionnel), avec la phase du mélange.
"""
import torch
import torch.nn as nn

from config import N_FREQ
from features import stft, istft, spatial_features, az_to_frames


def keep_topk(z, k):
    """Garde les k plus grandes valeurs de chaque trame, met le reste à 0."""
    vals, idx = z.topk(k, dim=-1)
    return torch.zeros_like(z).scatter(-1, idx, vals)


class MaskAutoencoder(nn.Module):
    def __init__(self, hidden=256, latent=64, sparse="none", topk=8, conditioned=False):
        super().__init__()
        assert sparse in ("none", "l1", "topk")
        self.sparse, self.topk, self.conditioned = sparse, topk, conditioned
        self.encoder = nn.Sequential(
            nn.Linear(3 * N_FREQ, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.ReLU(),       # ReLU : latent >= 0
        )
        if conditioned:
            # Petit réseau qui transforme (cos az, sin az) en gamma et beta (FiLM)
            self.film = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2 * latent))
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, N_FREQ), nn.Sigmoid(),
        )

    def encode(self, feats):
        z = self.encoder(feats)
        if self.sparse == "topk":
            z = keep_topk(z, self.topk)
        return z

    def forward(self, feats, az_frames=None):
        """feats : (B, N, 3F). az_frames : (B, N) en degrés (si conditionné).
        Retourne le masque (B, N, F) et le latent z (B, N, latent)."""
        z = self.encode(feats)
        h = z
        if self.conditioned:
            a = torch.deg2rad(az_frames)
            cond = torch.stack([torch.cos(a), torch.sin(a)], dim=-1)    # (B, N, 2)
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            h = (1 + gamma) * z + beta
        return self.decoder(h), z


def extract_voice(model, foa, az_deg=None):
    """foa : (B, 4, T). az_deg : (B, T) azimut visé (ignoré si le modèle n'est pas conditionné).
    Retourne la voix estimée (B, T)."""
    T = foa.shape[-1]
    X = stft(foa)                                   # (B, 4, F, N)
    feats = spatial_features(X)                     # (B, N, 3F)
    az_frames = az_to_frames(az_deg, X.shape[-1]) if model.conditioned else None
    mask, _ = model(feats, az_frames)               # (B, N, F)
    est_stft = mask.transpose(1, 2) * X[:, 0]       # (B, F, N) : masque x canal W
    return istft(est_stft, T)
