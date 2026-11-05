"""Base de référence classique : un microphone virtuel orienté vers une direction.

En ambisonie du 1er ordre, on peut fabriquer un micro directionnel en combinant les canaux :
    sortie = p · W + (1 − p) · (X cos θ + Y sin θ)
    p = 0.5  -> cardioïde (rejette totalement ce qui vient de derrière)
    p = 0.25 -> hypercardioïde (plus directif)
Aucun apprentissage : c'est le niveau à battre.
"""
import torch


def cardioid_beamformer(foa, az_deg, p=0.5):
    if az_deg.dim() == 1:
        az_deg = az_deg[:, None]            # (B, 1) -> sera diffusé sur T
    az = torch.deg2rad(az_deg)
    w, x, y = foa[:, 0], foa[:, 1], foa[:, 2]
    return p * w + (1 - p) * (x * torch.cos(az) + y * torch.sin(az))
