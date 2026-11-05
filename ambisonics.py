"""Encodage ambisonique du premier ordre (FOA), convention SN3D.

Une source s(t) à l'azimut θ et l'élévation φ donne 4 canaux :
    W = s                      (omnidirectionnel)
    X = s · cos θ · cos φ      (avant / arrière)
    Y = s · sin θ · cos φ      (gauche / droite)
    Z = s · sin φ              (haut / bas)

θ = 0° : devant, θ = 90° : à gauche (sens trigonométrique).
"""
import numpy as np


def encode_foa(signal, azimuth_deg, elevation_deg=0.0):
    """signal : (T,). azimuth_deg : nombre OU tableau (T,) si la source bouge.
    Retourne un tableau (4, T) : ordre W, X, Y, Z."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    w = signal
    x = signal * np.cos(az) * np.cos(el)
    y = signal * np.sin(az) * np.cos(el)
    z = signal * np.sin(el) * np.ones_like(signal)
    return np.stack([w, x, y, z]).astype(np.float32)
