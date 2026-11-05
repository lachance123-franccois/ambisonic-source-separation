#STFT et descripteurs spatiaux calculés à partir des 4 canaux ambisoniques.

#Idée clé : pour une seule source, le rapport entre X (ou Y) et W donne directement
#cos θ (ou sin θ). On calcule donc, pour chaque case temps-fréquence :
#    ix = Re(conj(W) · X) / |W|²   ≈ cos θ
#    iy = Re(conj(W) · Y) / |W|²   ≈ sin θ
#et on y ajoute le niveau log|W|. Ce sont les 3 "images" que le modèle reçoit.

import torch
from config import N_FFT, HOP, N_FREQ


def _window(x):
    return torch.hann_window(N_FFT, device=x.device)


def stft(x):
    shape = x.shape[:-1]
    out = torch.stft(x.reshape(-1, x.shape[-1]), N_FFT, HOP,
                     window=_window(x), return_complex=True)
    return out.reshape(*shape, *out.shape[-2:])


def istft(X, length):
    window = torch.hann_window(N_FFT, device=X.device)
    return torch.istft(X, N_FFT, HOP, window=window, length=length)


def spatial_features(foa_stft, eps=1e-6):
    W, X, Y = foa_stft[:, 0], foa_stft[:, 1], foa_stft[:, 2]
    power = W.abs() ** 2 + eps
    ix = ((W.conj() * X).real / power).clamp(-2, 2)  #cos(theta)
    iy = ((W.conj() * Y).real / power).clamp(-2, 2)  # sin(theta)
    level = torch.log1p(W.abs())
    feats = torch.stack([level, ix, iy], dim=1)            
    B, C, F, N = feats.shape
    return feats.permute(0, 3, 1, 2).reshape(B, N, C * F)


def az_to_frames(az, n_frames):
    #Ramène un azimut par échantillon (B, T) à un azimut par trame STFT (B, N)
    T = az.shape[-1]
    idx = (torch.arange(n_frames, device=az.device) * HOP).clamp(max=T - 1)
    return az[:, idx]
