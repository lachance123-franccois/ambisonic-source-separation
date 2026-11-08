# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 00:15:07 2026

@author: AWOUNANG
"""

import numpy as np
import pytest
import torch

from ambisonics import encode_foa
from beamformer import cardioid_beamformer
from config import N_FREQ
from evaluate import mirror
from features import az_to_frames, istft, spatial_features, stft
from metrics import si_sdr

T = 16000


@pytest.fixture
def signal():
    return np.random.default_rng(0).standard_normal(T).astype(np.float32)


#  ambisonics.py 
def test_encode_shape_and_dtype(signal):
    foa = encode_foa(signal, 30.0)
    assert foa.shape == (4, T)
    assert foa.dtype == np.float32


@pytest.mark.parametrize("az, gx, gy", [(0, 1, 0), (90, 0, 1), (180, -1, 0), (-90, 0, -1)])
def test_encode_gains(signal, az, gx, gy):
    w, x, y, z = encode_foa(signal, az)
    np.testing.assert_allclose(w, signal, atol=1e-6)
    np.testing.assert_allclose(x, gx * signal, atol=1e-5)
    np.testing.assert_allclose(y, gy * signal, atol=1e-5)
    np.testing.assert_allclose(z, 0.0, atol=1e-6)


def test_encode_moving_source(signal):
    az = np.linspace(0, 90, T)
    _, x, y, _ = encode_foa(signal, az)
    assert x[0] == pytest.approx(signal[0], abs=1e-5)
    assert y[-1] == pytest.approx(signal[-1], abs=1e-5)


#  beamformer.py 
def test_cardioid_passes_target_and_rejects_rear(signal):
    foa_front = torch.from_numpy(encode_foa(signal, 0.0))[None]
    foa_back = torch.from_numpy(encode_foa(signal, 180.0))[None]
    aim = torch.tensor([0.0])
    torch.testing.assert_close(cardioid_beamformer(foa_front, aim)[0], torch.from_numpy(signal), atol=1e-5, rtol=0)
    assert cardioid_beamformer(foa_back, aim).abs().max() < 1e-5


def test_beamformer_accepts_azimuth_per_sample(signal):
    foa = torch.from_numpy(encode_foa(signal, 45.0))[None]
    out = cardioid_beamformer(foa, torch.full((1, T), 45.0))
    torch.testing.assert_close(out[0], torch.from_numpy(signal), atol=1e-5, rtol=0)


#  metrics.py 
def _noise(seed=1):
    return torch.from_numpy(np.random.default_rng(seed).standard_normal(T).astype(np.float32))[None]


def test_si_sdr_perfect_estimate_is_very_high(signal):
    ref = torch.from_numpy(signal)[None]
    assert si_sdr(ref, ref).item() > 60


def test_si_sdr_is_gain_invariant(signal):
    ref = torch.from_numpy(signal)[None]
    est = ref + 0.3 * _noise()
    assert si_sdr(3.0 * est, ref).item() == pytest.approx(si_sdr(est, ref).item(), abs=1e-3)


def test_si_sdr_zero_db_for_equal_power_orthogonal_noise(signal):
    ref = torch.from_numpy(signal)[None]
    ref = ref - ref.mean()                                                 # si_sdr centre les signaux
    noise = _noise()
    noise = noise - noise.mean()
    noise = noise - (noise * ref).sum() / (ref * ref).sum() * ref          # orthogonal to ref
    noise = noise * ref.norm() / noise.norm()                              # same power
    assert si_sdr(ref + noise, ref).item() == pytest.approx(0.0, abs=0.1)


#  features.py 
def test_stft_istft_roundtrip(signal):
    x = torch.from_numpy(signal)[None]
    X = stft(x)
    assert X.shape[-2] == N_FREQ
    torch.testing.assert_close(istft(X, T), x, atol=1e-4, rtol=0)


@pytest.mark.parametrize("az", [0.0, 60.0, -120.0])
def test_spatial_features_recover_direction(signal, az):
    foa = torch.from_numpy(encode_foa(signal, az))[None]
    X = stft(foa)
    feats = spatial_features(X)
    B, N, C = feats.shape
    assert (B, C) == (1, 3 * N_FREQ) and N == X.shape[-1]
    ix = feats[0, :, N_FREQ:2 * N_FREQ]
    iy = feats[0, :, 2 * N_FREQ:]
    a = np.deg2rad(az)
    assert ix.median().item() == pytest.approx(np.cos(a), abs=1e-3)
    assert iy.median().item() == pytest.approx(np.sin(a), abs=1e-3)


def test_az_to_frames_shape():
    az = torch.linspace(-90, 90, T)[None].repeat(2, 1)
    out = az_to_frames(az, 126)
    assert out.shape == (2, 126)
    assert out[0, 0].item() == pytest.approx(-90.0)


#  evaluate.py : scène miroir-
def test_mirror_equals_encoding_at_opposite_azimuth(signal):
    foa = torch.from_numpy(encode_foa(signal, 40.0))[None]
    original = foa.clone()
    az = torch.full((1, T), 40.0)
    foa_m, az_m = mirror(foa, az)
    expected = torch.from_numpy(encode_foa(signal, -40.0))[None]
    torch.testing.assert_close(foa_m, expected, atol=1e-5, rtol=0)
    assert torch.all(az_m == -40.0)
    assert torch.equal(foa, original)                                      # l'original n'est pas modifié