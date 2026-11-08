# -*- coding: utf-8 -*-
"""
Created on Mon Nov  03 00:12:21 2026

@author: AWOUNANG
"""
import numpy as np
import pytest
import torch

from ambisonics import encode_foa
from config import N_FREQ, N_SAMPLES
from features import spatial_features, stft
from models import MaskAutoencoder, extract_voice, keep_topk
from train import compute_loss, run_epoch, sample_az_noise


def make_batch(b=2, seed=0):
    rng = np.random.default_rng(seed)
    s1 = rng.standard_normal((b, N_SAMPLES)).astype(np.float32) * 0.2
    s2 = rng.standard_normal((b, N_SAMPLES)).astype(np.float32) * 0.2
    foa = np.stack([encode_foa(s1[i], 0.0) + encode_foa(s2[i], 120.0) for i in range(b)])
    return {"foa": torch.from_numpy(foa), "target": torch.from_numpy(s1),
            "az": torch.zeros(b, N_SAMPLES)}


def test_keep_topk():
    z = torch.arange(10.0)[None]
    out = keep_topk(z, 3)
    assert (out != 0).sum() == 3
    assert set(out[out != 0].tolist()) == {7.0, 8.0, 9.0}


@pytest.mark.parametrize("conditioned", [False, True])
def test_forward_shapes_and_mask_range(conditioned):
    torch.manual_seed(0)
    model = MaskAutoencoder(sparse="topk", topk=8, conditioned=conditioned)
    feats = spatial_features(stft(make_batch()["foa"]))
    B, N, _ = feats.shape
    az = torch.zeros(B, N) if conditioned else None
    mask, z = model(feats, az)
    assert mask.shape == (B, N, N_FREQ)
    assert z.shape == (B, N, 64)
    assert mask.min() >= 0 and mask.max() <= 1
    assert ((z != 0).sum(-1) <= 8).all()          # parcimonie top-k


def test_film_changes_output_with_azimuth():
    torch.manual_seed(0)
    model = MaskAutoencoder(conditioned=True)
    feats = spatial_features(stft(make_batch()["foa"]))
    B, N, _ = feats.shape
    m0, _ = model(feats, torch.zeros(B, N))
    m90, _ = model(feats, torch.full((B, N), 90.0))
    assert not torch.allclose(m0, m90)


@pytest.mark.parametrize("conditioned", [False, True])
def test_extract_voice_shape(conditioned):
    model = MaskAutoencoder(conditioned=conditioned).eval()
    batch = make_batch()
    with torch.no_grad():
        out = extract_voice(model, batch["foa"], batch["az"])
    assert out.shape == (2, N_SAMPLES)
    assert torch.isfinite(out).all()


def test_sample_az_noise_bounds_and_spread():
    torch.manual_seed(0)
    az = torch.zeros(4, 20000)
    assert torch.equal(sample_az_noise(az, None), az)
    u = sample_az_noise(az, ("uniform", 5.0))
    assert u.abs().max() <= 5.0
    assert u.std().item() == pytest.approx(5 / np.sqrt(3), rel=0.05)
    g = sample_az_noise(az, ("gaussian", 10.0))
    assert g.std().item() == pytest.approx(10.0, rel=0.05)
    with pytest.raises(ValueError):
        sample_az_noise(az, ("laplace", 1.0))


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = MaskAutoencoder(conditioned=True)
    batch = make_batch(seed=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = [batch]
    first = run_epoch(model, loader, None, 1e-2, "cpu")
    for _ in range(10):
        run_epoch(model, loader, opt, 1e-2, "cpu", noise=("gaussian", 5.0))
    last = run_epoch(model, loader, None, 1e-2, "cpu")
    assert np.isfinite(first) and last < first


def test_validation_ignores_noise():
    """Sans optimiseur, run_epoch doit ignorer le bruit : deux passes donnent la même perte."""
    torch.manual_seed(0)
    model = MaskAutoencoder(conditioned=True)
    loader = [make_batch()]
    a = run_epoch(model, loader, None, 1e-2, "cpu", noise=("uniform", 30.0))
    b = run_epoch(model, loader, None, 1e-2, "cpu", noise=("uniform", 30.0))
    assert a == pytest.approx(b)


def test_l1_penalty_increases_loss():
    torch.manual_seed(0)
    model = MaskAutoencoder(sparse="l1")
    batch = make_batch()
    with torch.no_grad():
        base = compute_loss(model, batch, 0.0, "cpu").item()
        pen = compute_loss(model, batch, 1.0, "cpu").item()
    assert pen >= base