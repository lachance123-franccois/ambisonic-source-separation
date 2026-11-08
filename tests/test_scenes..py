# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 00:19:31 2026

@author: AWOUNANG
"""
import numpy as np
import pytest
import soundfile as sf

from config import N_SAMPLES, SR, TARGET_RMS
from scenes import SceneDataset, list_audio_files, load_audio, normalize_rms, random_clip, speaker_of


@pytest.fixture
def audio_dir(tmp_path):
    rng = np.random.default_rng(0)
    for spk in ("100", "200", "300"):
        d = tmp_path / spk / "1"
        d.mkdir(parents=True)
        for k in range(2):
            sf.write(d / f"{spk}-1-{k}.flac", rng.standard_normal(3 * SR).astype(np.float32) * 0.1, SR)
    return tmp_path


def test_list_and_speaker(audio_dir):
    files = list_audio_files(audio_dir)
    assert len(files) == 6
    assert {speaker_of(f) for f in files} == {"100", "200", "300"}


def test_load_audio_resamples(tmp_path):
    x = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
    sf.write(tmp_path / "a.wav", x, 8000)                 # 1 s à 8 kHz
    y = load_audio(tmp_path / "a.wav")
    assert y.dtype == np.float32
    assert len(y) == pytest.approx(SR, abs=2)             # 1 s à 16 kHz


def test_random_clip_pads_short_signal():
    out = random_clip(np.ones(100, dtype=np.float32), 500, np.random.default_rng(0))
    assert len(out) == 500 and out[100:].sum() == 0


def test_normalize_rms():
    x = np.random.default_rng(0).standard_normal(1000)
    assert np.sqrt(np.mean(normalize_rms(x) ** 2)) == pytest.approx(TARGET_RMS, rel=1e-4)


@pytest.mark.parametrize("moving", [False, True])
def test_dataset_item(audio_dir, moving):
    ds = SceneDataset(list_audio_files(audio_dir), n_items=3, seed=0, moving=moving)
    item = ds[0]
    assert tuple(item["foa"].shape) == (4, N_SAMPLES)
    assert tuple(item["target"].shape) == (N_SAMPLES,)
    assert tuple(item["az"].shape) == (N_SAMPLES,)
    az = item["az"].numpy()
    if moving:
        assert abs(az[-1] - az[0]) <= ds.max_move_deg + 1e-3
    else:
        assert np.all(az == az[0])


def test_dataset_is_reproducible_with_seed(audio_dir):
    files = list_audio_files(audio_dir)
    a = SceneDataset(files, n_items=2, seed=5, moving=True)[1]
    b = SceneDataset(files, n_items=2, seed=5, moving=True)[1]
    assert np.array_equal(a["foa"].numpy(), b["foa"].numpy())


def test_fixed_target_azimuth(audio_dir):
    ds = SceneDataset(list_audio_files(audio_dir), n_items=2, seed=0, fixed_target_az=0.0)
    assert np.all(ds[0]["az"].numpy() == 0.0)