# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 00:18:11 2026

@author: AWOUNANG
"""
"""Statistiques et figures de plot_results.py, sur des SI-SDR par scène synthétiques."""
import numpy as np
import pandas as pd
import pytest

from plot_results import make_all, summarize, tolerance


def test_tolerance_interpolates():
    # 10 dB à 0°, 8 dB à 10°, 4 dB à 20° : seuil 7 dB atteint à 12,5°
    assert tolerance([0, 10, 20], [10, 8, 4]) == pytest.approx(12.5)


def test_tolerance_infinite_and_missing_zero():
    assert np.isinf(tolerance([0, 10, 20], [10, 9.5, 9]))
    assert np.isnan(tolerance([5, 10], [10, 9]))


@pytest.fixture
def per_scene(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for nom, mode in [("model_cond_none", "conditioned"), ("model_cond_unif30", "conditioned"),
                      ("model_fixed_none", "fixed")]:
        for err in (-10.0, 0.0, 10.0):
            for sc in range(30):
                rows.append({"model_name": nom, "mode": mode, "moving": mode != "fixed", "mirror": False,
                             "az_error_deg": err, "scene": sc, "sisdr_input": 0.0,
                             "sisdr_beamformer": rng.choice([2.0, 40.0], p=[0.8, 0.2]),
                             "sisdr_hypercardioid": 5.0,
                             "sisdr_model": 8.0 - abs(err) / 5 + rng.normal(0, 0.1)})
    path = tmp_path / "eval_par_scene.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_summary_mean_median_and_wins(per_scene):
    s = summarize(pd.read_csv(per_scene))
    r = s[(s.model_name == "model_cond_none") & (s.az_error_deg == 0)].iloc[0]
    assert r.median_beamformer == pytest.approx(2.0)          # médiane robuste aux 40 dB
    assert r.mean_beamformer > r.median_beamformer             # moyenne tirée par les 40 dB
    assert r.mean_cap30_beamformer < r.mean_beamformer         # plafonnement
    assert 0.6 < r.win_rate_vs_bf < 1.0
    assert r.mean_model_ci_low <= r.mean_model <= r.mean_model_ci_high


def test_make_all_writes_outputs(per_scene, tmp_path):
    out = tmp_path / "res"
    make_all(per_scene, out)
    for name in ["eval_comparatif.csv", "tolerance.csv", "asymetrie.csv", "robustesse_amplitude.png",
                 "compromis.png", "victoires.png", "distribution_0deg.png", "fixe_vs_beamformer.png"]:
        assert (out / name).exists(), name
