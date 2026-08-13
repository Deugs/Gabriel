"""Unit tests for the ANN+GSBF baseline's trained-ANN stage (training/train_ann_gsbf.py)."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml  # type: ignore[import-untyped]

from baselines.ann_gsbf import ANNGSBFBaseline, ANNPredictor, extract_features
from cran_env import CRANEnv
from training.train_ann_gsbf import generate_labelled_dataset, train_ann_predictor


@pytest.fixture
def tiny_config_path(tmp_path):
    """A tiny (R=3, U=2) network config, kept small so the exhaustive
    label-generation search (1..R candidates per sample) stays fast in tests.
    """
    cfg = {
        "network": {"n_rrh": 3, "n_ue": 2, "n_bbu": 1, "bandwidth_mhz": 10},
        "power": {"rrh": {"p_max_dbm": 30}},
    }
    path = tmp_path / "tiny_network.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return str(path)


def test_extract_features_shape():
    gains_mag = np.random.rand(4, 3)
    demands_mbps = np.array([10.0, 20.0, 30.0])
    feat = extract_features(gains_mag, demands_mbps, n_rrh=4)

    assert feat.shape == (ANNPredictor.FEATURE_DIM,)
    assert feat.dtype == np.float32
    assert not np.isnan(feat).any()


def test_ann_predictor_forward_shape():
    model = ANNPredictor()
    x = torch.randn(5, ANNPredictor.FEATURE_DIM)
    out = model(x)

    assert out.shape == (5, 1)
    assert torch.all(out > 0.0) and torch.all(out <= 1.0)


def test_generate_labelled_dataset(tiny_config_path):
    features, labels = generate_labelled_dataset(
        tiny_config_path, n_samples=5, seed=0
    )

    assert features.shape == (5, ANNPredictor.FEATURE_DIM)
    assert labels.shape == (5,)
    assert np.all(labels > 0.0) and np.all(labels <= 1.0)


def test_train_ann_predictor_short_run(tiny_config_path, tmp_path):
    save_path = str(tmp_path / "ann_gsbf_predictor.pt")
    result = train_ann_predictor(
        config_path=tiny_config_path,
        n_samples=5,
        epochs=5,
        seed=0,
        save_path=save_path,
    )

    assert "final_loss" in result
    assert not np.isnan(result["final_loss"])
    assert Path(save_path).exists()

    # A baseline instantiated with this checkpoint should use the trained
    # ANN path (not the fallback heuristic) and still produce valid actions.
    with open(tiny_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env = CRANEnv(cfg)
    obs, _ = env.reset(seed=1)

    model = ANNGSBFBaseline(
        env.n_rrh, env.n_ue, env.p_max_w, checkpoint_path=save_path
    )
    assert model.ann is not None

    action = model.select_action(obs)
    assert action["rrh_on"].shape == (env.n_rrh,)
    assert action["power"].shape == (env.n_rrh,)
    assert np.all(action["power"] >= 0.0)
    assert np.all(action["power"] <= env.p_max_w + 1e-5)


def test_ann_gsbf_baseline_falls_back_without_checkpoint(tiny_config_path):
    """Without a trained checkpoint, ANNGSBFBaseline should still work via
    its fallback heuristic, not raise."""
    with open(tiny_config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env = CRANEnv(cfg)
    obs, _ = env.reset(seed=2)

    model = ANNGSBFBaseline(
        env.n_rrh, env.n_ue, env.p_max_w, checkpoint_path="/nonexistent/path.pt"
    )
    assert model.ann is None

    action = model.select_action(obs)
    assert action["rrh_on"].shape == (env.n_rrh,)
