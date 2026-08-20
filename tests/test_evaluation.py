"""Unit tests for Evaluation & Analysis Infrastructure (evaluation/)."""

from pathlib import Path
import numpy as np
import pytest

from evaluation import (
    analyze_convergence,
    analyze_scalability,
    compute_confidence_interval,
    plot_ablation_comparison,
    plot_energy_efficiency_bar,
    plot_learning_curves,
    run_ablation_study,
)
from evaluation.convergence import compute_cohens_d, perform_paired_ttest


def test_run_ablation_study_applies_variant_overrides(monkeypatch, tmp_path):
    """Each ablation variant's config_overrides must actually reach
    train_hybrid_agent, not just be defined and silently dropped."""
    import evaluation.ablation as ablation_module

    captured_overrides = []

    def fake_train_hybrid_agent(*, config_overrides=None, **kwargs):
        captured_overrides.append(config_overrides)
        return {"final_eval_reward": 0.0}

    monkeypatch.setattr(ablation_module, "train_hybrid_agent", fake_train_hybrid_agent)

    ablation_module.run_ablation_study(
        seeds=[42], episodes=1, save_dir=str(tmp_path / "figures")
    )

    assert captured_overrides == [
        {},
        {"gamma_switch": 0.0},
        {"gamma_fronthaul": 0.0},
        {"beta_qos": 0.0},
    ]


def test_compute_confidence_interval():
    data = np.random.randn(5, 20)
    mean, lower, upper = compute_confidence_interval(data)

    assert mean.shape == (20,)
    assert lower.shape == (20,)
    assert upper.shape == (20,)
    assert np.all(lower <= mean + 1e-5)
    assert np.all(mean <= upper + 1e-5)


def test_perform_paired_ttest():
    proposed = np.array([10.0, 12.0, 11.5, 13.0, 10.5])
    baseline = np.array([5.0, 6.0, 5.5, 6.5, 5.0])

    t_stat, p_val, is_sig = perform_paired_ttest(proposed, baseline)

    assert t_stat > 0.0
    assert p_val < 0.05
    assert is_sig is True


def test_plot_utils(tmp_path):
    fig_path = str(tmp_path / "test_curve.png")
    data = {"AlgoA": np.random.randn(3, 10), "AlgoB": np.random.randn(3, 10)}
    plot_learning_curves(data, save_path=fig_path)
    assert Path(fig_path).exists()

    bar_path = str(tmp_path / "test_bar.png")
    metrics = {"AlgoA": {"mean": 10.5, "std": 1.2}, "AlgoB": {"mean": 8.0, "std": 0.5}}
    plot_energy_efficiency_bar(metrics, save_path=bar_path)
    assert Path(bar_path).exists()

    ablation_path = str(tmp_path / "test_ablation.png")
    ablation_data = {"Variant1": 150.0, "Variant2": 120.0}
    plot_ablation_comparison(ablation_data, save_path=ablation_path)
    assert Path(ablation_path).exists()


def test_analyze_convergence_with_mock_results(tmp_path):
    """Also guards against the proposed-method name drifting out of sync with
    training/train_hybrid.py's actual saved `algorithm` field (it was previously
    hardcoded to the superseded "Hybrid_SAC_DDQN", silently breaking every
    paired t-test/Cohen's-d comparison against the real proposed method)."""
    res_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"
    table_dir = tmp_path / "tables"

    import json

    for algo, seed, reward in [
        ("Branching_MP_DQN", 42, 150.5),
        ("Branching_MP_DQN", 123, 148.0),
        ("DDQN", 42, 120.0),
        ("DDQN", 123, 118.5),
    ]:
        algo_dir = res_dir / f"{algo.lower()}_seed{seed}"
        algo_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "algorithm": algo,
            "seed": seed,
            "final_eval_reward": reward,
            "final_eval_power_w": 420.0,
            "final_qos_rate": 0.98,
        }
        with open(algo_dir / "summary.json", "w") as f:
            json.dump(summary, f)

    report = analyze_convergence(
        results_dir=str(res_dir),
        save_dir=str(fig_dir),
        table_save_dir=str(table_dir),
    )

    assert "Branching_MP_DQN" in report["algorithms"]
    assert "DDQN" in report["paired_ttests"]
    assert "cohens_d" in report["paired_ttests"]["DDQN"]
    assert report["paired_ttests"]["DDQN"]["n_paired_seeds"] == 2
    assert (table_dir / "convergence_summary.tex").exists()


def test_analyze_convergence_pairs_by_seed_not_list_position(tmp_path):
    """The paired t-test/Cohen's d must pair by seed, not by the order
    Path.rglob() happens to discover summary.json files in — the proposed
    method saves one summary.json per seed directory (discovered in
    filesystem order) while baselines save all seeds in one list (in
    numeric seed order), so list-position pairing can silently compare
    different seeds against each other."""
    res_dir = tmp_path / "results"
    import json

    # Proposed method: one file per seed, written in an order that would
    # NOT match ascending numeric seed order if paired by list position.
    for seed, reward in [(123, 100.0), (42, 200.0)]:
        algo_dir = res_dir / f"branching_mp_dqn_seed{seed}"
        algo_dir.mkdir(parents=True, exist_ok=True)
        with open(algo_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "algorithm": "Branching_MP_DQN",
                    "seed": seed,
                    "final_eval_reward": reward,
                    "final_eval_power_w": 0.0,
                    "final_qos_rate": 1.0,
                },
                f,
            )

    # Baseline: a single list, seeds in ascending numeric order.
    ddqn_dir = res_dir / "benchmark_ddqn"
    ddqn_dir.mkdir(parents=True, exist_ok=True)
    with open(ddqn_dir / "summary.json", "w") as f:
        json.dump(
            [
                {
                    "algorithm": "DDQN",
                    "seed": 42,
                    "mean_reward": 10.0,
                    "mean_power_w": 0.0,
                    "qos_satisfaction_rate": 1.0,
                },
                {
                    "algorithm": "DDQN",
                    "seed": 123,
                    "mean_reward": 20.0,
                    "mean_power_w": 0.0,
                    "qos_satisfaction_rate": 1.0,
                },
            ],
            f,
        )

    report = analyze_convergence(
        results_dir=str(res_dir),
        save_dir=str(tmp_path / "figures"),
        table_save_dir=str(tmp_path / "tables"),
    )

    ttest_info = report["paired_ttests"]["DDQN"]
    assert ttest_info["n_paired_seeds"] == 2

    # Correctly paired by seed: seed 42 -> (proposed=200.0, ddqn=10.0),
    # seed 123 -> (proposed=100.0, ddqn=20.0). Compute the expected result
    # from those seed-aligned arrays directly (rather than hand-deriving the
    # t-stat/Cohen's-d arithmetic) so the test can't itself encode an
    # arithmetic mistake.
    expected_t, expected_p, _ = perform_paired_ttest(
        np.array([200.0, 100.0]), np.array([10.0, 20.0])
    )
    expected_d = compute_cohens_d(np.array([200.0, 100.0]), np.array([10.0, 20.0]))
    assert ttest_info["t_statistic"] == pytest.approx(expected_t)
    assert ttest_info["p_value"] == pytest.approx(expected_p)
    assert ttest_info["cohens_d"] == pytest.approx(expected_d)

    # If pairing were done by list/write position instead of by seed --
    # proposed=[100.0, 200.0] (write order: seed123, seed42) against
    # ddqn=[10.0, 20.0] (list order: seed42, seed123) -- seed 123's proposed
    # score would be wrongly compared against seed 42's ddqn score and vice
    # versa, giving a different result. Guard against the fix silently
    # regressing to that wrong pairing.
    wrong_d = compute_cohens_d(np.array([100.0, 200.0]), np.array([10.0, 20.0]))
    assert ttest_info["cohens_d"] != pytest.approx(wrong_d)


def test_ablation_short_run(tmp_path):
    orig_path = Path(__file__).parent.parent / "config" / "default.yaml"
    fig_dir = str(tmp_path / "figures")

    res = run_ablation_study(
        config_path=str(orig_path),
        seeds=[42],
        episodes=2,
        save_dir=fig_dir,
    )

    assert len(res) == 4
    assert (Path(fig_dir) / "ablation_study.pdf").exists()


def test_scalability_short_run(tmp_path):
    orig_path = Path(__file__).parent.parent / "config" / "default.yaml"
    fig_dir = str(tmp_path / "figures")

    res = analyze_scalability(
        config_path=str(orig_path),
        episodes=2,
        save_dir=fig_dir,
    )

    assert len(res) == 5
    # RQ5 (Section 6) asks about the energy/QoS/switching trade-off vs.
    # scale, not just power/time -- guard against silently dropping these.
    for scale_metrics in res.values():
        algo_metrics = scale_metrics["Branching_MP_DQN"]
        assert "power" in algo_metrics
        assert "time" in algo_metrics
        assert "qos_rate" in algo_metrics
        assert "switching_events" in algo_metrics
    assert (Path(fig_dir) / "scalability_analysis.pdf").exists()
