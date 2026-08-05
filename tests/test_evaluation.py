"""Unit tests for Evaluation & Analysis Infrastructure (evaluation/)."""

from pathlib import Path
import numpy as np

from evaluation import (
    analyze_convergence,
    analyze_scalability,
    compute_confidence_interval,
    plot_ablation_comparison,
    plot_energy_efficiency_bar,
    plot_learning_curves,
    run_ablation_study,
)
from evaluation.convergence import perform_paired_ttest


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
    res_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"
    table_dir = tmp_path / "tables"

    algo_dir = res_dir / "hybrid_sac_dqn_seed42"
    algo_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "algorithm": "Hybrid_SAC_DDQN",
        "final_eval_reward": 150.5,
        "final_eval_power_w": 420.0,
        "final_qos_rate": 0.98,
    }

    import json

    with open(algo_dir / "summary.json", "w") as f:
        json.dump(summary, f)

    report = analyze_convergence(
        results_dir=str(res_dir),
        save_dir=str(fig_dir),
        table_save_dir=str(table_dir),
    )

    assert "Hybrid_SAC_DDQN" in report["algorithms"]
    assert (table_dir / "convergence_summary.tex").exists()


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
    fig_dir = str(tmp_path / "figures")

    res = analyze_scalability(
        episodes=2,
        save_dir=fig_dir,
    )

    # Concept Note v4.0's five mandated scalability points: R={5,12,20,35,50}.
    assert len(res) == 5
    assert (Path(fig_dir) / "scalability_analysis.pdf").exists()
