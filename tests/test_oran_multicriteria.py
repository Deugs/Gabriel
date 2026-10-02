"""Unit tests for oran_evaluation/multicriteria.py's MCDA methods.

A fully separate test module from tests/test_oran_evaluation.py, matching
this repo's one-test-file-per-module convention for the O-RAN track.
"""

from pathlib import Path

import pytest

from oran_evaluation.multicriteria import (
    CRITERIA,
    compute_entropy_weights,
    compute_topsis,
    compute_vikor,
    export_mcda_robustness_table,
    export_multicriteria_table,
    plot_ranking_robustness,
)


def _metrics(rows):
    """Build a {algo: {seed: {criterion: value}}} dict from
    {algo: {criterion: value}} rows, using a single seed per algo (these
    tests only exercise the ranking math, not seed-aggregation)."""
    return {algo: {0: vals} for algo, vals in rows.items()}


_BEST = {
    "power": 50.0,
    "qos": 0.99,
    "qos_per_ue": 0.99,
    "switching": 0.01,
    "throughput": 2000.0,
}
_WORST = {
    "power": 300.0,
    "qos": 0.40,
    "qos_per_ue": 0.40,
    "switching": 2.0,
    "throughput": 200.0,
}
_MIDDLE = {
    "power": 150.0,
    "qos": 0.70,
    "qos_per_ue": 0.70,
    "switching": 0.5,
    "throughput": 1000.0,
}


def test_compute_topsis_ranks_dominant_alternative_first():
    """An alternative that is simultaneously best on every criterion must
    rank #1 regardless of weighting -- the simplest possible sanity check
    on the ideal-best/ideal-worst distance math."""
    metrics = _metrics({"best": _BEST, "worst": _WORST, "middle": _MIDDLE})
    result = compute_topsis(metrics)
    assert result["best"]["rank"] == 1
    assert result["worst"]["rank"] == 3
    assert 0.0 <= result["best"]["topsis_score"] <= 1.0


def test_compute_topsis_equal_weights_by_default():
    metrics = _metrics(
        {"a": {c: 1.0 for c in CRITERIA}, "b": {c: 2.0 for c in CRITERIA}}
    )
    result = compute_topsis(metrics)
    assert set(result.keys()) == {"a", "b"}


def test_exported_tables_escape_percent_signs(tmp_path):
    """Regression guard: CRITERION_LABELS' literal '(%)' (correct for the
    matplotlib labels this same dict feeds) must be escaped to '(\\%)' in
    the LaTeX tables specifically -- an unescaped '%' starts a LaTeX
    comment, silently truncating the rest of that line (a real compile
    bug this once caused)."""
    metrics = _metrics({"a": _BEST, "b": _WORST})
    topsis = compute_topsis(metrics)
    multicriteria_path = export_multicriteria_table(topsis, str(tmp_path))
    multicriteria_tex = Path(multicriteria_path).read_text()
    assert "(%)" not in multicriteria_tex
    assert "(\\%)" in multicriteria_tex

    entropy_weights = compute_entropy_weights(metrics)
    vikor = compute_vikor(metrics, entropy_weights)
    robustness_path = export_mcda_robustness_table(
        topsis, topsis, vikor, entropy_weights, str(tmp_path)
    )
    robustness_tex = Path(robustness_path).read_text()
    assert "(%)" not in robustness_tex
    assert "(\\%)" in robustness_tex


def test_compute_entropy_weights_sum_to_one_and_zero_for_constant_criterion():
    """A criterion identical across every method carries no discriminating
    information and must get weight ~0; the remaining weights must still
    sum to 1."""
    tied = {"qos": 0.5, "qos_per_ue": 0.9}
    metrics = _metrics(
        {
            "a": {"power": 100.0, "switching": 0.5, "throughput": 500.0, **tied},
            "b": {"power": 200.0, "switching": 1.5, "throughput": 1500.0, **tied},
        }
    )
    weights = compute_entropy_weights(metrics)
    assert set(weights.keys()) == set(CRITERIA)
    assert weights["qos"] == 0.0
    assert weights["qos_per_ue"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["power"] > 0.0
    assert weights["throughput"] > 0.0


def test_compute_entropy_weights_falls_back_to_equal_when_everything_tied():
    metrics = _metrics(
        {"a": {c: 1.0 for c in CRITERIA}, "b": {c: 1.0 for c in CRITERIA}}
    )
    weights = compute_entropy_weights(metrics)
    expected = 1.0 / len(CRITERIA)
    for c in CRITERIA:
        assert weights[c] == pytest.approx(expected)


def test_compute_vikor_ranks_dominant_alternative_first():
    metrics = _metrics({"best": _BEST, "worst": _WORST, "middle": _MIDDLE})
    result = compute_vikor(metrics)
    assert result["best"]["rank"] == 1
    assert result["worst"]["rank"] == 3
    # Lower Q is better -- the dominant alternative's Q must be the minimum.
    assert result["best"]["vikor_q"] <= result["middle"]["vikor_q"]
    assert result["middle"]["vikor_q"] <= result["worst"]["vikor_q"]


def test_compute_vikor_handles_tied_criterion_without_dividing_by_zero():
    """A criterion with identical values across all methods (zero spread)
    must not raise or produce NaN/inf -- it should simply contribute zero
    to every method's S and R."""
    tied = {"qos": 0.5, "qos_per_ue": 0.9}
    metrics = _metrics(
        {
            "a": {"power": 100.0, "switching": 0.5, "throughput": 500.0, **tied},
            "b": {"power": 200.0, "switching": 1.5, "throughput": 1500.0, **tied},
        }
    )
    result = compute_vikor(metrics)
    for r in result.values():
        assert r["vikor_q"] == r["vikor_q"]  # not NaN
        assert abs(r["vikor_q"]) != float("inf")


def test_plot_ranking_robustness_writes_pdf_and_png(tmp_path):
    metrics = _metrics({"BMPP_DQN": _BEST, "ddpg": _WORST, "dqn": _MIDDLE})
    topsis_equal = compute_topsis(metrics)
    entropy_weights = compute_entropy_weights(metrics)
    topsis_entropy = compute_topsis(metrics, weights=entropy_weights)
    vikor_results = compute_vikor(metrics, weights=entropy_weights)

    save_path = tmp_path / "mcda_robustness_oran.pdf"
    plot_ranking_robustness(topsis_equal, topsis_entropy, vikor_results, str(save_path))

    assert save_path.exists()
    assert save_path.with_suffix(".png").exists()
