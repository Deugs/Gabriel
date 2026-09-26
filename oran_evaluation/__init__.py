"""O-RAN Evaluation Package (additive, separate from evaluation/)."""

from oran_evaluation.convergence import (
    PROPOSED_ALGO,
    analyze_convergence,
    load_algo_seed_metrics,
)
from oran_evaluation.latency_benchmark import run_latency_benchmark
from oran_evaluation.multicriteria import compute_topsis, run_multicriteria_analysis
from oran_evaluation.plot_utils import compute_confidence_interval, plot_bar_comparison
from oran_evaluation.results_plots import (
    generate_comparison_plots,
    plot_convergence_curve,
    plot_pareto_scatter,
)

__all__ = [
    "analyze_convergence",
    "load_algo_seed_metrics",
    "PROPOSED_ALGO",
    "run_latency_benchmark",
    "compute_confidence_interval",
    "plot_bar_comparison",
    "generate_comparison_plots",
    "plot_pareto_scatter",
    "plot_convergence_curve",
    "compute_topsis",
    "run_multicriteria_analysis",
]
