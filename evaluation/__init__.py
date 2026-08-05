"""Evaluation and analysis package for C-RAN Simulation."""

from evaluation.ablation import run_ablation_study
from evaluation.convergence import analyze_convergence, compute_cohens_d
from evaluation.csi_robustness import run_csi_robustness_evaluation
from evaluation.generalization import run_generalization_evaluation
from evaluation.latency_benchmark import run_latency_benchmark
from evaluation.plot_utils import (
    compute_confidence_interval,
    plot_ablation_comparison,
    plot_degradation_curve,
    plot_energy_efficiency_bar,
    plot_learning_curves,
    plot_scalability_analysis,
)
from evaluation.scalability import analyze_scalability

__all__ = [
    "analyze_convergence",
    "compute_cohens_d",
    "run_ablation_study",
    "analyze_scalability",
    "run_csi_robustness_evaluation",
    "run_generalization_evaluation",
    "run_latency_benchmark",
    "compute_confidence_interval",
    "plot_learning_curves",
    "plot_energy_efficiency_bar",
    "plot_scalability_analysis",
    "plot_ablation_comparison",
    "plot_degradation_curve",
]
