"""Evaluation and analysis package for C-RAN Simulation."""

from evaluation.ablation import run_ablation_study
from evaluation.convergence import analyze_convergence, compute_cohens_d
from evaluation.csi_robustness import evaluate_csi_robustness
from evaluation.generalization import evaluate_generalization
from evaluation.inference_latency import benchmark_inference_latency
from evaluation.plot_utils import (
    compute_confidence_interval,
    plot_ablation_comparison,
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
    "evaluate_csi_robustness",
    "evaluate_generalization",
    "benchmark_inference_latency",
    "compute_confidence_interval",
    "plot_learning_curves",
    "plot_energy_efficiency_bar",
    "plot_scalability_analysis",
    "plot_ablation_comparison",
]
