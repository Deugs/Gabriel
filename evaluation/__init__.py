"""Evaluation and analysis package for C-RAN Simulation."""

from evaluation.ablation import run_ablation_study
from evaluation.convergence import analyze_convergence
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
    "run_ablation_study",
    "analyze_scalability",
    "compute_confidence_interval",
    "plot_learning_curves",
    "plot_energy_efficiency_bar",
    "plot_scalability_analysis",
    "plot_ablation_comparison",
]
