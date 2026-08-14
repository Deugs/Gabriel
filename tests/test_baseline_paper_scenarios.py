"""Qualitative baseline validation at the exact scenarios Iqbal et al.
(2021) and Fathy et al. (2021) study.

Concept Note v4.0 Section 14's risk-mitigation row commits to validating
each reproduced baseline "against its source paper's reported operating
point." Exact numeric agreement with the papers' published Tables/Figures
cannot be verified in this environment: no primary-source access is
available here (the same constraint already documented for the HySoft
citation, Concept Note v4.0 Section 4.2's verification note), and two
independent attempts to obtain the real papers during this audit did not
succeed (a sci-hub-style mirror is blocked by this environment's network
egress policy; a manually supplied PDF turned out to be an unrelated paper
by a different author who happens to share a surname with Fathy, Abood &
Hamdi, 2021). Fabricating a specific "reported number" that cannot be
checked against a primary source would violate the project's own Ethical
AI Rule (docs/rules.md Section 10) -- the same standard applied to the
OREO citation (initially retracted for not resolving via search, then
restored once the primary source was actually obtained and read -- a
failed search is inconclusive, not disconfirming) and to the
not-yet-primary-source-confirmed HySoft reference.

What CAN be validated without a primary-source read is that a reproduced
baseline exhibits the *qualitative* behavior its source paper claims, at
the *exact* network scenarios that paper studies. This module does that
for the DDQN+SOCP baseline (Iqbal et al., 2021) at both scenarios their
paper uses. An equivalent generalization test was attempted for ANN+GSBF
(Fathy et al., 2021) and is deliberately NOT included — see the note below
the DDQN+SOCP test for why. Section 14's risk-mitigation text has been
corrected to describe this scoped-down, partial validation rather than
overclaiming coverage of "each" baseline (see the accompanying edit to
manuscript/MPhil_Thesis_Concept_Note_v4.md).
"""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from training.train_baselines import run_baseline_benchmarks

# Concept Note v4.0 Section 12.2: "Re-run Iqbal's R=5,U=2 and R=12,U=4
# scenarios too, for direct comparability."
IQBAL_SCENARIOS = [(5, 2), (12, 4)]


@pytest.fixture
def base_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _write_scenario_config(base_cfg, n_rrh, n_ue, tmp_path):
    cfg = deepcopy(base_cfg)
    cfg["network"]["n_rrh"] = n_rrh
    cfg["network"]["n_ue"] = n_ue
    cfg_path = tmp_path / f"scenario_r{n_rrh}_u{n_ue}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    return str(cfg_path)


@pytest.mark.parametrize("n_rrh,n_ue", IQBAL_SCENARIOS)
def test_ddqn_socp_reduces_power_vs_all_on_at_iqbal_scenario(
    base_config, tmp_path, n_rrh, n_ue
):
    """Iqbal et al. (2021)'s headline qualitative claim: a trained
    DDQN-based discrete-activation policy (coupled here to a SOCP power
    solver, baselines/ddqn_socp.py) uses meaningfully less power than
    leaving every RRH permanently on, at the paper's own studied network
    sizes. This does not assert a specific power/EE number from the paper
    (unverifiable here — see module docstring); it asserts the directional
    claim the paper makes, reproduced under this project's own environment.
    """
    cfg_path = _write_scenario_config(base_config, n_rrh, n_ue, tmp_path)

    results = run_baseline_benchmarks(
        config_path=cfg_path,
        seeds=[42],
        episodes=40,
        algorithms=["all_on", "ddqn_socp"],
        save_dir=str(tmp_path / "results"),
    )

    all_on_power = results["all_on"][0]["mean_power_w"]
    ddqn_socp_power = results["ddqn_socp"][0]["mean_power_w"]

    assert ddqn_socp_power < all_on_power


# NOTE: an ANN+GSBF generalization test (does the trained ANN predict the
# near-optimal active-RRH fraction on *held-out* scenarios better than a
# naive constant predictor?) was attempted here and deliberately left out.
# It failed: on both config/small_network.yaml (R=5) and config/default.yaml
# (R=12), the exhaustive-search ground-truth labels are heavily skewed
# toward one or two low RRH-count values, and the trained ANN's held-out
# predictions did not beat a naive "always predict the training-set mean"
# baseline on MAE, nor show a statistically significant correlation with
# the true labels (Spearman/Pearson both ~0, p > 0.4). This is a genuine
# finding about the current ANN+GSBF implementation's generalization,
# surfaced during this audit and reported separately rather than silently
# worked around with a weaker assertion.
