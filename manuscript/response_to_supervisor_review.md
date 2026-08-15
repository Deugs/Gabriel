RESPONSE TO SUPERVISOR REVIEW

Re: MPhil Thesis Concept Note — consolidated review (Overall Assessment, Methodology Assessment, Critical Gaps G1–G14, Scientific Relevance and Timeliness, Recommendations B1–B4 / S1–S6 / A1–A6)

Candidate: Gabriel Kwame Freeman (Index No. PG7373923)
Degree: MPhil · Institution: KNUST
Supervisor: Prof. J. J. Kponyo
Date: 05 August 2026
Current reference document: `manuscript/MPhil_Thesis_Concept_Note_v4.md` (v4.0)

---

Dear Prof. Kponyo,

Thank you for the detailed review. It identified real gaps in the original hybrid SAC-DDQN proposal — the unspecified critic architecture, the unaddressed combinatorial action space, the thin literature review, the missing O-RAN context, and the unrealistic timeline chief among them — and I have worked through every item since. This note summarizes where things stand: what has been resolved in the concept note itself, what has since been built and tested in code, and what honestly remains outstanding before Chapter 4 can be written.

## 1. How the review's items were resolved in the concept note

Every item in your letter — B1–B4, S1–S6, A1–A6, and the fourteen critical gaps G1–G14 — is addressed in Concept Note v4.0, which supersedes the v1.0 document you originally reviewed. In summary:

- **The architecture question (B2, B3)** is resolved by abandoning the hybrid SAC-DDQN design entirely, not patching it. The proposed method is now a **branching, multi-pass, twin-critic parameterized DQN** (Section 10): R independent per-RRH dueling heads (2R outputs, not 2^R), each coupled to its own continuous power/bandwidth parameters via MP-DQN's multi-pass masking (removing the false-gradient cross-talk P-DQN's original design would introduce), with TD3-style twin critics for overestimation control. There is no more ambiguity about how a hybrid critic ingests a mixed action — Section 10.3 specifies it concretely, with a diagram.
- **The novelty claim (B1, G1–G3)** has been rewritten around the actual literature: the bibliography grew from 4 to 34 references (30 at the time of the v3.0 revision this section originally described, with a further 4 added since — see Concept Note v4.0 §17), engaging directly with the P-DQN-through-HySoft parameterized-action-space lineage, the O-RAN DRL energy literature (OREO, ES-xApp, federated TD3, EExApp), and the 2025 hybrid A3C-Dueling-DQN C-RAN paper you flagged. EExApp is now named explicitly as the closest published related work, and the contribution is argued against it directly (Section 4.4) rather than asserted in the abstract.
- **O-RAN relevance (S1)** is addressed by framing the policy as an rApp — discrete RRH-activation decisions via O1, continuous power/bandwidth decisions via E2 (Section 11) — an honest simulation-based framing, not a claim of O-RAN-interface implementation.
- **Baselines and statistical rigor (S2, S4)**: the baseline suite now has ten methods, adding P-DQN and MP-DQN specifically to isolate branching's contribution; seeds increased from 5 to 10 with Cohen's d reported alongside every p-value.
- **The perfect-CSI limitation (S3) and generalization (A5)** are addressed with bounded, evaluation-only experiments — a CSI-noise degradation curve and a zero-shot cross-profile (weekday/urban → weekend/suburban) evaluation — rather than a training-time scope expansion.
- **The timeline (B4)** is now a week-by-week Gantt chart extending to roughly 27 weeks from the original 6, with R=50 explicitly demoted to a stretch goal rather than a committed deliverable.

The full item-by-item mapping, including the handful of genuinely new items from your detailed review round (G4, G5, G9, G10), is in Concept Note v4.0 Sections 0 and 0.1, and cross-referenced in `docs/supervisor_feedback_log.md`.

## 2. What has since been implemented and tested in code

Having the concept note say a baseline or evaluation "will be added" is not the same as it existing. Over the past several days I audited the codebase against v4.0's claims and closed every gap that turned up:

- **`agents/pdqn_agent.py`, `agents/mpdqn_agent.py`, `agents/ddpg_agent.py`** — the P-DQN, MP-DQN, and pure-DDPG baselines were documented as done but did not exist. All three are now implemented and unit-tested, including a dedicated test confirming MP-DQN's multi-pass masking genuinely zeroes out irrelevant RRH parameters rather than just claiming to.
- **`evaluation/csi_robustness.py`, `evaluation/generalization.py`, `evaluation/latency_benchmark.py`** — the CSI-robustness curve, cross-profile generalization result, and inference-latency benchmark (at R=5,12,20,35,50) now have working, tested implementations.
- **`evaluation/convergence.py`** — Cohen's d is now computed alongside every paired t-test. Fixing this also surfaced a real bug: the code was still comparing against the superseded "Hybrid_SAC_DDQN" label instead of the actual proposed method's saved name, which had been silently excluding it from every statistical comparison.
- **`training/hyperparam_search.py::run_proxy_sensitivity_sweep`** — Section 12.11's specific hyperparameter sensitivity check (R=5, U=2, 100 episodes, 2 seeds, sweeping the branch/continuous-net learning rates and τ by half an order of magnitude) is now implemented, having previously existed only as a documented commitment with no matching code.

All of this is covered by 45 passing unit tests, and the status trackers (`AGENTS.md`, `docs/workflow.md`, `README.md`) have been synced to reflect actual file existence rather than aspirational planning.

## 3. What remains outstanding — stated plainly

- **No full-scale experiment has been run.** Every item in Section 2 above is verified correct at small network sizes and short training runs only. The real 10-seed × 11-method comparison, the CSI-robustness and generalization curves at full scale, the R=5–50 latency sweep, and the Section 12.11 sensitivity sweep (with its keep/change decision on the default learning rates and τ) all still need to be executed.
- **Chapters 3, 4, and 5 of the thesis are not yet written.** Chapter 3 needs the formal MDP writeup from Section 10.2; Chapters 4 and 5 depend on the experiments above.
- **The HySoft (2025) reference** is corroborated through independent secondary evidence but not yet confirmed against the primary ScienceDirect source, which requires institutional library access I don't have in this environment — I will complete this check directly through KNUST's library before the bibliography is finalized.

## 4. What I am asking of you

1. **Sign-off on Concept Note v4.0** as the governing document, so I can treat the architecture, baseline suite, and evaluation plan as settled and move to running the actual experiments.
2. **Confirmation that the current single-agent, simulation-only, O-RAN-framed-but-not-O-RAN-implemented scope is still acceptable**, given that is unchanged from what v3.0 already proposed and you did not flag it again in the detailed review.
3. Any remaining concerns before I begin the full 10-seed experiment matrix — I would rather find out now than after several days of compute.

I'm confident the architecture and evaluation plan are now fully specified and implemented; what's left is running the experiments and writing them up. I'm happy to walk through any part of this in more detail at your convenience.

Regards,
Gabriel Kwame Freeman
