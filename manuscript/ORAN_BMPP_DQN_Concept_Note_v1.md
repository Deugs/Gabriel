# Research Concept Document — v1.0

**Topic Title:** BMPP-DQN Energy Optimization in Open Cloud-RAN with Hybrid Discrete-Continuous Control

> **Status note (this repository):** this document governs a separate, additive research track (the O-RAN / BMPP-DQN codebase under `oran_env/`, `oran_agents/`, `oran_training/`, `oran_evaluation/`, `config/oran_default.yaml`) developed for the actual MPhil thesis submission. It is supervisor-approved and does not supersede, modify, or otherwise affect `manuscript/MPhil_Thesis_Concept_Note_v4.md`, which continues to govern the existing, separately-scoped C-RAN / Branching MP-DQN + TD3 codebase (used for publications). The two tracks share no code and are evaluated independently.

## 1. Purpose of the Document

This research concept document is submitted for approval for the Master of Philosophy (MPhil) in Telecommunications Engineering. It outlines a focused research project to develop and validate a reinforcement learning framework — branching Multi-Pass Parameterized Deep Q-Network (BMPP-DQN) — for energy optimization in Open Cloud-Radio Access Networks (O-RAN).

## 2. Background and Problem Statement

### 2.1 Background

The evolution toward 5G/6G has introduced Open Radio Access Networks (O-RAN), which disaggregate the traditional gNB into Radio Units (RUs), Distributed Units (DUs), and Central Units (CUs). While this offers flexibility, it significantly increases energy management complexity. RAN accounts for 70-80% of network energy usage, making energy optimization a critical operational expenditure.

O-RAN enables intelligent control via the RAN Intelligent Controllers (RICs) — Non-RT RIC (1-10s decisions) and Near-RT RIC (10-100ms decisions). Deep Reinforcement Learning (DRL) is a natural fit for this dynamic optimization problem. However, O-RAN control inherently involves a hybrid (parameterized) action space: discrete decisions (e.g., RU on/off, functional split selection) coupled with continuous parameters (e.g., transmit power, PRB allocation fractions).

### 2.2 The Core Technical Challenge

Standard DRL algorithms fail to handle this structure efficiently:

- DQN handles discrete actions only.
- DDPG handles continuous actions only.
- P-DQN (Parameterized DQN) handles hybrid actions but suffers from over-parameterization (Q-network processes all discrete-parameter pairs simultaneously).
- MP-DQN (Multi-Pass DQN) fixes the over-parameterization but uses a flat architecture, leading to action interference.
- Branching DQN elegantly decomposes actions but is limited to discrete spaces.

No existing algorithm combines Branching DQN's action decomposition with MP-DQN's efficient parameterized action handling.

### 2.3 Problem Statement

How can a DRL framework — BMPP-DQN — combine action branching and multi-pass parameterized processing to efficiently optimize energy consumption in O-RAN while managing the hybrid discrete-continuous action space under dynamic traffic conditions?

## 3. Gaps in Existing Papers (Up to 2026)

The literature review reveals three specific, addressable gaps that justify this MPhil research:

### 3.1 Gap 1: OREO's Limitations

The most recent O-RAN DRL framework (OREO) has significant limitations:

- Uses PPO with continuous action spaces, handling discrete RU decisions via thresholding (an approximation, not a principled solution).
- Employs a flat action representation without decomposition, causing interference between decision types.
- Uses static reward weights that cannot adapt to dynamic conditions.
- Optimizes decisions at a single timescale (Non-RT RIC only), ignoring faster Near-RT RIC loops.

**Gap:** No O-RAN energy optimization framework provides a principled, decomposed approach to hybrid action spaces with multi-timescale awareness.

### 3.2 Gap 2: No Integration of Branching and Multi-Pass Architectures

Branching DQN decomposes action spaces into independent branches, reducing output from ∏|A_b| to ∑|A_b|, but handles discrete actions only. MP-DQN efficiently computes Q-values for parameterized actions but uses a flat architecture where all actions are learned jointly, causing interference (e.g., RU activation decisions interfering with power control learning).

**Gap:** No work has combined these two architectures to leverage the decomposition benefits of branching with the parameterized action efficiency of multi-pass processing.

### 3.3 Gap 3: Limited Multi-Timescale Coordination

Existing frameworks treat slow (RU activation, split selection) and fast (power control, scheduling) decisions as independent problems or optimize them at the same frequency, ignoring the natural temporal hierarchy of O-RAN control loops.

**Gap:** No existing work provides a unified learning framework that explicitly models the different timescales of O-RAN decisions while handling their hybrid action spaces.

## 4. Research Questions and Objectives

### 4.1 Research Questions

- **RQ1:** How can branching action decomposition and multi-pass parameterized processing be integrated to efficiently handle O-RAN's hybrid action space?
- **RQ2:** What is the energy-saving performance of BMPP-DQN compared to standard DQN, DDPG, and MP-DQN in an O-RAN simulation environment?
- **RQ3:** How does the multi-timescale design (separate upper/lower networks) affect learning convergence and overall energy efficiency?

### 4.2 Research Objectives

- Design the BMPP-DQN algorithm integrating branching architecture with multi-pass parameterized action handling.
- Formulate the O-RAN energy optimization as a parameterized MDP with 4 action branches.
- Implement a focused O-RAN simulation environment (single-gNB, 3 split options).
- Demonstrate energy savings of ≥15% compared to baseline DRL algorithms.
- Analyze the impact of multi-timescale branch separation on learning convergence.

## 5. Methodological Approach

The methodology is structured into logical phases:

### 5.1 System Modeling

A single-gNB O-RAN system with multiple RUs, one DU, and one CU will be modeled with:

- **Functional Splits:** Simplified to 3 representative options (e.g., Options 2, 6, 8 from 3GPP TR 38.801) to reduce action space complexity.
- **Power Consumption:** RU active/sleep power, DU/CU processing power (based on literature models).
- **Traffic:** Time-varying Poisson arrival with a daily trapezoidal pattern.
- **Wireless:** Simplified SINR with path loss and Rayleigh fading.

### 5.2 Algorithm Design

**BMPP-DQN Architecture.**

**Key Innovation:** Each branch uses multi-pass processing (from MP-DQN) to handle its continuous parameters, while the branching architecture decomposes decisions. This is the core novel contribution.

**Multi-Timescale Handling:** Two Q-networks are used:

- **Upper-level Network:** Trains on 1-10s decisions (RU activation, functional split) using longer replay buffers.
- **Lower-level Network:** Trains on 10-100ms decisions (power, PRB) with faster updates.

**Propagation:** Lower-level performance metrics (average throughput, power consumption) feed into the upper-level state.

### 5.3 Implementation and Training

- **Platform:** Python with PyTorch; Gym-style custom O-RAN environment.
- **Baselines:** Comparison only against DQN (discrete baseline), DDPG (continuous baseline), and MP-DQN (parameterized baseline). OREO will be discussed qualitatively but not reproduced due to complexity.
- **Training:** Single-GPU setup; 3 random seeds for statistical confidence.
- **Evaluation Metrics:** Energy savings (%), throughput (Mbps), convergence speed.

## 6. Scope and Limitations

### 6.1 Scope

- Single-gNB O-RAN simulation (3 functional split options).
- Downlink energy optimization.
- Comparison against 3 baselines: DQN, DDPG, MP-DQN.
- Simulation-based validation.
- Computational analysis limited to convergence time and inference latency profiling.

### 6.2 Limitations (Explicitly Acknowledged)

- No real-world testbed validation.
- Simplified channel and traffic models.
- No multi-gNB or inter-cell coordination.
- Hardware-specific power models not implemented.

## 7. Significance of the Research

- Integration of branching DQN and MP-DQN architectures, contributing to the DRL literature on hybrid action spaces.
- **Focused Contribution:** Provides a clear, incremental advancement over existing parameterized DRL methods.
- **O-RAN Application:** Demonstrates how advanced DRL can address the specific hybrid-action challenges of O-RAN.
- **Energy Savings:** Directly addresses the critical OPEX challenge of RAN energy consumption.

## 8. Timeline of Thesis Completion

| Weeks | Milestone |
|---|---|
| 1-3 | Literature review, gap identification, problem refinement |
| 4-6 | O-RAN system modeling, BMPP-DQN algorithm design, branch architecture definition |
| 7-9 | Simulation environment development in Python/PyTorch, BMPP-DQN implementation |
| 10-13 | Training experiments, baseline comparisons, data collection |
| 14-16 | Results analysis, thesis writing, final submission and defense |

## 9. References

[1] Qazzaz, M. M. H., Salama, A., Hafeez, M., & Zaidi, S. A. R. "OREO: Open RAN Energy Optimization via Deep Reinforcement Learning for 6G Networks," IEEE Open Journal of the Communications Society, vol. 7, pp. 4165-4182, 2026.

[2] H. Shengren, E. M. Salazar Duque, P. P. Vergara, and P. Palensky, "Performance comparison of deep RL algorithms for energy systems optimal scheduling," in 2022 IEEE PES Innovative Smart Grid Technologies Conference Europe (ISGT-Europe), IEEE, 2022, pp. 1-6, doi: 10.1109/ISGT-Europe54678.2022.9960642.

[3] A. Iqbal, M.-L. Tham, and Y. C. Chang, "Double Deep Q-Network-Based Energy-Efficient Resource Allocation in Cloud Radio Access Network," IEEE Access, vol. 9, pp. 20440-[end page not confirmed], 2021, doi: 10.1109/ACCESS.2021.3054909.

[4] M. Fathy, M. S. Abood, and M. M. Hamdi, "Optimization of Energy-Efficient Cloud Radio Access Networks for 5G using Neural Networks," in 2021 International Conference on Intelligent Technology, System and Service for Internet of Everything (ITSS-IoE), 2021, doi: 10.1109/ITSSIoE53029.2021.9615290.

[5] W. H. A. Al-Zubaedi, "Planning a C-RAN Deployment for the Next Generation Cellular Networks," Ph.D. dissertation, Dept. of Electronic and Computer Engineering, Brunel University London, London, U.K., 2019. Supervisors: H. Al-Raweshidy, A. Zobaa. Available: http://bura.brunel.ac.uk/handle/2438/17865

[6] 3GPP, "Study on channel model for frequencies from 0.5 to 100 GHz," TR 38.901, V16.1.0, 2020.

[7] IEEE, "Self-Optimized Agent for Load Balancing and Energy Efficiency: A Reinforcement Learning Framework With Hybrid Action Space," IEEE Trans. on Network and Service Management, vol. 21, no. 4, pp. 4902-4919, Jul. 2024.

[8] S. Sharma and W. Yoon, "Energy Efficient Power Allocation in Massive MIMO Based on Parameterized Deep DQN," Electronics, vol. 12, no. 21, p. 4517, 2023.

[9] H. Kabir, M.-L. Tham, and Y. C. Chang, "Mobility-Aware Resource Allocation in IoT Network for Post-Disaster Communications with Parameterized Reinforcement Learning," Sensors, vol. 23, no. 14, p. 6448, Jul. 2023.

[10] J. Lu, P. Yan, and H. Zeng, "EExApp: GNN-Based Reinforcement Learning for Radio Unit Energy Optimization in 5G O-RAN," in Proc. IEEE INFOCOM, 2026, pp. 1-10.

[11] C. Yan et al., "Hybrid Reinforcement Learning in parameterized action space via fluctuates constraint," Engineering Applications of Artificial Intelligence, vol. 162, p. 112499, 2025.

[12] S. Hou, E. M. Salazar Duque, P. P. Vergara, and P. Palensky, "Deep Reinforcement Learning for Energy Management in Microgrids," in Proc. ISGT-Europe, 2022, pp. 1-6, doi: 10.1109/ISGTEurope54678.2022.9960642.

[13] "Hybrid Cognitive IoT with Cooperative Caching and SWIPT-EH: A Hierarchical Reinforcement Learning Framework," IEEE Internet of Things Journal, Early Access, 2025.

[14] F. Rezazadeh, H. Chergui, L. Christofi, and C. Verikoukis, "Actor-Critic-Based Learning for Zero-touch Joint Resource and Energy Control in Network Slicing," arXiv preprint, 2022.

[15] O-RAN Alliance, "O-RAN Minimum Viable Plan and Commercialization Strategy," June 2021.

[16] M. Bordin et al., "Design and Evaluation of Deep Reinforcement Learning for Energy Saving in Open RAN," arXiv: 2410.14021, Oct. 2024.

[17] Esmaeil Amiri, Ning Wang, Mohammad Shojafar, Rahim Tafazolli, "Energy-Aware Dynamic VNF Splitting in O-RAN Using Deep Reinforcement Learning," LCN 2022, pp. 422-429. Available: https://dblp.org/pid/06/8812.

[18] Ryan Barker, Tolunay Seyfi, Fatemeh Afghah, "Advancements in Mobile Edge Computing and Open RAN: Leveraging Artificial Intelligence and Machine Learning for Wireless Systems," arXivlens, 2025.

[19] X. Liang, A. Al-Tahmeesschi, S. B. Chetty, and H. Ahmadi, "Green O-RAN Operation: a Modern ML-Driven Network Energy Consumption Optimization," GLOBECOM 2025, pp. 4475-4480.

[20] M. Bordin, A. Lacava, M. Polese, S. Satish, M. AnanthaSwamy Nittoor, R. Sivaraj, F. Cuomo, and T. Melodia, "Design and Evaluation of Deep Reinforcement Learning for Energy Saving in Open RAN," 2025 IEEE 22nd Consumer Communications & Networking Conference (CCNC), Las Vegas, NV, USA, pp. 1-6, January 2025.

[21] Narjes Lassoued and Noureddine Boujnah, "A Comprehensive Review of Energy Efficiency in 5G Networks: Past Strategies, Present Advances, and Future Research Directions," https://www.mdpi.com/2073-431X/15/1/50.

---

## 10. Implementation Addendum (this repository, resolves design ambiguities left open by the source document above)

The source document specifies the research design at the concept-note level; the items below resolve concrete implementation details needed to write code, following the same "needs validation" flagging convention `MPhil_Thesis_Concept_Note_v4.md` uses for its own literature-sourced placeholders.

### 10.1 The "4 action branches" (§4.2 Objectives)

1. **RU activation** — discrete, per-RU on/off, upper (slow) timescale.
2. **Functional split selection** — discrete, per-RU choice of one of the 3 representative options (§5.1), upper (slow) timescale.
3. **Transmit power** — continuous, per-RU, lower (fast) timescale.
4. **PRB allocation fraction** — continuous, per-RU, lower (fast) timescale.

This reads directly off §5.1/5.2's own text (RU on/off + split selection = discrete; power + PRB = continuous), and the discrete/continuous split maps directly onto the upper/lower timescale split with no separate assignment decision needed.

**Split selection is per-RU, not one global gNB-wide choice** (needs validation as a design choice, not a literature fact) — this preserves symmetry with per-RU activation and gives each RU (potentially at a different fronthaul distance/cost) its own trade-off to learn.

### 10.2 Functional split → centralization mapping (needs validation)

3GPP TR 38.801 split numbering is not monotonic in "how centralized" the processing is. For simulation tractability, this implementation defines a monotonic centralization level `c ∈ {0, 1, 2}`:

| `c` | 3GPP Option | RU-side processing | Fronthaul bandwidth need |
|---|---|---|---|
| 0 | Option 2 | Most | Least |
| 1 | Option 6 | Intermediate | Intermediate |
| 2 | Option 8 | Least (CPRI-like) | Most |

This is a modeling simplification for a monotonic energy/bandwidth trade-off, not an assertion of physical accuracy — **flagged needs validation** against the split-specific literature before being stated as fact in the thesis text.

**2026-08-29 literature check**: the O-RAN Alliance's own "Minimum Viable Plan and Acceleration towards Commercialization" white paper (June 2021) states that the actual specified Open Fronthaul interface (O-DU to O-RU) is Lower Layer Split (LLS) **Option 7-2x**, not literally Option 2/6/8 — the real O-RAN Alliance architecture uses one specific split, not this document's 3-level Option 2/6/8 abstraction. This doesn't change the modeling choice (a 3-level monotonic abstraction is still needed for simulation tractability, per this section's own reasoning), but it is a concrete data point that the thesis text should acknowledge rather than imply Option 2/6/8 is how real O-RAN deployments actually work.

**2026-08-30 literature check**: Rony et al. 2021 (IEEE Access) independently corroborates the *direction* of this monotonic mapping, using their own PHY-layer split taxonomy (Split-A through Split-D): their most-centralized split ("almost no processing at RRH") requires the most fronthaul capacity, and their least-centralized split ("all PHY layer processing... performed at RRH") requires the least — the same RU-processing-vs-fronthaul-need trade-off direction as this table's Option 2→Option 8 progression. Their evidence is CAPEX/OPEX cost-percentage weights, not fronthaul bandwidth or power figures, so it supports only the qualitative shape of this mapping, not any numeric parameter derived from it.

**2026-08-30 literature check, part 2**: a HUBER+SUHNER/CubeOptics infographic ("5G Fundamentals: Functional Split Overview," reproducing 3GPP TR 38.801's own split taxonomy) gives real, quantitative per-split fronthaul bandwidth requirements for a 100 MHz carrier, precisely pixel-verified against the infographic's column alignment (not hand-read from linearized PDF text, which initially scrambled the option-to-value mapping):

| Option | 1 | 2 | 3 | 4 | 5 | 6 | 7-3 (DL only) | 7-2 | 7-1 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| UL (Gbps) | 3 | 3 | 3 | 4.5 | 7.1 | 7.1 | 15.2 | 15.2 | 60.4 | 157.3 |
| DL (Gbps) | 4 | 4 | 4 | 5.2 | 5.6 | 5.6 | 9.8 | 9.8 | 9.2 | 157.3 |

(Options 1-6 on an 8-layer/256QAM reference basis; Options 7-3 through 8 on a 32-antenna reference basis — the two reference conditions differ, so the two halves of the table are not on a strictly uniform basis.) For the three options this document's mapping actually uses: Option 2 = 3/4 Gbps, Option 6 = 7.1/5.6 Gbps, Option 8 = 157.3/157.3 Gbps — a real, quantitative confirmation of the monotonic direction this mapping assumes. It also reveals that this model's own `p_fh_per_ru_by_split` array (§10.5, ratio 1:2:5 across c=0/1/2) is far shallower than the real bandwidth-requirement ratio these figures imply (roughly 1:1.4-2.4:39-52 depending on UL/DL) — fronthaul *power* is not established to scale linearly with fronthaul *bandwidth* requirement, so this was not used to rescale `p_fh_per_ru_by_split`, but it is a disclosable, quantified version of the "shape confirmed, magnitude unconfirmed" gap already on record for that constant (see §10.5).

**2026-08-30 literature check, part 3**: obtained 3GPP TR 38.801 itself (V0.4.0, 2016-08, Release 14 — the actual primary document behind both the Option 2/6/8 definitions and the bandwidth figures cited above, not a secondary reproduction). §6.1.2.2 confirms this document's own characterization of each option is accurate: Option 2 ("PDCP/RLC, 3C-like split") — "RRC, PDCP are in the central unit. RLC, MAC, physical layer and RF are in the distributed unit"; Option 6 ("MAC-PHY split") — "Physical layer and RF are in the distributed unit. Upper layers are in the central unit"; Option 8 ("PHY-RF split") — "RF functionality is in the distributed unit and upper layer are in the central unit." Annex A's Table A-1 ("Requirements on the underlying transport network due to a certain functional split") gives the *exact*, non-rounded primary-source bandwidth figures — which exactly cross-validate the pixel-verified HUBER+SUHNER infographic reading above:

| Option | 1 | 2 | 3 | 4 | 5 | 6 | 7a | 7b | 7c | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| DL | 4 Gb/s | 4016 Mb/s | lower than Option 2 | 5226.7 Mb/s | 5626.7 Mb/s | 5626.7 Mb/s | 9.8 Gb/s | 9.2 Gb/s | 9.8 Gb/s | 157.3 Gb/s |
| UL | 3 Gb/s | 3024 Mb/s | lower than Option 2 | 4500 Mb/s | 7140 Mb/s | 7140 Mb/s | 15.2 Gb/s | 60.4 Gb/s | 60.4 Gb/s | 157.3 Gb/s |
| Max one-way latency | 10 ms | 1.5-10 ms | 1.5-10 ms | ~100 µs | hundreds of µs | 250 µs | 250 µs | 250 µs | 250 µs | 250 µs |

(Table's own note: "values are examples provided by LTE reference... to be replaced by NR values when available.") For this document's three mapped options: Option 2 = 4016/3024 Mb/s ≈ 4.0/3.0 Gbps, Option 6 = 5626.7/7140 Mb/s ≈ 5.6/7.1 Gbps, Option 8 = 157.3/157.3 Gbps — an **exact** match to the HUBER+SUHNER infographic's pixel-verified figures above (a nice independent confirmation that the pixel-verification methodology was correct), now cited from the primary 3GPP document rather than a secondary reproduction. The table's latency column is new context: Option 2's 1.5-10 ms tolerance and Option 6/8's 250 µs requirement are both far below this model's `step_duration_s=0.1 s` (100 ms) Near-RT RIC decision rate, consistent with that rate being a coarser RL-decision-loop period rather than a raw fronthaul HARQ-level latency constraint — no change needed, but a useful sanity check newly available from a primary source.

### 10.3 Default scenario (needs validation on exact counts)

`n_ru=4, n_du=1, n_cu=1, n_ue=8, n_splits=3` — small enough that the flat MP-DQN baseline's joint action space (`2^4 * 3^4 = 1296`) stays tractable, consistent with §6.1's "focused... single-gNB" scope.

**2026-08-30 literature check**: no new sources were supplied for this flag, so the 8 O-RAN-context PDFs already on hand (from the §10.2/§10.5/§10.6 checks) were re-mined for scenario-scale content instead. Two directly comparable RAN-DRL papers give concrete scenario scales: Eskandarinia et al.'s DQRL paper fixes 12 RUs and varies UE count at 12/14/16 (UE:RU ratio ~1.0-1.3), and Qazzaz et al. 2026 (OREO) evaluates "a single deployment scenario with 42 RUs and 100 UEs" (UE:RU ratio ~2.4). This repo's own UE:RU ratio (8/4 = 2) sits inside that same range, though its *absolute* RU count (4) is markedly smaller than either paper's (12, 42) -- a real difference, not a validated match, since neither paper's ratio or scale was chosen to inform this repo's and the comparison is coincidental. More useful than the ratio: OREO's own discussion section states its single-agent centralised formulation "may... face scalability challenges as the number of RUs grows substantially" -- independent, literature-stated corroboration that keeping RU count small is a real concern for this class of algorithm (single-agent/centralized RL, as BMPP-DQN is), not merely an ad hoc convenience. The O-RAN Alliance's own 2021 MVP white paper separately describes "Deployment Scenario E.1" as a single cloudified O-DU associated with "several" non-virtualized O-RUs at a cell site -- qualitatively consistent with this repo's single-DU/n_ru=4 structure, though "several" is not a specific count. None of this validates `n_ru=4`/`n_ue=8` as the "correct" numbers -- it remains a tractability-driven choice -- but it does show the choice sits within precedented ranges and that the underlying tractability rationale is independently echoed in directly comparable literature, not just an internal justification.

### 10.4 No TD3 / twin-critic machinery

Unlike `MPhil_Thesis_Concept_Note_v4.md`'s Branching MP-DQN + TD3 (which explicitly adds twin critics and target-policy smoothing), this document's own description of BMPP-DQN's "core novel contribution" (§5.2) mentions only branching decomposition + multi-pass parameterized processing — never twin critics. The implementation therefore uses a single critic, standard Double-DQN target computation, and no target-policy-smoothing noise or delayed-policy-update gating.

**Correction:** §5.2 describes "two Q-networks," one per timescale. The actual implementation has exactly **one** critic (`BranchingCritic`, producing both the activation and split branch-groups' Q-values), fed by `upper_encoder`. The continuous side (`ContinuousParameterNetwork`, fed by `lower_encoder`) is a deterministic-policy-gradient actor trained by maximizing this same shared critic's output — not an independently-trained second Q-network. `update_lower()`/`update_upper()` are both invoked every environment step in `oran_training/train_bmpp_dqn.py`; the two-timescale separation is realized in decision cadence (discrete choices held for `upper_level_period_steps` steps) and buffer refill rate (the upper buffer only gains a new transition every N steps), not in a differential gradient-step frequency. See `docs/skills/skill_oran_bmpp_dqn.md` for the full description of what's actually implemented.

### 10.5 RU/DU/CU/Fronthaul power model constants (needs validation)

`oran_env/power_model.py`'s per-split RU processing power, DU per-RU processing power, CU dynamic power, fronthaul per-RU power, and switching costs (`config/oran_default.yaml`'s `power:` section) are literature-style placeholders, chosen only to keep the split centralization level `c` (§10.2) monotonically trading off RU-side vs. DU/fronthaul-side power — not verified against O-RAN Alliance/ETSI power-model literature. Resolve/cite before the thesis states any of these as fact (mirrors the `docs/oran_thesis_guide.md`'s "Needs-Validation Flags" checklist).

**2026-08-29 literature check**: five O-RAN-context sources were checked (Qazzaz et al. 2026 OREO; Barker/Seyfi/Afghah 2025 MEC/Open RAN survey; Lassoued & Boujnah 2026 Computers 5G energy-efficiency review; Eskandarinia et al.'s DQRL clustered-RAN paper; the O-RAN Alliance's 2021 MVP white paper) — none gives a split-level RU/DU/CU/fronthaul wattage table matching this model's parameterization. See `oran_env/power_model.py`'s own updated docstring for what partial, order-of-magnitude support does exist (the general EARTH-style RU active/sleep model, a comparable small-cell RU power scale and per-RU activation-cost concept in the DQRL paper) versus what remains genuinely unvalidated (the specific per-split DU/CU/fronthaul breakdown — OREO's own energy model explicitly excludes these components from its scope, suggesting a validated source may not yet exist in the RL literature and would need O-RAN Alliance/vendor hardware measurements instead).

**2026-08-30 literature check**: three more sources checked (the Open RAN Handbook 2nd Edition, Vodafone + Keysight, Feb 2025; a Hoffmann/Dryjanski/Kliks Rimedo Labs/i4y Lab E2E energy-testing-framework presentation; Rony et al. 2021's PHY-split cost analysis, also cited in §10.2 above) — still no matching wattage table. What is new: both the Handbook and the presentation report *real measured* hardware power figures (a macro-cell Fujitsu O-RU at roughly 200-550 W; an enterprise server used as an O-DU/O-CU compute-host proxy at roughly 625-780 W), and both are 20-100x larger than this model's own RU/DU/CU placeholder scale. Neither source specifies what scale a small, `n_ru=4`-style simulation scenario should assume, so this is disclosed as a genuine scale mismatch between the placeholders and real macro-cell/enterprise-hardware deployments — not resolved by an invented rescaling. Both sources also independently reconfirm (beyond the OREO footnote already cited) that no O-RAN Alliance normative power-measurement framework yet exists, and point to the real standardized test methodologies (ETSI ES 202 706, ETSI TS 103 786, 3GPP TR 38.864) this model's own linear/step form only loosely resembles. See `oran_env/power_model.py`'s docstring for the full writeup.

**2026-08-30 literature check, part 2**: obtained 3GPP TR 38.864 itself ("Study on network energy savings for NR," the actual document behind the TR 38.864 citation above, not just a secondary reference to it). §5.1 defines the real 3GPP NR BS power-consumption model: `P_DL = P_static,DL + P_dynamic,DL` and `P_UL = P_static,UL + P_dynamic,UL`, with the dynamic term scaled by the fraction of active TRX/RUs, the RF-to-system bandwidth ratio, and the PSD ratio — independently confirming (from the actual governing 3GPP source, not a secondary citation) that a static + scaled-dynamic linear-style structure is the right family of model, the same family as the EARTH model already used for the C-RAN track. Table 5.1-3 gives concrete relative-power ratios (not absolute Watts) across sleep/active states for two "BS Category" classes and three reference configurations — e.g. BS Category 2/Set 1: Deep sleep=1, Micro sleep=5.5, Active DL=32 — order-of-magnitude comparable to (though not identical to) this model's own active:sleep ratio. Crucially, TR 38.864's model is **whole-BS**, not disaggregated into O-RAN's RU/DU/CU/fronthaul components, and its Table 5.1-3 values are relative units with no stated absolute-Watt anchor — converting them to Watts for this model would require inventing a scale factor, which was not done. No constant in `oran_env/power_model.py` was changed based on this source; it is cited as additional structural corroboration only. See `oran_env/power_model.py`'s docstring for the full writeup.

**2026-08-30 literature check, part 3 (one constant actually changed)**: obtained a real small-cell O-RU vendor datasheet — Benetel's RAN550 (n78/n79, Split 7.2x indoor O-RU). Its "Typical power consumption: 40 W" is a genuine small-cell-class (not macro, not enterprise-server) reference figure, 5-14x this model's own composite active-RU power estimate — narrowing, not resolving, the scale mismatch found in part 1 above, since the datasheet doesn't decompose 40 W into processing/RF/fronthaul-interface shares the way this model's constants require. Its "Maximum TX output power (total EIRP): 2 W" (33 dBm), however, *is* a clean same-quantity match for `oran_env/oran_env.py`'s `p_max_dbm` (the RU transmit-power action-space ceiling) — updated from 30 dBm (1 W, an unvalidated placeholder) to 33 dBm (2 W), citing this datasheet directly; this is not a guess, it's the same physical quantity in the same units from a real commercial product. Separately, this model's own `carrier_freq_ghz=3.5` (§10.3) already falls within RAN550's real n78 band (3.3-3.8 GHz) — existing corroboration, no change needed there. See `oran_env/power_model.py`'s docstring and §10.2's "part 2" note above (the fronthaul-bandwidth table, from a companion source supplied the same day) for the full writeup.

**2026-08-30 literature check, part 4**: obtained Al-Tahmeesschi et al. 2025 ("Enhancing Open RAN Digital Twin Through Power Consumption Measurement," arXiv:2507.00928) — the first source found across either literature-check round that gives real, *component-decomposed* (RU vs. DU vs. CU, not whole-BS) O-RAN power measurements, on real O-RAN testbeds including the same Benetel RAN550 already cited above. Their **Split 8** testbed is an *exact, unambiguous match* to this model's `c=2` (both are literally "Option 8, PHY-RF split"): measured RU power (a USRP) is ~43-45 W, essentially load-independent across 0-100% PRB utilization; combined DU+CU power (a single shared Dell PowerEdge server) is ~119.5-141.6 W across the same load range. This model's own composite `c=2` RU estimate (~11 W, using `p_ru_proc_by_split[2]=3.0` + max-TX/eta) is now only ~4x below this real measurement — the closest gap found across any source in either literature-check round (was 20-100x against the earlier macro-cell figures, 5-14x against the RAN550 datasheet's own "typical" figure). Still not a clean match, and still not decomposable into this model's separate processing/RF terms without guessing that split, but the closest, most directly-comparable (real O-RAN hardware, correct split option) RU anchor found yet. Their **Split 7.2b** testbed (RAN550 indoor RU, RAN650 outdoor RU, on two *separate* dedicated servers for DU and CU) is not one of this model's three mapped options, and its DU (~187-194 W) and CU (~189.6-192.7 W) figures must **not** be compared directly against Split 8's combined DU+CU figure to infer a split-dependent power difference: the two testbeds use different hardware (one shared server vs. two dedicated servers), so most of the gap reflects hardware choice, not split-driven processing cost — flagged explicitly to avoid a false quantitative conclusion. Two more findings, neither used to change a constant: (1) the paper's own conclusion — "power consumption does not scale significantly with network load... a large portion of energy consumption remains constant regardless of traffic demand" — independently corroborates this model's existing structural choice that `compute_du_power`/`compute_cu_power` depend on active-RU-count and split choice, not on instantaneous PRB/throughput load (no design change needed, cited as validation); (2) the RAN550's *measured* Split-7.2b RU power here (~28.3-30.1 W) is somewhat lower than its own datasheet's "typical power consumption: 40 W" claim used for the `p_max_dbm` fix above — a real-vs-nominal discrepancy worth noting, though it doesn't affect that fix (a different physical quantity, max TX power, not typical total consumption).

### 10.6 Traffic model trapezoidal breakpoints and Poisson rate (partially resolved)

`oran_env/traffic_model.py`'s trapezoid breakpoints (`t1`-`t4`), peak/floor Poisson arrival rates (`lambda_peak`, `floor_ratio`), and packet size (`config/oran_default.yaml`'s `traffic:` section) were literature-style placeholders chosen only to produce a plausible diurnal demand shape. As of the 2026-08-30 checks below, `lambda_peak` and `packet_size_bits` are now derived directly from a primary 3GPP source (not verified/matched, *derived* — see part 2); `floor_ratio` and `t1`-`t4` remain unverified placeholders (mirrors `docs/oran_thesis_guide.md`'s "Needs-Validation Flags" checklist).

**2026-08-30 literature check**: re-examined all 8 O-RAN-context sources already supplied for the power-model checks above (§10.5), searching specifically for traffic-shape content rather than power figures. One genuinely relevant result: Lassoued & Boujnah 2026 (Computers, the same 5G energy-efficiency review already cited in §10.5) Figure 7, "Daily traffic load variations during a 24 h weekday" (itself citing an external source), shows the same *qualitative* diurnal shape this model assumes -- a near-zero floor roughly 00:00-06:00, a rise through the morning, a noisy but sustained daytime peak, and an evening decline -- with rough (not exact) timing consistent with this model's own `t1=7` (rise start) and `t4=23` (floor reached); the figure's decline appears to start closer to ~18:00 than this model's `t3=20`, and its "peak" is noisy/bimodal (50-90% occupation) rather than a flat plateau. This figure reports a generic macro-cellular network's *relative occupation rate* (%), not a 5G/O-RAN small-cell Poisson arrival rate or bps demand, so it corroborates only the general diurnal shape and rough breakpoint timing -- not `lambda_peak`, `floor_ratio`, `packet_size_bits`, or the exact `t1`-`t4` values. Separately, at the time of this first pass, a temporal Poisson-arrival design appeared not to be precedented in the sources checked (OREO's and the MEC-survey citation's Poisson processes are both spatial-only) -- **superseded by part 2 below**.

**2026-08-30 literature check, part 2**: obtained 3GPP TR 38.864 itself. Its Annex A defines the real, standard 3GPP traffic models used for NR system-level simulation: FTP Model 3 (0.5 MB packet/file size, 200 ms mean inter-arrival time), FTP Model 3 IM (0.1 MB, 2 s mean inter-arrival), and VoIP. FTP Model 3's file arrivals are themselves a per-UE Poisson process -- this *does* directly precedent this model's temporal-Poisson-arrival design, correcting the first pass's finding above (that source was less directly relevant than this one). `lambda_peak` and `packet_size_bits` have been updated to FTP Model 3's own numbers: 200 ms mean inter-arrival = 5 arrivals/s = 0.5 per this model's 0.1 s step (`lambda_peak=0.5`, was 5.0); 0.5 MB = 4e6 bits (`packet_size_bits=4.0e6`, was 1.0e6). This is a genuine primary-source-derived value, not a guess, though two caveats apply: (1) 3GPP leaves the choice among FTP3/FTP3-IM/VoIP to the evaluating party -- FTP3 was adopted here as the most commonly used baseline in 3GPP energy-saving evaluations, a defensible but not the only possible choice; (2) `floor_ratio` and `t1`-`t4` remain unvalidated, since TR 38.864 Annex A's own "load (L)%" scenarios (Table A-1: idle=0%, low<=15%, light<=30%, medium<=50%) are instantaneous PRB-utilization snapshots with no time-of-day association, and the TR's own scope explicitly stops at "medium load" ("The study prioritizes idle/empty and low/medium load scenarios") with no busy-hour/full-load reference point to derive a floor:peak ratio or diurnal timing from. See `oran_env/traffic_model.py`'s docstring for the full writeup, including a noted (not implemented) option of using FTP3-IM's lighter traffic to inform off-peak behavior in a future model revision.
