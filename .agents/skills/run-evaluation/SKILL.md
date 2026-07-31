---
name: run-evaluation
description: Run convergence, energy-efficiency, ablation, scalability, and statistical-significance analysis, and generate the publication-quality figures/tables required by Chapter 4. Use when analyzing experiment results or producing thesis figures.
---

Follow the full protocol in [docs/skills/skill_evaluation.md](../../../docs/skills/skill_evaluation.md) — it defines the convergence/energy/ablation/scalability analysis functions, the statistical-significance test (paired t-test + Cohen's d), the shared `COLORS` scheme, and the LaTeX figure-export settings.

Steps:
1. Read `docs/skills/skill_evaluation.md` in full before running or writing any analysis code.
2. Pull results from `data/results/` — never re-run training as part of evaluation; if a required run is missing, say so rather than fabricating numbers.
3. Use the required outputs table in the spec (and `docs/thesis_guide.md` Ch.4) to know exactly which figure/table maps to which thesis section — e.g. `thesis/figures/convergence.pdf` → Section 4.2.
4. Every comparison must include ≥5 seeds with 95% confidence intervals, and every head-to-head claim (proposed vs. baseline) needs the paired-t-test + Cohen's d significance check, per `docs/rules.md`'s Statistical Significance quality gate.
5. Save figures as vector PDFs into `thesis/figures/` and tables as `.tex` booktabs snippets — follow the Figure Standards checklist in the spec (consistent colors, ≥8pt fonts, self-contained captions, error bars everywhere).
