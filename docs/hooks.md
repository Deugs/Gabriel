# Hooks: C-RAN DRL Thesis Development Lifecycle

> **Status (corrected, sixth audit round)**: the script is committed and executable at
> `.githooks/pre-commit` (tracked in git, not `.git/hooks/`), but it is **not** active by default in any
> given checkout — `git config core.hooksPath .githooks` must be run once, locally, per clone (this
> setting lives in `.git/config`, which `git clone` never copies, so no checkout has it pre-wired; a
> previous version of this note incorrectly implied it was already active). Run that command once to
> activate it. It runs in lenient mode: each check (black/flake8/mypy/pytest) skips gracefully if there's
> no code or tests yet, or if the tool isn't installed, so it won't block commits during early
> scaffolding. As real implementation and tests land in `cran_env/`, `agents/`, etc., it starts enforcing
> for real.
>
> The remaining hooks (pre-experiment, post-episode, post-training, chapter-completion, pre-submission)
> are Python/bash callbacks meant to be called from inside `training/`, `thesis/hooks/`, and
> `scripts/` once that code exists — they aren't wired to anything yet because the code they hook into
> hasn't been written. Wire them in as you implement the corresponding phase.

## Hook: pre-commit
**Trigger**: Before every git commit
**Purpose**: Ensure code quality and consistency

```bash
#!/bin/bash
# .githooks/pre-commit  (lenient variant actually installed — see status note above)

echo "Running pre-commit checks..."

# 1. Code formatting
black --check cran_env/ agents/ baselines/ training/ evaluation/ || {
    echo "Code not formatted. Run: black cran_env/ agents/ baselines/ training/ evaluation/"
    exit 1
}

# 2. Linting
flake8 cran_env/ agents/ baselines/ training/ evaluation/ --max-line-length=100 || {
    echo "Linting failed. Fix style issues."
    exit 1
}

# 3. Type checking (optional)
mypy cran_env/ agents/ --ignore-missing-imports || {
    echo "Type checking failed."
    exit 1
}

# 4. Run tests
pytest tests/ -x -q || {
    echo "Tests failed. Fix before committing."
    exit 1
}

# 5. Check thesis text consistency
python scripts/check_code_text_consistency.py || {
    echo "Code-text inconsistency detected. Check docs/equation_code_mapping.md"
    exit 1
}

echo "All checks passed."
```

---

## Hook: pre-experiment
**Trigger**: Before starting any training run
**Purpose**: Validate configuration and prevent wasted compute

```python
# training/hooks/pre_experiment.py

def pre_experiment_hook(config):
    """Validate experiment configuration before training."""

    # 1. Check required fields
    required = ["n_rrh", "n_ue", "algorithm", "max_episodes", "random_seed"]
    for field in required:
        assert hasattr(config, field), f"Missing required config: {field}"

    # 2. Validate power parameters against EARTH model
    assert config.p_stat == 175.0, "BBU static power must be 175W (EARTH model)"
    assert config.p_dyn == 250.0, "BBU dynamic power must be 250W (EARTH model)"

    # 3. Check algorithm validity — matches training/train_baselines.py's
    # `algorithms` list plus the proposed method (branching_mp_dqn);
    # hybrid_sac_dqn is the superseded alternative, kept for comparison only.
    valid_algorithms = ["branching_mp_dqn", "hybrid_sac_dqn", "ddpg", "ddqn",
                        "ddqn_socp", "pdqn", "mpdqn", "ann_gsbf",
                        "all_on", "greedy", "nmbs", "convex"]
    assert config.algorithm in valid_algorithms, f"Unknown algorithm: {config.algorithm}"

    # 4. Check for existing results (prevent accidental overwrite)
    result_path = f"data/results/{config.algorithm}_R{config.n_rrh}_U{config.n_ue}_seed{config.random_seed}"
    if os.path.exists(result_path):
        response = input(f"Results exist at {result_path}. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborting experiment.")
            return False

    # 5. Log experiment start
    wandb.init(project="cran-drl-thesis", config=config.__dict__)

    # 6. Save config
    os.makedirs(result_path, exist_ok=True)
    with open(f"{result_path}/config.yaml", 'w') as f:
        yaml.dump(config.__dict__, f)

    print(f"Experiment validated. Starting training: {result_path}")
    return True
```

---

## Hook: post-episode
**Trigger**: After every training episode
**Purpose**: Monitor training health and detect issues early

```python
# training/hooks/post_episode.py

def post_episode_hook(agent, episode, episode_reward, episode_info, config):
    """Monitor training health after each episode."""

    # 1. Check for NaN/Inf in networks
    for name, param in agent.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"NaN/Inf detected in {name} at episode {episode}")

    # 2. Check reward scale
    if abs(episode_reward) > 1e6:
        print(f"WARNING: Extreme reward {episode_reward} at episode {episode}")

    # 3. Monitor Q-value explosion
    if hasattr(agent, 'critic1'):
        # Sample a batch and check Q-values
        # (implementation depends on agent structure)
        pass

    # 4. Early stopping checks
    if episode > 1000 and episode_reward < -1e4:
        print(f"WARNING: Reward collapsed at episode {episode}. Consider restarting.")

    # 5. Save checkpoint periodically
    if episode % config.checkpoint_freq == 0:
        save_checkpoint(agent, episode, config)

    # 6. Log to W&B
    wandb.log({
        "episode": episode,
        "episode_reward": episode_reward,
        **episode_info
    })
```

---

## Hook: post-training
**Trigger**: After training completes
**Purpose**: Generate evaluation artifacts and validate results

```python
# training/hooks/post_training.py

def post_training_hook(agent, config, results_dir):
    """Generate evaluation artifacts after training."""

    # 1. Run evaluation episodes
    eval_results = evaluate_agent(agent, config, n_episodes=100)

    # 2. Generate convergence plot
    plot_convergence(results_dir, [config.algorithm])

    # 3. Generate energy profile
    plot_energy_profile(eval_results, save_path=f"{results_dir}/energy_profile.pdf")

    # 4. Save final model
    torch.save(agent.state_dict(), f"{results_dir}/final_model.pt")

    # 5. Generate summary report
    summary = {
        "algorithm": config.algorithm,
        "n_rrh": config.n_rrh,
        "n_ue": config.n_ue,
        "final_eval_reward": eval_results["mean_reward"],
        "final_eval_energy": eval_results["mean_energy"],
        "qos_violation_rate": eval_results["qos_violation_rate"],
        "training_time_hours": eval_results["training_time"] / 3600,
    }

    with open(f"{results_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Training complete. Summary saved to {results_dir}/summary.json")
    return summary
```

---

## Hook: chapter-completion
**Trigger**: When a chapter is marked complete
**Purpose**: Enforce quality gates before proceeding

```python
# thesis/hooks/chapter_completion.py

CHAPTER_GATES = {
    "ch1": ["background", "problem_statement", "objectives", "scope"],
    "ch2": ["cran_architecture", "traditional_methods", "rl_wireless", 
            "drl_cran", "research_gap"],
    "ch3": ["network_architecture", "channel_model", "traffic_model", 
            "power_model", "mdp_formulation", "problem_formulation", 
            "proposed_algorithm"],
    "ch4": ["simulation_setup", "convergence", "energy_comparison", 
            "qos_performance", "ablation", "scalability"],
    "ch5": ["contributions", "findings", "limitations", "future_work"]
}

def chapter_completion_hook(chapter_num):
    """Validate chapter completeness before marking done."""

    chapter_key = f"ch{chapter_num}"
    required_sections = CHAPTER_GATES.get(chapter_key, [])

    print(f"Validating Chapter {chapter_num}...")

    # 1. Check all required sections exist
    tex_file = f"thesis/chapters/chapter{chapter_num}.tex"
    with open(tex_file, 'r') as f:
        content = f.read()

    missing = []
    for section in required_sections:
        if section not in content.lower():
            missing.append(section)

    if missing:
        print(f"MISSING SECTIONS: {missing}")
        return False

    # 2. Check equation numbering
    equations = re.findall(r'\\begin\{equation\}', content)
    print(f"Found {len(equations)} equations")

    # 3. Check figure references
    figures = re.findall(r'\\label\{fig:[^}]+\}', content)
    figure_refs = re.findall(r'\\ref\{fig:[^}]+\}', content)
    print(f"Figures: {len(figures)}, References: {len(figure_refs)}")

    # 4. Check citations
    citations = re.findall(r'\\cite\{[^}]+\}', content)
    print(f"Citations: {len(citations)}")

    # 5. Word count
    word_count = len(content.split())
    print(f"Word count: {word_count}")

    print(f"Chapter {chapter_num} validation complete.")
    return True
```

---

## Hook: pre-submission
**Trigger**: Before thesis submission
**Purpose**: Final comprehensive check

```bash
#!/bin/bash
# scripts/pre_submission.sh

echo "=== PRE-SUBMISSION CHECKLIST ==="

# 1. Build LaTeX
pdflatex thesis/main.tex
bibtex thesis/main
pdflatex thesis/main.tex
pdflatex thesis/main.tex

# 2. Check for compilation errors
if grep -q "Error" thesis/main.log; then
    echo "LaTeX compilation errors found!"
    exit 1
fi

# 3. Check figure quality
python scripts/check_figure_quality.py || exit 1

# 4. Verify all figures referenced
python scripts/check_figure_references.py || exit 1

# 5. Run plagiarism check
# (University-specific tool)

# 6. Generate final statistics
echo "=== THESIS STATISTICS ==="
echo "Total words: $(python scripts/count_words.py thesis/main.tex)"
echo "Total figures: $(ls thesis/figures/*.pdf | wc -l)"
echo "Total tables: $(grep -c 'begin{table}' thesis/chapters/*.tex)"
echo "Total equations: $(grep -c 'begin{equation}' thesis/chapters/*.tex)"
echo "Total citations: $(grep -o 'cite{[^}]*}' thesis/chapters/*.tex | wc -l)"

echo "=== ALL CHECKS PASSED ==="
```
