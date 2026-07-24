# Skill: Evaluation and Analysis

## Purpose
Systematically evaluate DRL agents, compare against baselines, perform ablation studies, and generate publication-quality figures and tables for the thesis.

## Evaluation Protocol

### 1. Convergence Analysis
```python
def evaluate_convergence(results_dir, algorithms, n_seeds=5):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

    for idx, algo in enumerate(algorithms):
        # Load results: shape (n_seeds, n_episodes)
        rewards = load_results(results_dir, algo, metric="episode_reward")

        # Smooth with moving average
        window = 50
        smoothed = np.array([np.convolve(r, np.ones(window)/window, mode='valid') 
                             for r in rewards])

        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        episodes = np.arange(window, len(rewards[0]) + 1)

        axes[0, 0].plot(episodes, mean, label=algo, color=colors[idx], linewidth=2)
        axes[0, 0].fill_between(episodes, mean - std, mean + std, 
                                alpha=0.2, color=colors[idx])

    axes[0, 0].set_xlabel("Episode", fontsize=12)
    axes[0, 0].set_ylabel("Episode Reward", fontsize=12)
    axes[0, 0].set_title("(a) Convergence Comparison", fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Similar for energy, QoS, switching cost subplots

    plt.tight_layout()
    plt.savefig("thesis/figures/convergence.pdf", dpi=300, bbox_inches='tight')
    plt.close()
```

### 2. Energy Efficiency Comparison
```python
def evaluate_energy_efficiency(results_dir, algorithms, n_eval_episodes=100):
    results = {}

    for algo in algorithms:
        energy_data = []
        for seed in range(5):
            env = CRANEnv(config)
            agent = load_agent(results_dir, algo, seed)

            episode_energies = []
            for _ in range(n_eval_episodes):
                state, _ = env.reset(seed=seed)
                episode_energy = 0
                for step in range(config.max_steps):
                    action = agent.select_action(state, evaluate=True)
                    state, _, _, _, info = env.step(action)
                    episode_energy += info["total_power"]
                episode_energies.append(episode_energy)

            energy_data.extend(episode_energies)

        results[algo] = {
            "mean": np.mean(energy_data),
            "std": np.std(energy_data),
            "ci95": 1.96 * np.std(energy_data) / np.sqrt(len(energy_data))
        }

    # Compute savings vs. All ON baseline
    baseline_energy = results["All_ON"]["mean"]
    for algo in algorithms:
        if algo != "All_ON":
            savings = (baseline_energy - results[algo]["mean"]) / baseline_energy * 100
            results[algo]["savings_pct"] = savings

    return results
```

### 3. Ablation Study
```python
def ablation_study(base_config, variants):
    results = {}

    for name, cfg_override in variants:
        config = copy.deepcopy(base_config)
        config.update(cfg_override)

        # Train agent
        agent = HybridSACDDQN(config)
        train(agent, config)

        # Evaluate
        eval_results = evaluate(agent, config, n_episodes=100)
        results[name] = eval_results

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    energies = [results[n]["mean_energy"] for n in names]
    errors = [results[n]["std_energy"] for n in names]

    bars = ax.bar(names, energies, yerr=errors, capsize=5, 
                  color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])
    ax.set_ylabel("Average Energy Consumption (W)", fontsize=12)
    ax.set_title("Ablation Study: Impact of Reward Components", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig("thesis/figures/ablation.pdf", dpi=300, bbox_inches='tight')

    return results

# Usage
variants = [
    ("Full Model", {}),
    ("No Switching Cost", {"gamma_switch": 0}),
    ("No Fronthaul Power", {"fronthaul_weight": 0}),
    ("No QoS Penalty", {"beta_qos": 0}),
]
```

### 4. Scalability Analysis
```python
def scalability_analysis(base_config, scenarios):
    results = {"energy": [], "time": [], "qos": []}

    for scenario in scenarios:
        config = copy.deepcopy(base_config)
        config.update(scenario)

        start_time = time.time()
        agent = HybridSACDDQN(config)
        train(agent, config)
        train_time = time.time() - start_time

        eval_results = evaluate(agent, config)

        results["energy"].append(eval_results["mean_energy"])
        results["time"].append(train_time)
        results["qos"].append(eval_results["qos_violation_rate"])

    # Plot scalability
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = [s["label"] for s in scenarios]
    x = range(len(labels))

    axes[0].plot(x, results["energy"], 'o-', linewidth=2, markersize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Energy (W)")
    axes[0].set_title("Energy vs. Network Size")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, results["time"], 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Training Time (hours)")
    axes[1].set_title("Training Time vs. Network Size")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, results["qos"], '^-', linewidth=2, markersize=8, color='red')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("QoS Violation Rate")
    axes[2].set_title("QoS vs. Network Size")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("thesis/figures/scalability.pdf", dpi=300, bbox_inches='tight')

    return results
```

### 5. Statistical Significance Testing
```python
from scipy import stats

def compare_algorithms(results_dir, algo1, algo2, metric="episode_reward", 
                       n_seeds=5, n_episodes=100):
    data1 = load_metric(results_dir, algo1, metric, n_seeds, n_episodes)
    data2 = load_metric(results_dir, algo2, metric, n_seeds, n_episodes)

    # Paired t-test (same seeds)
    t_stat, p_value = stats.ttest_rel(data1, data2)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(data1)**2 + np.std(data2)**2) / 2)
    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std

    return {
        "algo1_mean": np.mean(data1),
        "algo2_mean": np.mean(data2),
        "difference": np.mean(data1) - np.mean(data2),
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05
    }
```

## Figure Standards

### Color Scheme
```python
# Consistent across all figures
COLORS = {
    "All_ON": "#e74c3c",      # Red
    "Greedy": "#f39c12",       # Orange
    "NMBS": "#9b59b6",         # Purple
    "Convex": "#3498db",       # Blue
    "DDQN": "#1abc9c",         # Teal
    "DDPG": "#2ecc71",         # Green
    "TD3": "#34495e",          # Dark gray
    "SAC": "#e67e22",          # Dark orange
    "Hybrid": "#2980b9",       # Dark blue (proposed method)
}
```

### LaTeX Figure Export
```python
def setup_matplotlib_for_latex():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
    })
```

## Required Outputs for Thesis

| Output | File | Section |
|--------|------|---------|
| Convergence curves | `figures/convergence.pdf` | 4.2 |
| 24-hour energy profile | `figures/energy_profile.pdf` | 4.3 |
| SINR CDF | `figures/sinr_cdf.pdf` | 4.4 |
| Ablation bar chart | `figures/ablation.pdf` | 4.5 |
| Scalability triple plot | `figures/scalability.pdf` | 4.6 |
| Parameter table | `tables/parameters.tex` | 4.1 |
| Results table | `tables/results.tex` | 4.3 |
| Ablation table | `tables/ablation.tex` | 4.5 |
| Scalability table | `tables/scalability.tex` | 4.6 |

## Validation Checklist
- [ ] All figures use consistent color scheme
- [ ] All error bars represent 95% confidence intervals
- [ ] All tables have units in headers
- [ ] Statistical significance reported for all comparisons
- [ ] Best results bolded in tables
- [ ] Figures are vector graphics (PDF)
- [ ] All figures referenced in text before they appear
- [ ] Captions are self-contained (explain what is shown and key takeaway)
