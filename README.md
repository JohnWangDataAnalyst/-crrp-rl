# -crrp-rl
Simplex-constrained route-specific RL with cortico-subcortical observer dynamics 
This repository contains the simulation code and benchmark scripts for the paper:

**CRRP-RL: Simplex-Constrained Route-Specific Reinforcement Learning with THA/BG Balance Regularization**  
Zheng Wang, Wenlin Zhang, Shuai Li  
Institute of Brain and Mind, Irvine, California, United States  
*Submitted to PLOS Computational Biology*

---

## Overview

CRRP-RL is a reinforcement-learning extension of the Linear Observer-Control Framework (LOCF) and Cerebellum-Referenced Route Plasticity (CRRP) model. Route weights across thalamic (ρ_tha), basal ganglia (ρ_bg), and cerebellar (ρ_cb) observer routes are explicitly constrained to a probability simplex via softmax allocation, producing interpretable allocation proportions that serve as trial-level biomarkers of cortico-subcortical learning dynamics.

---

## Repository structure

```
crrp_rl/
├── crrp_rl_constrained_mismatch.py   # Core model: single-run simulation
├── run_benchmark.py                   # 100-seed benchmark vs Q-learning and actor-critic
├── run_reversal.py                    # Long reversal-learning simulation (Simulation 2)
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── figures/                           # Pre-generated publication figures
    ├── representative_accuracy.png
    ├── representative_route_weights.png
    ├── representative_errors.png
    ├── representative_mismatch_terms.png
    ├── benchmark_learning_curves.png
    ├── benchmark_early_late_accuracy.png
    ├── benchmark_route_weights.png
    ├── benchmark_mismatch_terms.png
    ├── long_reversal_learning_curves.png
    ├── long_reversal_routes_current.png
    ├── long_reversal_routes_ratio004.png
    ├── long_reversal_routes_ratio008.png
    └── model_diagram.png
```

---

## Requirements

```
numpy>=1.24
pandas>=1.5
matplotlib>=3.6
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Quickstart

### Run a single representative simulation (Simulation 1)

```bash
python crrp_rl_constrained_mismatch.py
```

Outputs saved to `crrp_rl_constrained_outputs/`:
- `crrp_rl_constrained_trials.csv` — trial-level data (700 rows)
- `crrp_rl_constrained_summary.csv` — early/late summary statistics
- Four figures (accuracy, route weights, RPE/mismatch, route-specific mismatch)

### Run the 100-seed benchmark (Table 1, Figures 3–4)

```bash
python run_benchmark.py
```

Compares CRRP-RL constrained against tabular Q-learning and actor-critic across 100 random seeds. Outputs saved to `crrp_rl_constrained_outputs/`.

### Run the reversal-learning simulation (Table 2, Figures 5–6)

```bash
python run_reversal.py
```

1800-trial simulation with contingency reversal at trial 450. Compares CRRP-current, CRRP-ratio 0.04, CRRP-ratio 0.08, and Q-learning.

---

## Model summary

The core route allocation is a softmax over slow biases z_k and fast task-driven inputs u_k:

```
ρ_k(t) = exp(z_k(t) + u_k(t)) / Σ_j exp(z_j(t) + u_j(t))
```

This enforces ρ_tha + ρ_bg + ρ_cb = 1, ρ_k ≥ 0 to machine precision.

Plasticity update rules:
- **Thalamic and cerebellar routes**: broadband-reference mismatch competition
  - z_tha ← z_tha + η_tha (e_mix − e_tha)
  - z_cb  ← z_cb  + η_cb  (e_mix − e_cb)
- **Basal ganglia route**: reward prediction error
  - z_bg ← z_bg + η_bg · δ(t) · ρ_bg(t) · ε_π(t)
- **THA/BG ratio regularization** (optional): soft stability–plasticity regulator

---

## Key results

| Model               | First 100 accuracy | Last 100 accuracy | Overall accuracy |
|---------------------|--------------------|-------------------|------------------|
| Actor-critic        | 0.862 ± 0.036      | 0.996 ± 0.007     | 0.970 ± 0.005    |
| CRRP-RL constrained | 0.948 ± 0.022      | 0.999 ± 0.004     | 0.989 ± 0.003    |
| Q-learning          | 0.922 ± 0.038      | 0.984 ± 0.012     | 0.974 ± 0.008    |

Route weights shifted from early distributed allocation (ρ_tha = 0.367, ρ_bg = 0.392, ρ_cb = 0.241) to basal ganglia dominance (ρ_tha = 0.122, ρ_bg = 0.785, ρ_cb = 0.093) by the last 100 trials.

---

## Citation

If you use this code, please cite:

```bibtex
@article{wang2025crrprl,
  title   = {{CRRP-RL}: Simplex-Constrained Route-Specific Reinforcement Learning
             with {THA/BG} Balance Regularization},
  author  = {Wang, Zheng and Zhang, Wenlin and Li, Shuai},
  journal = {PLOS Computational Biology},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License. See `LICENSE` for details.
