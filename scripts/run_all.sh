#!/usr/bin/env bash
set -euo pipefail

# Example full pipeline. Defaults are manuscript-like but can take time.
# For a quick smoke test, set N_SEEDS=3 before running.
N_SEEDS=${N_SEEDS:-30}

python scripts/run_main_simulations.py --n-seeds "$N_SEEDS" --output-dir outputs/main_simulations
python scripts/run_ablation_analysis.py --n-seeds "$N_SEEDS" --output-dir outputs/ablation_analysis
python scripts/run_eta_sensitivity.py --n-seeds "$N_SEEDS" --output-dir outputs/eta_sensitivity
python scripts/run_long_reversal.py --n-seeds "$N_SEEDS" --output-dir outputs/long_reversal
