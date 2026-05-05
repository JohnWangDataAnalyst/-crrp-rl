#!/usr/bin/env python
"""Run main stable and reversal CRRP-RL simulations with Q-learning and actor-critic baselines."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crrp_rl_core import Config, run_task, plot_learning, plot_routes, save_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/main_simulations")
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--n-trials-stable", type=int, default=700)
    p.add_argument("--n-trials-reversal", type=int, default=1800)
    p.add_argument("--reversal-trial", type=int, default=450)
    p.add_argument("--eta-cb", type=float, default=0.020)
    p.add_argument("--eta-bg-rpe", type=float, default=0.018)
    p.add_argument("--no-baselines", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        n_seeds=args.n_seeds,
        n_trials_stable=args.n_trials_stable,
        n_trials_reversal=args.n_trials_reversal,
        reversal_trial=args.reversal_trial,
        output_dir=str(outdir),
        eta_cb=args.eta_cb,
        eta_bg_rpe=args.eta_bg_rpe,
    )
    save_config(cfg, outdir)

    for task in ["stable", "reversal"]:
        df, summary = run_task(task, cfg, ablations=("full",), include_baselines=not args.no_baselines)
        df.to_csv(outdir / f"{task}_trials.csv", index=False)
        summary.to_csv(outdir / f"{task}_summary.csv", index=False)
        plot_learning(df, task, cfg, outdir)
        plot_routes(df[df["ablation"].eq("full")], task, cfg, outdir)
        print(f"\n{task.upper()} SUMMARY")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
