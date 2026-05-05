#!/usr/bin/env python
"""Run long reversal-learning simulation to inspect late route dynamics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crrp_rl_core import Config, run_task, plot_learning, plot_routes, save_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/long_reversal")
    p.add_argument("--n-seeds", type=int, default=30)
    p.add_argument("--n-trials-reversal", type=int, default=5000)
    p.add_argument("--reversal-trial", type=int, default=450)
    p.add_argument("--eta-cb", type=float, default=0.020)
    p.add_argument("--eta-bg-rpe", type=float, default=0.018)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        n_seeds=args.n_seeds,
        n_trials_reversal=args.n_trials_reversal,
        reversal_trial=args.reversal_trial,
        output_dir=str(outdir),
        eta_cb=args.eta_cb,
        eta_bg_rpe=args.eta_bg_rpe,
    )
    save_config(cfg, outdir)

    df, summary = run_task("reversal", cfg, ablations=("full",), include_baselines=True)
    df.to_csv(outdir / "long_reversal_trials.csv", index=False)
    summary.to_csv(outdir / "long_reversal_summary.csv", index=False)
    plot_learning(df, "reversal", cfg, outdir, suffix="_long")
    plot_routes(df[df["ablation"].eq("full")], "reversal", cfg, outdir, suffix="_long")
    print("\nLONG REVERSAL SUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
