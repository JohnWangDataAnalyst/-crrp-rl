#!/usr/bin/env python
"""Run CRRP-RL ablation analysis for stable and reversal tasks."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crrp_rl_core import Config, run_task, plot_learning, plot_routes, save_config

DEFAULT_ABLATIONS = (
    "full",
    "fixed_routes",
    "no_cb_to_tha",
    "no_bg_rpe",
    "no_bg_confidence_gate",
    "softmax_z_only",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/ablation_analysis")
    p.add_argument("--n-seeds", type=int, default=30)
    p.add_argument("--n-trials-stable", type=int, default=700)
    p.add_argument("--n-trials-reversal", type=int, default=1800)
    p.add_argument("--reversal-trial", type=int, default=450)
    p.add_argument("--eta-cb", type=float, default=0.020)
    p.add_argument("--eta-bg-rpe", type=float, default=0.018)
    p.add_argument("--include-no-tha-cue", action="store_true")
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

    ablations = list(DEFAULT_ABLATIONS)
    if args.include_no_tha_cue:
        ablations.append("no_tha_cue_gate")

    for task in ["stable", "reversal"]:
        df, summary = run_task(task, cfg, ablations=ablations, include_baselines=True)
        df.to_csv(outdir / f"{task}_ablation_trials.csv", index=False)
        summary.to_csv(outdir / f"{task}_ablation_summary.csv", index=False)
        plot_learning(df, task, cfg, outdir, suffix="_ablation")
        plot_routes(df, task, cfg, outdir, suffix="_ablation")
        print(f"\n{task.upper()} ABLATION SUMMARY")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
