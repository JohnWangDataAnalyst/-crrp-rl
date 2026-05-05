#!/usr/bin/env python
"""Run eta_CB / eta_BG-RPE sensitivity sweep for reversal learning."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crrp_rl_core import Config, run_task, save_config


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/eta_sensitivity")
    p.add_argument("--n-seeds", type=int, default=30)
    p.add_argument("--n-trials-reversal", type=int, default=1800)
    p.add_argument("--reversal-trial", type=int, default=450)
    p.add_argument("--eta-cb-grid", default="0.02,0.04,0.06,0.08,0.10")
    p.add_argument("--eta-bg-grid", default="0.008,0.012,0.018,0.024")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    eta_cb_grid = parse_float_list(args.eta_cb_grid)
    eta_bg_grid = parse_float_list(args.eta_bg_grid)

    all_summary = []
    all_trials = []
    for eta_cb in eta_cb_grid:
        for eta_bg in eta_bg_grid:
            cfg = Config(
                n_seeds=args.n_seeds,
                n_trials_reversal=args.n_trials_reversal,
                reversal_trial=args.reversal_trial,
                output_dir=str(outdir),
                eta_cb=eta_cb,
                eta_bg_rpe=eta_bg,
            )
            df, summary = run_task("reversal", cfg, ablations=("full",), include_baselines=False)
            df["eta_cb"] = eta_cb
            df["eta_bg_rpe"] = eta_bg
            summary["eta_cb"] = eta_cb
            summary["eta_bg_rpe"] = eta_bg
            all_trials.append(df)
            all_summary.append(summary)
            print(f"eta_cb={eta_cb:.3g}, eta_bg_rpe={eta_bg:.3g}")
            print(summary.to_string(index=False))

    trials = pd.concat(all_trials, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)
    trials.to_csv(outdir / "reversal_eta_sensitivity_trials.csv", index=False)
    summary.to_csv(outdir / "reversal_eta_sensitivity_summary.csv", index=False)

    # A simple route-score for ranking: strong THA rebound + BG reconsolidation - switching, with faster T80 better.
    ranked = summary.copy()
    recover = ranked.get("recover_80_median", 0).fillna(ranked.get("recover_80_median", 0).max())
    ranked["route_score"] = (
        ranked.get("tha_rebound_mean", 0).fillna(0)
        + ranked.get("bg_reconsolidation_mean", 0).fillna(0)
        - ranked.get("rho_switching_mean", 0).fillna(0)
        - 0.0005 * recover.fillna(0)
    )
    ranked = ranked.sort_values("route_score", ascending=False)
    ranked.to_csv(outdir / "top_eta_pairs_by_route_score.csv", index=False)

    save_config(Config(n_seeds=args.n_seeds, n_trials_reversal=args.n_trials_reversal), outdir)
    print("\nTOP ETA PAIRS")
    print(ranked.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
