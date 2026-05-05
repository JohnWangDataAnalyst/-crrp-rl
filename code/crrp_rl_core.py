"""
crrp_rl_core.py

Core simulation, ablation, analysis, and plotting functions for the
fast-gated signed-CB CRRP-RL companion computational/theory paper.

Final model
-----------
ell_tha = z_tha + alpha_cue * |x_tha|
ell_bg  = z_bg  + alpha_bg  * |V_bg(s)|
rho_tha, rho_bg = softmax(ell_tha, ell_bg)
x_mix = rho_tha * x_tha + rho_bg * x_bg

eps_cb = x_cb - x_mix
dz_tha_cb = eta_cb * eps_cb * rho_tha * rho_bg * (x_tha - x_bg)
dz_bg = eta_bg_rpe * rpe * rho_bg * policy_error + lambda_bg * |V_bg(s)|

Author: John Wang / ChatGPT draft
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TaskName = Literal["stable", "reversal"]
AblationName = Literal[
    "full",
    "fixed_routes",
    "no_cb_to_tha",
    "no_bg_rpe",
    "no_bg_confidence_gate",
    "no_tha_cue_gate",
    "softmax_z_only",
]

MODEL_NAME = "CRRP-RL-fast-gated-signed-CB"
ABLATION_LABELS: dict[str, str] = {
    "full": "Full signed-CB fast-gated CRRP-RL",
    "fixed_routes": "Ablation: fixed routes",
    "no_cb_to_tha": "Ablation: no CB->THA correction",
    "no_bg_rpe": "Ablation: no BG-RPE route update",
    "no_bg_confidence_gate": "Ablation: no BG confidence gate",
    "no_tha_cue_gate": "Ablation: no THA cue gate",
    "softmax_z_only": "Ablation: softmax-z-only",
}


@dataclass
class Config:
    n_seeds: int = 100
    n_trials_stable: int = 700
    n_trials_reversal: int = 1800
    reversal_trial: int = 450
    rolling_window: int = 50
    output_dir: str = "outputs/crrp_rl"

    # policy / actor
    beta: float = 4.0
    eta_actor: float = 0.06

    # critic / value
    eta_value: float = 0.10

    # route learning: manuscript main values
    eta_cb: float = 0.020
    eta_bg_rpe: float = 0.018
    lambda_bg: float = 0.003
    bias_decay: float = 0.0015

    # fast route gates
    cue_gate_strength: float = 0.80
    bg_confidence_gate_strength: float = 1.15

    # observer noise
    initial_noise: float = 0.55
    final_noise_stable: float = 0.25
    final_noise_reversal: float = 0.30

    # initial route bias and actor
    initial_z_tha: float = 0.00
    initial_z_bg: float = -0.20
    initial_actor_weight: float = 0.20

    # baseline model parameters
    q_learning_alpha: float = 0.12
    q_learning_beta: float = 4.0
    ac_actor_alpha: float = 0.08
    ac_critic_alpha: float = 0.12
    ac_beta: float = 4.0


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax2(a: float, b: float) -> tuple[float, float]:
    z = np.array([a, b], dtype=float)
    z -= np.max(z)
    ev = np.exp(z)
    rho = ev / ev.sum()
    return float(rho[0]), float(rho[1])


def make_trial(rng: np.random.Generator, trial: int, task: TaskName, cfg: Config) -> tuple[float, int, int, str]:
    cue = float(rng.choice([-1.0, 1.0]))
    state = 1 if cue > 0 else 0

    if task == "stable":
        correct_action = 1 if cue > 0 else 0
        phase = "stable"
    elif task == "reversal":
        if trial < cfg.reversal_trial:
            correct_action = 1 if cue > 0 else 0
            phase = "pre-reversal"
        else:
            correct_action = 0 if cue > 0 else 1
            phase = "post-reversal"
    else:
        raise ValueError(f"Unknown task: {task}")

    return cue, state, correct_action, phase


def run_crrp_rl(
    seed: int,
    task: TaskName,
    cfg: Config,
    ablation: AblationName = "full",
) -> pd.DataFrame:
    """Run one seed of CRRP-RL with optional ablation."""
    if ablation not in ABLATION_LABELS:
        raise ValueError(f"Unknown ablation: {ablation}")

    rng = np.random.default_rng(seed)
    n_trials = cfg.n_trials_stable if task == "stable" else cfg.n_trials_reversal
    final_noise = cfg.final_noise_stable if task == "stable" else cfg.final_noise_reversal

    z_tha = cfg.initial_z_tha
    z_bg = cfg.initial_z_bg
    w_actor = cfg.initial_actor_weight
    value = np.zeros(2)
    records: list[dict] = []

    for t in range(n_trials):
        cue, state, correct_action, phase = make_trial(rng, t, task, cfg)

        frac = t / max(n_trials - 1, 1)
        noise = max(cfg.initial_noise + frac * (final_noise - cfg.initial_noise), final_noise)

        # Scalar toy observer states. In the full LOCF setting, these are replaced
        # by route-specific state-space observer estimates.
        x_tha = 1.05 * cue + rng.normal(0, noise)
        bg_conf = abs(value[state])
        bg_gain = 0.25 + 1.10 * sigmoid(2.2 * bg_conf)
        x_bg = bg_gain * cue + rng.normal(0, noise * 0.80)
        x_cb = 1.00 * cue + rng.normal(0, noise * 0.40)

        # Fast-gated logits. Ablations remove selected gates.
        cue_gate = 0.0 if ablation in {"no_tha_cue_gate", "softmax_z_only"} else cfg.cue_gate_strength * abs(x_tha)
        bg_gate = 0.0 if ablation in {"no_bg_confidence_gate", "softmax_z_only"} else cfg.bg_confidence_gate_strength * bg_conf
        logit_tha = z_tha + cue_gate
        logit_bg = z_bg + bg_gate

        if ablation == "fixed_routes":
            rho_tha, rho_bg = 0.5, 0.5
        else:
            rho_tha, rho_bg = softmax2(logit_tha, logit_bg)

        x_mix = rho_tha * x_tha + rho_bg * x_bg

        # Actor policy.
        p_action1 = float(sigmoid(cfg.beta * w_actor * x_mix))
        action = 1 if rng.random() < p_action1 else 0
        reward = 1.0 if action == correct_action else 0.0

        # Critic and actor updates.
        rpe = reward - value[state]
        value[state] += cfg.eta_value * rpe
        w_actor += cfg.eta_actor * rpe * cfg.beta * x_mix * (action - p_action1)

        # Signed policy error for the currently rewarded/correct action channel.
        if correct_action == 1:
            policy_error = reward - p_action1
        else:
            policy_error = reward - (1.0 - p_action1)

        # Signed CB-reference mismatch and gradient-like THA correction.
        eps_cb = x_cb - x_mix
        d_mix_d_ztha = rho_tha * rho_bg * (x_tha - x_bg)
        dz_tha_cb = cfg.eta_cb * eps_cb * d_mix_d_ztha
        if ablation in {"no_cb_to_tha", "fixed_routes"}:
            dz_tha_cb = 0.0

        # BG reward/value consolidation.
        dz_bg_reward = cfg.eta_bg_rpe * rpe * rho_bg * policy_error + cfg.lambda_bg * abs(value[state])
        if ablation in {"no_bg_rpe", "fixed_routes"}:
            # Remove the explicit RPE route-plasticity term, but keep the value-consolidation bias.
            dz_bg_reward = cfg.lambda_bg * abs(value[state]) if ablation == "no_bg_rpe" else 0.0

        if ablation != "fixed_routes":
            z_tha += dz_tha_cb
            z_bg += dz_bg_reward
            z_tha *= 1.0 - cfg.bias_decay
            z_bg *= 1.0 - cfg.bias_decay

        records.append({
            "seed": seed,
            "trial": t + 1,
            "task": task,
            "model": ABLATION_LABELS[ablation],
            "ablation": ablation,
            "phase": phase,
            "cue": cue,
            "state": state,
            "correct_action": correct_action,
            "action": action,
            "reward": reward,
            "rpe": rpe,
            "abs_rpe": abs(rpe),
            "policy_error": policy_error,
            "rho_tha": rho_tha,
            "rho_bg": rho_bg,
            "rho_sum": rho_tha + rho_bg,
            "x_mix": x_mix,
            "x_tha": x_tha,
            "x_bg": x_bg,
            "x_cb": x_cb,
            "eps_cb": eps_cb,
            "cb_mismatch_energy": eps_cb ** 2,
            "d_mix_d_ztha": d_mix_d_ztha,
            "dz_tha_cb": dz_tha_cb,
            "dz_bg_reward": dz_bg_reward,
            "logit_tha": logit_tha,
            "logit_bg": logit_bg,
            "cue_gate": cue_gate,
            "bg_gate": bg_gate,
            "bg_conf": bg_conf,
            "w_actor": w_actor,
            "value_state": value[state],
            "z_tha": z_tha,
            "z_bg": z_bg,
        })

    return pd.DataFrame(records)


def run_q_learning(seed: int, task: TaskName, cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_trials = cfg.n_trials_stable if task == "stable" else cfg.n_trials_reversal
    q = np.zeros((2, 2))
    records = []

    for t in range(n_trials):
        cue, state, correct_action, phase = make_trial(rng, t, task, cfg)
        logits = cfg.q_learning_beta * q[state]
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        action = 1 if rng.random() < probs[1] else 0
        reward = 1.0 if action == correct_action else 0.0
        pe = reward - q[state, action]
        q[state, action] += cfg.q_learning_alpha * pe
        records.append({
            "seed": seed, "trial": t + 1, "task": task, "model": "Q-learning",
            "ablation": "baseline", "phase": phase, "reward": reward,
            "rpe": pe, "abs_rpe": abs(pe), "rho_tha": np.nan, "rho_bg": np.nan, "rho_sum": np.nan,
        })
    return pd.DataFrame(records)


def run_actor_critic(seed: int, task: TaskName, cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_trials = cfg.n_trials_stable if task == "stable" else cfg.n_trials_reversal
    h = np.zeros((2, 2))
    value = np.zeros(2)
    records = []

    for t in range(n_trials):
        cue, state, correct_action, phase = make_trial(rng, t, task, cfg)
        logits = cfg.ac_beta * h[state]
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        action = 1 if rng.random() < probs[1] else 0
        reward = 1.0 if action == correct_action else 0.0
        delta = reward - value[state]
        value[state] += cfg.ac_critic_alpha * delta
        for a in [0, 1]:
            indicator = 1.0 if a == action else 0.0
            h[state, a] += cfg.ac_actor_alpha * delta * (indicator - probs[a])
        records.append({
            "seed": seed, "trial": t + 1, "task": task, "model": "Actor-critic",
            "ablation": "baseline", "phase": phase, "reward": reward,
            "rpe": delta, "abs_rpe": abs(delta), "rho_tha": np.nan, "rho_bg": np.nan, "rho_sum": np.nan,
        })
    return pd.DataFrame(records)


def add_rolling_metrics(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    sort_cols = ["task", "model", "seed", "trial"]
    df["rolling_accuracy"] = (
        df.sort_values(sort_cols)
          .groupby(["task", "model", "seed"])["reward"]
          .transform(lambda s: s.rolling(cfg.rolling_window, min_periods=1).mean())
    )
    df["rolling_abs_rpe"] = (
        df.sort_values(sort_cols)
          .groupby(["task", "model", "seed"])["abs_rpe"]
          .transform(lambda s: s.rolling(cfg.rolling_window, min_periods=1).mean())
    )
    return df


def summarize_stable(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    def _summ(g: pd.DataFrame) -> pd.Series:
        early = g["trial"] <= 100
        late = g["trial"] > cfg.n_trials_stable - 100
        return pd.Series({
            "early_acc": g.loc[early, "reward"].mean(),
            "late_acc": g.loc[late, "reward"].mean(),
            "overall_acc": g["reward"].mean(),
            "rho_tha_early": g.loc[early, "rho_tha"].mean(),
            "rho_bg_early": g.loc[early, "rho_bg"].mean(),
            "rho_tha_late": g.loc[late, "rho_tha"].mean(),
            "rho_bg_late": g.loc[late, "rho_bg"].mean(),
            "rho_bg_growth": g.loc[late, "rho_bg"].mean() - g.loc[early, "rho_bg"].mean(),
            "cb_mismatch_energy": g.get("cb_mismatch_energy", pd.Series(np.nan, index=g.index)).mean(),
            "rho_switching": g["rho_tha"].diff().abs().mean(),
            "rho_sum_error_max": np.nanmax(np.abs(g["rho_sum"] - 1.0)) if g["rho_sum"].notna().any() else np.nan,
        })
    seed_summary = df.groupby(["model", "seed"], group_keys=False).apply(_summ).reset_index()
    return seed_summary.groupby("model").agg(
        early_acc_mean=("early_acc", "mean"), early_acc_sd=("early_acc", "std"),
        late_acc_mean=("late_acc", "mean"), late_acc_sd=("late_acc", "std"),
        overall_acc_mean=("overall_acc", "mean"), overall_acc_sd=("overall_acc", "std"),
        rho_tha_early_mean=("rho_tha_early", "mean"), rho_bg_early_mean=("rho_bg_early", "mean"),
        rho_tha_late_mean=("rho_tha_late", "mean"), rho_bg_late_mean=("rho_bg_late", "mean"),
        rho_bg_growth_mean=("rho_bg_growth", "mean"),
        cb_mismatch_energy_mean=("cb_mismatch_energy", "mean"),
        rho_switching_mean=("rho_switching", "mean"), rho_sum_error_max=("rho_sum_error_max", "max"),
    ).reset_index()


def summarize_reversal(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    def recover(g: pd.DataFrame, threshold: float = 0.80) -> float:
        sub = g[g["trial"] >= cfg.reversal_trial + cfg.rolling_window]
        hits = sub[sub["rolling_accuracy"] >= threshold]
        return np.nan if hits.empty else int(hits["trial"].iloc[0] - cfg.reversal_trial)

    def _summ(g: pd.DataFrame) -> pd.Series:
        pre = (g["trial"] >= cfg.reversal_trial - 100) & (g["trial"] < cfg.reversal_trial)
        post_first = (g["trial"] >= cfg.reversal_trial + 1) & (g["trial"] <= cfg.reversal_trial + 100)
        post_mid = (g["trial"] >= 1200) & (g["trial"] < 1300)
        post_final = g["trial"] > cfg.n_trials_reversal - 100
        rho_tha_pre = g.loc[pre, "rho_tha"].mean()
        rho_bg_post = g.loc[post_first, "rho_bg"].mean()
        return pd.Series({
            "pre_acc": g.loc[pre, "reward"].mean(),
            "post_first100_acc": g.loc[post_first, "reward"].mean(),
            "post_1200_1300_acc": g.loc[post_mid, "reward"].mean(),
            "post_final100_acc": g.loc[post_final, "reward"].mean(),
            "recover_80": recover(g),
            "rho_tha_pre": rho_tha_pre,
            "rho_bg_pre": g.loc[pre, "rho_bg"].mean(),
            "rho_tha_post_first100": g.loc[post_first, "rho_tha"].mean(),
            "rho_bg_post_first100": rho_bg_post,
            "rho_tha_final100": g.loc[post_final, "rho_tha"].mean(),
            "rho_bg_final100": g.loc[post_final, "rho_bg"].mean(),
            "tha_rebound": g.loc[post_first, "rho_tha"].mean() - rho_tha_pre,
            "bg_reconsolidation": g.loc[post_final, "rho_bg"].mean() - rho_bg_post,
            "cb_mismatch_energy": g.get("cb_mismatch_energy", pd.Series(np.nan, index=g.index)).mean(),
            "rho_switching": g["rho_tha"].diff().abs().mean(),
            "rho_sum_error_max": np.nanmax(np.abs(g["rho_sum"] - 1.0)) if g["rho_sum"].notna().any() else np.nan,
        })
    seed_summary = df.groupby(["model", "seed"], group_keys=False).apply(_summ).reset_index()
    return seed_summary.groupby("model").agg(
        pre_acc_mean=("pre_acc", "mean"), pre_acc_sd=("pre_acc", "std"),
        post_first100_acc_mean=("post_first100_acc", "mean"), post_first100_acc_sd=("post_first100_acc", "std"),
        post_1200_1300_acc_mean=("post_1200_1300_acc", "mean"), post_1200_1300_acc_sd=("post_1200_1300_acc", "std"),
        post_final100_acc_mean=("post_final100_acc", "mean"), post_final100_acc_sd=("post_final100_acc", "std"),
        recover_80_median=("recover_80", "median"),
        recover_80_iqr_low=("recover_80", lambda x: np.nanpercentile(x, 25)),
        recover_80_iqr_high=("recover_80", lambda x: np.nanpercentile(x, 75)),
        rho_tha_pre_mean=("rho_tha_pre", "mean"), rho_bg_pre_mean=("rho_bg_pre", "mean"),
        rho_tha_post_first100_mean=("rho_tha_post_first100", "mean"), rho_bg_post_first100_mean=("rho_bg_post_first100", "mean"),
        rho_tha_final100_mean=("rho_tha_final100", "mean"), rho_bg_final100_mean=("rho_bg_final100", "mean"),
        tha_rebound_mean=("tha_rebound", "mean"), bg_reconsolidation_mean=("bg_reconsolidation", "mean"),
        cb_mismatch_energy_mean=("cb_mismatch_energy", "mean"), rho_switching_mean=("rho_switching", "mean"),
        rho_sum_error_max=("rho_sum_error_max", "max"),
    ).reset_index()


def run_task(
    task: TaskName,
    cfg: Config,
    ablations: Iterable[AblationName] = ("full",),
    include_baselines: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for seed in range(cfg.n_seeds):
        for ablation in ablations:
            frames.append(run_crrp_rl(seed, task, cfg, ablation=ablation))
        if include_baselines:
            frames.append(run_q_learning(seed, task, cfg))
            frames.append(run_actor_critic(seed, task, cfg))
    df = pd.concat(frames, ignore_index=True)
    df = add_rolling_metrics(df, cfg)
    summary = summarize_stable(df, cfg) if task == "stable" else summarize_reversal(df, cfg)
    return df, summary


def plot_learning(df: pd.DataFrame, task: TaskName, cfg: Config, outdir: Path, suffix: str = "") -> None:
    curve = df.groupby(["model", "trial"])["rolling_accuracy"].agg(["mean", "std", "count"]).reset_index()
    curve["sem"] = curve["std"] / np.sqrt(curve["count"])
    plt.figure(figsize=(9.5, 5.2))
    for model, sub in curve.groupby("model"):
        x = sub["trial"].to_numpy()
        y = sub["mean"].to_numpy()
        sem = sub["sem"].fillna(0).to_numpy()
        plt.plot(x, y, label=model)
        plt.fill_between(x, y - sem, y + sem, alpha=0.15)
    if task == "reversal":
        plt.axvline(cfg.reversal_trial, linestyle="--", label="Reversal")
    plt.ylim(0, 1.03)
    plt.xlabel("Trial")
    plt.ylabel(f"Rolling accuracy, window={cfg.rolling_window}")
    plt.title(f"CRRP-RL {task} task")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"{task}_learning_curves{suffix}.png", dpi=220)
    plt.close()


def plot_routes(df: pd.DataFrame, task: TaskName, cfg: Config, outdir: Path, suffix: str = "") -> None:
    cr = df[df["rho_tha"].notna()].copy()
    if cr.empty:
        return
    plt.figure(figsize=(9.5, 5.2))
    for model, subdf in cr.groupby("model"):
        rho = subdf.groupby("trial")[["rho_tha", "rho_bg"]].mean().reset_index()
        plt.plot(rho["trial"], rho["rho_tha"], label=f"{model}: rho_tha")
        if model == cr["model"].iloc[0] and len(cr["model"].unique()) == 1:
            plt.plot(rho["trial"], rho["rho_bg"], label=f"{model}: rho_bg")
    if task == "reversal":
        plt.axvline(cfg.reversal_trial, linestyle="--", label="Reversal")
    plt.ylim(0, 1.0)
    plt.xlabel("Trial")
    plt.ylabel("Route weight")
    plt.title(f"CRRP-RL route allocation: {task}")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"{task}_route_weights{suffix}.png", dpi=220)
    plt.close()


def save_config(cfg: Config, outdir: Path) -> None:
    pd.DataFrame([asdict(cfg)]).to_csv(outdir / "config.csv", index=False)


def with_output_dir(cfg: Config, output_dir: str) -> Config:
    return replace(cfg, output_dir=output_dir)
