
"""
crrp_rl_constrained_mismatch.py

CRRP-RL constrained route-allocation version.

Main changes:
1. Route weights are explicitly constrained:
       rho_tha + rho_bg + rho_cb = 1, rho_k >= 0
   using softmax logits.

2. Broadband mismatch drives THA and CB route plasticity:
       e_mix = ||x_bb - x_mix||^2
       e_tha = ||x_bb - x_tha||^2
       e_cb  = ||x_bb - x_cb ||^2

   If a route is closer to the BB reference than the current mixture,
   its slow logit increases:
       z_tha <- z_tha + eta_tha_mismatch * (e_mix - e_tha)
       z_cb  <- z_cb  + eta_cb_mismatch  * (e_mix - e_cb)

3. BG route plasticity remains reward/RPE-driven.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class CRRPRLConfig:
    n_trials: int = 700
    random_seed: int = 11
    rolling_window: int = 50

    beta: float = 4.0
    eta_actor: float = 0.06

    eta_value: float = 0.10
    eta_bg_bias: float = 0.018

    eta_tha_mismatch: float = 0.012
    eta_cb_mismatch: float = 0.012

    bias_decay: float = 0.0015

    cue_gate_strength: float = 0.80
    bg_confidence_gate_strength: float = 1.15
    cb_uncertainty_gate_strength: float = 0.70

    initial_sensory_noise: float = 0.55
    final_sensory_noise: float = 0.25

    initial_actor_weight: float = 0.20
    initial_route_bias_tha: float = 0.00
    initial_route_bias_bg: float = -0.20
    initial_route_bias_cb: float = 0.00


def softmax(v: np.ndarray) -> np.ndarray:
    """Numerically stable softmax. This enforces rho_k >= 0 and sum_k rho_k = 1."""
    v = np.asarray(v, dtype=float)
    v = v - np.max(v)
    ev = np.exp(v)
    rho = ev / np.sum(ev)
    if not np.all(np.isfinite(rho)):
        raise FloatingPointError("Route weights contain non-finite values.")
    if abs(np.sum(rho) - 1.0) > 1e-10:
        raise FloatingPointError("Route-weight simplex constraint violated.")
    return rho


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_trial(rng: np.random.Generator) -> Tuple[float, int, int]:
    cue = float(rng.choice([-1.0, 1.0]))
    state = 1 if cue > 0 else 0
    correct_action = 1 if cue > 0 else 0
    return cue, state, correct_action


class CRRPRLAgent:
    def __init__(self, cfg: CRRPRLConfig):
        self.cfg = cfg
        self.z_bias = np.array(
            [
                cfg.initial_route_bias_tha,
                cfg.initial_route_bias_bg,
                cfg.initial_route_bias_cb,
            ],
            dtype=float,
        )
        self.w_actor = float(cfg.initial_actor_weight)
        self.v_bg_state = np.zeros(2, dtype=float)

    def sensory_noise(self, trial_index: int) -> float:
        frac = trial_index / max(self.cfg.n_trials - 1, 1)
        noise = (
            self.cfg.initial_sensory_noise
            + frac * (self.cfg.final_sensory_noise - self.cfg.initial_sensory_noise)
        )
        return float(max(noise, min(self.cfg.initial_sensory_noise, self.cfg.final_sensory_noise)))

    def observer_states(
        self,
        cue: float,
        state: int,
        noise: float,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        x_tha = 1.05 * cue + rng.normal(0, noise)

        bg_confidence_state = abs(self.v_bg_state[state])
        bg_gain = 0.25 + 1.10 * sigmoid(2.2 * bg_confidence_state)
        x_bg = bg_gain * cue + rng.normal(0, noise * 0.80)

        x_cb = 0.80 * cue + rng.normal(0, noise * 0.55)
        x_bb = 1.00 * cue + rng.normal(0, noise * 0.35)

        return {
            "x_tha": float(x_tha),
            "x_bg": float(x_bg),
            "x_cb": float(x_cb),
            "x_bb": float(x_bb),
        }

    def route_weights(
        self,
        *,
        x_tha: float,
        x_cb: float,
        state: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        cue_strength = abs(x_tha)
        bg_confidence = abs(self.v_bg_state[state])
        trial_uncertainty = abs(x_tha - x_cb)

        route_logits = self.z_bias.copy()
        route_logits[0] += self.cfg.cue_gate_strength * cue_strength
        route_logits[1] += self.cfg.bg_confidence_gate_strength * bg_confidence
        route_logits[2] += self.cfg.cb_uncertainty_gate_strength * trial_uncertainty

        rho = softmax(route_logits)

        signals = {
            "cue_strength": float(cue_strength),
            "bg_confidence": float(bg_confidence),
            "trial_uncertainty": float(trial_uncertainty),
            "logit_tha": float(route_logits[0]),
            "logit_bg": float(route_logits[1]),
            "logit_cb": float(route_logits[2]),
            "rho_sum": float(np.sum(rho)),
        }
        return rho, signals

    def act(self, x_mix: float, rng: np.random.Generator) -> Tuple[float, int]:
        p_action1 = float(sigmoid(self.cfg.beta * self.w_actor * x_mix))
        action = 1 if rng.random() < p_action1 else 0
        return p_action1, action

    def update(
        self,
        *,
        state: int,
        action: int,
        reward: float,
        correct_action: int,
        p_action1: float,
        x_mix: float,
        x_tha: float,
        x_bg: float,
        x_cb: float,
        x_bb: float,
        rho_bg: float,
    ) -> Dict[str, float]:
        value_before = self.v_bg_state[state]
        delta = reward - value_before
        self.v_bg_state[state] += self.cfg.eta_value * delta

        self.w_actor += self.cfg.eta_actor * delta * self.cfg.beta * x_mix * (action - p_action1)

        if correct_action == 1:
            signed_policy_error = reward - p_action1
        else:
            signed_policy_error = reward - (1.0 - p_action1)

        self.z_bias[1] += self.cfg.eta_bg_bias * delta * rho_bg * signed_policy_error
        self.z_bias[1] += 0.003 * abs(self.v_bg_state[state])

        e_mix = (x_bb - x_mix) ** 2
        e_tha = (x_bb - x_tha) ** 2
        e_cb = (x_bb - x_cb) ** 2

        dz_tha = self.cfg.eta_tha_mismatch * (e_mix - e_tha)
        dz_cb = self.cfg.eta_cb_mismatch * (e_mix - e_cb)

        self.z_bias[0] += dz_tha
        self.z_bias[2] += dz_cb
        self.z_bias *= 1.0 - self.cfg.bias_decay

        return {
            "rpe": float(delta),
            "value_before": float(value_before),
            "e_mix": float(e_mix),
            "e_tha": float(e_tha),
            "e_cb": float(e_cb),
            "delta_z_tha_mismatch": float(dz_tha),
            "delta_z_cb_mismatch": float(dz_cb),
        }


def run_simulation(cfg: CRRPRLConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    agent = CRRPRLAgent(cfg)
    records = []

    for t in range(cfg.n_trials):
        cue, state, correct_action = make_trial(rng)
        noise = agent.sensory_noise(t)

        states = agent.observer_states(cue, state, noise, rng)
        x_tha = states["x_tha"]
        x_bg = states["x_bg"]
        x_cb = states["x_cb"]
        x_bb = states["x_bb"]

        rho, route_signals = agent.route_weights(
            x_tha=x_tha,
            x_cb=x_cb,
            state=state,
        )
        rho_tha, rho_bg, rho_cb = rho

        x_mix = rho_tha * x_tha + rho_bg * x_bg + rho_cb * x_cb

        p_action1, action = agent.act(x_mix, rng)
        reward = 1.0 if action == correct_action else 0.0

        update_signals = agent.update(
            state=state,
            action=action,
            reward=reward,
            correct_action=correct_action,
            p_action1=p_action1,
            x_mix=x_mix,
            x_tha=x_tha,
            x_bg=x_bg,
            x_cb=x_cb,
            x_bb=x_bb,
            rho_bg=rho_bg,
        )

        records.append(
            {
                "trial": t + 1,
                "cue": cue,
                "state": state,
                "correct_action": correct_action,
                "action": action,
                "reward": reward,
                "p_action1": p_action1,
                "rho_tha": rho_tha,
                "rho_bg": rho_bg,
                "rho_cb": rho_cb,
                "rho_sum": rho_tha + rho_bg + rho_cb,
                "x_mix": x_mix,
                "w_actor": agent.w_actor,
                "v_bg_state": agent.v_bg_state[state],
                "z_bias_tha": agent.z_bias[0],
                "z_bias_bg": agent.z_bias[1],
                "z_bias_cb": agent.z_bias[2],
                **states,
                **route_signals,
                **update_signals,
            }
        )

    df = pd.DataFrame(records)
    w = cfg.rolling_window
    df["rolling_accuracy"] = df["reward"].rolling(w, min_periods=1).mean()
    df["rolling_abs_rpe"] = df["rpe"].abs().rolling(w, min_periods=1).mean()
    df["rolling_mismatch"] = df["e_mix"].rolling(w, min_periods=1).mean()
    return df


def summarize(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    early = df.iloc[:n]
    late = df.iloc[-n:]
    rows = [
        ("accuracy", early["reward"].mean(), late["reward"].mean()),
        ("mean_abs_rpe", early["rpe"].abs().mean(), late["rpe"].abs().mean()),
        ("mean_e_mix", early["e_mix"].mean(), late["e_mix"].mean()),
        ("rho_tha", early["rho_tha"].mean(), late["rho_tha"].mean()),
        ("rho_bg", early["rho_bg"].mean(), late["rho_bg"].mean()),
        ("rho_cb", early["rho_cb"].mean(), late["rho_cb"].mean()),
        ("rho_sum_error_max", abs(early["rho_sum"] - 1).max(), abs(late["rho_sum"] - 1).max()),
        ("w_actor", early["w_actor"].mean(), late["w_actor"].mean()),
        ("v_bg_state", early["v_bg_state"].mean(), late["v_bg_state"].mean()),
    ]
    return pd.DataFrame(rows, columns=["metric", f"first_{n}", f"last_{n}"])


def plot_results(df: pd.DataFrame, outdir: str | Path = "crrp_rl_constrained_outputs") -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(df["trial"], df["rolling_accuracy"])
    plt.ylim(0, 1.05)
    plt.xlabel("Trial")
    plt.ylabel("Rolling accuracy")
    plt.title("CRRP-RL constrained: cortical decision accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "01_constrained_learning_accuracy.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(df["trial"], df["rho_tha"], label="rho_tha cue/context + BB mismatch")
    plt.plot(df["trial"], df["rho_bg"], label="rho_bg reward/value")
    plt.plot(df["trial"], df["rho_cb"], label="rho_cb BB mismatch correction")
    plt.xlabel("Trial")
    plt.ylabel("Route weight")
    plt.title("Simplex-constrained route allocation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "02_constrained_route_weights.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(df["trial"], df["rolling_abs_rpe"], label="mean |RPE|")
    plt.plot(df["trial"], df["rolling_mismatch"], label="BB-reference mismatch e_mix")
    plt.xlabel("Trial")
    plt.ylabel("Rolling value")
    plt.title("Reward error and BB-reference mismatch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "03_constrained_errors.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(df["trial"], df["e_mix"], label="e_mix = |x_bb - x_mix|^2", alpha=0.8)
    plt.plot(df["trial"], df["e_tha"], label="e_tha = |x_bb - x_tha|^2", alpha=0.8)
    plt.plot(df["trial"], df["e_cb"], label="e_cb = |x_bb - x_cb|^2", alpha=0.8)
    plt.xlabel("Trial")
    plt.ylabel("Mismatch")
    plt.title("Route-specific mismatch to BB reference")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "04_route_specific_mismatch.png", dpi=220)
    plt.close()


def main() -> None:
    cfg = CRRPRLConfig()
    df = run_simulation(cfg)
    summary = summarize(df)

    outdir = Path("crrp_rl_constrained_outputs")
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / "crrp_rl_constrained_trials.csv", index=False)
    summary.to_csv(outdir / "crrp_rl_constrained_summary.csv", index=False)
    plot_results(df, outdir=outdir)

    print("CRRP-RL constrained mismatch simulation complete.")
    print("\nConfiguration:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    print("\nSummary:")
    print(summary.to_string(index=False))

    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
