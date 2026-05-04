"""
100-seed benchmark: CRRP-RL constrained vs Q-learning vs Actor-critic
Generates all publication figures
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '/home/claude')
from crrp_rl_constrained_mismatch import CRRPRLConfig, run_simulation

OUTDIR = Path("/home/claude/crrp_rl_constrained_outputs")
OUTDIR.mkdir(exist_ok=True)
N_SEEDS = 100
N_TRIALS = 700
WINDOW = 50

# ── Baselines ──────────────────────────────────────────────────────────────

def run_qlearning(seed, n_trials=N_TRIALS, alpha=0.35, temp=3.5):
    rng = np.random.default_rng(seed)
    Q = np.zeros((2, 2))
    acc, rho_rec = [], []
    for t in range(n_trials):
        cue = rng.choice([-1.0, 1.0])
        s = 1 if cue > 0 else 0
        correct = s
        logits = temp * Q[s]
        logits -= logits.max()
        p = np.exp(logits) / np.exp(logits).sum()
        a = rng.choice([0, 1], p=p)
        r = float(a == correct)
        Q[s, a] += alpha * (r - Q[s, a])
        acc.append(r)
        rho_rec.append(0.0)
    return np.array(acc), np.array(rho_rec)

def run_actor_critic(seed, n_trials=N_TRIALS, alpha_v=0.12, alpha_a=0.07, temp=3.0):
    rng = np.random.default_rng(seed)
    V = np.zeros(2)
    theta = np.zeros((2, 2))
    acc = []
    for t in range(n_trials):
        cue = rng.choice([-1.0, 1.0])
        s = 1 if cue > 0 else 0
        correct = s
        logits = temp * theta[s]
        logits -= logits.max()
        p = np.exp(logits) / np.exp(logits).sum()
        a = rng.choice([0, 1], p=p)
        r = float(a == correct)
        delta = r - V[s]
        V[s] += alpha_v * delta
        theta[s, a] += alpha_a * delta
        acc.append(r)
    return np.array(acc)

# ── Run all seeds ──────────────────────────────────────────────────────────

print("Running 100-seed benchmark...")
crrp_acc = np.zeros((N_SEEDS, N_TRIALS))
crrp_rho_tha = np.zeros((N_SEEDS, N_TRIALS))
crrp_rho_bg  = np.zeros((N_SEEDS, N_TRIALS))
crrp_rho_cb  = np.zeros((N_SEEDS, N_TRIALS))
crrp_emix    = np.zeros((N_SEEDS, N_TRIALS))
crrp_etha    = np.zeros((N_SEEDS, N_TRIALS))
crrp_ecb     = np.zeros((N_SEEDS, N_TRIALS))

ql_acc  = np.zeros((N_SEEDS, N_TRIALS))
ac_acc  = np.zeros((N_SEEDS, N_TRIALS))

for i in range(N_SEEDS):
    cfg = CRRPRLConfig(random_seed=i, n_trials=N_TRIALS)
    df = run_simulation(cfg)
    crrp_acc[i]     = df['reward'].values
    crrp_rho_tha[i] = df['rho_tha'].values
    crrp_rho_bg[i]  = df['rho_bg'].values
    crrp_rho_cb[i]  = df['rho_cb'].values
    crrp_emix[i]    = df['e_mix'].values
    crrp_etha[i]    = df['e_tha'].values
    crrp_ecb[i]     = df['e_cb'].values
    ql_acc[i],_     = run_qlearning(i)
    ac_acc[i]       = run_actor_critic(i)
    if (i+1) % 20 == 0:
        print(f"  {i+1}/100 done")

print("All seeds complete.")

# rolling helpers
def rolling(arr, w=WINDOW):
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        s = pd.Series(arr[i]).rolling(w, min_periods=1).mean().values
        out[i] = s
    return out

crrp_roll = rolling(crrp_acc)
ql_roll   = rolling(ql_acc)
ac_roll   = rolling(ac_acc)
rho_tha_roll = rolling(crrp_rho_tha)
rho_bg_roll  = rolling(crrp_rho_bg)
rho_cb_roll  = rolling(crrp_rho_cb)
emix_roll    = rolling(crrp_emix)
etha_roll    = rolling(crrp_etha)
ecb_roll     = rolling(crrp_ecb)

trials = np.arange(1, N_TRIALS+1)

# ── Summary table ──────────────────────────────────────────────────────────
def summ(arr, n=100):
    return arr[:, :n].mean(axis=1).mean(), arr[:, :n].mean(axis=1).std(), \
           arr[:, -n:].mean(axis=1).mean(), arr[:, -n:].mean(axis=1).std(), \
           arr.mean(axis=1).mean(), arr.mean(axis=1).std()

rows = []
for name, arr in [("CRRP-RL constrained", crrp_acc), ("Q-learning", ql_acc), ("Actor-critic", ac_acc)]:
    m1,s1,m2,s2,mo,so = summ(arr)
    rows.append({"Model": name,
                 "First 100": f"{m1:.3f} ± {s1:.3f}",
                 "Last 100":  f"{m2:.3f} ± {s2:.3f}",
                 "Overall":   f"{mo:.3f} ± {so:.3f}"})
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUTDIR / "benchmark_summary.csv", index=False)
print("\nBenchmark summary:")
print(summary_df.to_string(index=False))

# ── FIGURE 1: Learning curves ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
def plot_band(ax, arr, label, color):
    m = arr.mean(0); s = arr.std(0)/np.sqrt(N_SEEDS)
    ax.plot(trials, m, label=label, color=color, lw=2)
    ax.fill_between(trials, m-s, m+s, alpha=0.15, color=color)

plot_band(ax, crrp_roll, "CRRP-RL constrained", "#1f77b4")
plot_band(ax, ql_roll,   "Q-learning",           "#ff7f0e")
plot_band(ax, ac_roll,   "Actor-critic",          "#2ca02c")
ax.set_xlabel("Trial", fontsize=12)
ax.set_ylabel("Rolling accuracy, window=50", fontsize=12)
ax.set_title("Constrained CRRP-RL vs standard RL baselines")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_learning_curves.png", dpi=220)
plt.close()

# ── FIGURE 2: Early vs late bar chart ─────────────────────────────────────
models = ["Actor-critic", "CRRP-RL constrained", "Q-learning"]
arrs   = [ac_acc, crrp_acc, ql_acc]
early_m = [a[:, :100].mean(1).mean() for a in arrs]
early_s = [a[:, :100].mean(1).std()  for a in arrs]
late_m  = [a[:, -100:].mean(1).mean() for a in arrs]
late_s  = [a[:, -100:].mean(1).std()  for a in arrs]

x = np.arange(len(models)); w = 0.35
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x-w/2, early_m, w, yerr=early_s, label="First 100", color="#1f77b4", capsize=4)
ax.bar(x+w/2, late_m,  w, yerr=late_s,  label="Last 100",  color="#ff7f0e", capsize=4)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Accuracy", fontsize=12); ax.set_ylim(0, 1.05)
ax.set_title("Early vs late accuracy across 100 seeds")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_early_late_accuracy.png", dpi=220)
plt.close()

# ── FIGURE 3: Route weights across seeds ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
for arr, label, color in [
    (rho_tha_roll, "rho_tha cue/BB mismatch", "#1f77b4"),
    (rho_bg_roll,  "rho_bg reward/value",     "#ff7f0e"),
    (rho_cb_roll,  "rho_cb BB mismatch correction", "#2ca02c"),
]:
    m = arr.mean(0)
    ax.plot(trials, m, label=label, color=color, lw=2)
ax.set_xlabel("Trial", fontsize=12); ax.set_ylabel("Route weight", fontsize=12)
ax.set_title("Constrained CRRP-RL route allocation")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_route_weights.png", dpi=220)
plt.close()

# ── FIGURE 4: BB mismatch terms ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
for arr, label, color in [
    (emix_roll, "e_mix", "#1f77b4"),
    (etha_roll, "e_tha", "#ff7f0e"),
    (ecb_roll,  "e_cb",  "#2ca02c"),
]:
    m = arr.mean(0)
    ax.plot(trials, m, label=label, color=color, lw=1.8)
ax.set_xlabel("Trial", fontsize=12); ax.set_ylabel("Mean mismatch", fontsize=12)
ax.set_title("BB-reference mismatch terms across seeds")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTDIR / "benchmark_mismatch_terms.png", dpi=220)
plt.close()

print(f"\nAll figures saved to {OUTDIR.resolve()}")
