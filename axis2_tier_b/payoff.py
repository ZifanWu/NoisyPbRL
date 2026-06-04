"""
Axis 2 Tier B — H2.5 Payoff Test.

Loads a checkpoint saved by train_PEBBLE_axis2.py, applies each of three remedies in
isolation, continues training for K steps, and measures gold reward recovery (Δgold).

Usage:
    python -m axis2_tier_b.payoff \\
        --ckpt_dir /path/to/ckpt_500000 \\
        --k_steps 20000 \\
        --n_remedy_labels 1000 5000 \\
        --out_dir results/axis2_tier_b/payoff

The script produces:
  - payoff_results.json: per-remedy Δgold for this checkpoint
  - 6 scatter plots: each instrument × each remedy (3×2 grid saved as payoff_plots.png)

Remedy definitions:
  1. Relabel at θ: collect n_remedy_labels[0] fresh prefs with configured teacher eps
     (NOT eps=0), retrain same-arch RM, continue K steps.
  2. More clean labels: collect n_remedy_labels[1] prefs with eps=0 teacher,
     retrain same-arch RM, continue K steps.
  3. Bigger RM: replace RM with full 256×3 arch, retrain on all clean on-policy data,
     continue K steps.
"""

import os
import sys
import json
import pickle
import argparse
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_model import RewardModel
from axis2_tier_b.monitor import (
    _build_clean_pref_pairs,
    _build_teacher_pref_pairs,
    _train_fresh_rm,
)
import utils


# ── loading ───────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_dir: str):
    """
    Load agent, reward_model, replay_buffer, and meta from a checkpoint directory.

    The checkpoint was saved by train_PEBBLE_axis2._save_checkpoint().
    Returns (agent, reward_model, replay_buffer, meta_dict).
    """
    meta_path = os.path.join(ckpt_dir, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    cfg_dict = meta['cfg']
    step = meta['step']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Rebuild environment to get obs/act dims
    env_name = cfg_dict['env']
    if 'metaworld' in env_name:
        from types import SimpleNamespace
        env = utils.make_metaworld_env(SimpleNamespace(env=env_name, seed=cfg_dict['seed']))
    else:
        import dmc2gym
        if env_name == 'ball_in_cup_catch':
            domain_name, task_name = 'ball_in_cup', 'catch'
        else:
            parts = env_name.split('_')
            domain_name = parts[0]
            task_name = '_'.join(parts[1:])

        env = dmc2gym.make(domain_name=domain_name, task_name=task_name,
                           seed=cfg_dict['seed'], visualize_reward=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Rebuild reward model.  Mirror the original training config so the loaded RM has the same
    # teacher (BT-with-normalization) and labeling behaviour as ψ_cur at checkpoint time.
    rm = RewardModel(
        ds=obs_dim, da=act_dim,
        ensemble_size=5,
        size_segment=cfg_dict['segment'],
        hidden_dim=cfg_dict['rm_hidden_dim'],
        num_layers=cfg_dict['rm_num_layers'],
        output_activation=cfg_dict['rm_output_activation'],
        teacher_beta=cfg_dict.get('teacher_beta', -1),
        teacher_eps_mistake=cfg_dict['teacher_eps_mistake'],
        bt_normalize_by_gap_std=cfg_dict.get('bt_normalize_by_gap_std', False),
        bt_gap_ema_alpha=cfg_dict.get('bt_gap_ema_alpha', 0.9),
        buffer_window_rounds=cfg_dict.get('buffer_window_rounds') or None,
    )
    rm.load(ckpt_dir, step)   # restores ensemble weights AND inputs/targets/buffers/gap_std_ema

    # Rebuild and load agent
    import hydra
    from omegaconf import OmegaConf
    # We can't reload the full Hydra cfg, so we rebuild agent from scratch with same dims
    # and load state dicts directly.  Hidden sizes are stored by train_PEBBLE_axis2;
    # older checkpoints fall back to the repo defaults for their env family.
    from agent.actor import DiagGaussianActor
    from agent.critic import DoubleQCritic
    from agent.sac import SACAgent

    default_hidden_dim = 256 if 'metaworld' in env_name else 1024
    default_hidden_depth = 3 if 'metaworld' in env_name else 2
    actor_hidden_dim = cfg_dict.get('actor_hidden_dim', default_hidden_dim)
    actor_hidden_depth = cfg_dict.get('actor_hidden_depth', default_hidden_depth)
    critic_hidden_dim = cfg_dict.get('critic_hidden_dim', default_hidden_dim)
    critic_hidden_depth = cfg_dict.get('critic_hidden_depth', default_hidden_depth)

    actor = DiagGaussianActor(obs_dim=obs_dim, action_dim=act_dim,
                               hidden_depth=actor_hidden_depth, hidden_dim=actor_hidden_dim,
                               log_std_bounds=[-5, 2]).to(device)
    critic = DoubleQCritic(obs_dim=obs_dim, action_dim=act_dim,
                            hidden_dim=critic_hidden_dim, hidden_depth=critic_hidden_depth,
                            dormant_log_period=5000, dormant_threshold=0.1).to(device)

    actor.load_state_dict(
        torch.load(os.path.join(ckpt_dir, f'actor_{step}.pt'), map_location=device))
    critic.load_state_dict(
        torch.load(os.path.join(ckpt_dir, f'critic_{step}.pt'), map_location=device))

    # Load replay buffer
    with open(os.path.join(ckpt_dir, 'replay_buffer.pkl'), 'rb') as f:
        replay_buffer = pickle.load(f)

    return env, actor, critic, rm, replay_buffer, meta, device


# ── gold evaluation ────────────────────────────────────────────────────────────

def evaluate_gold(env, actor, n_episodes=10, device='cuda') -> float:
    """Return mean gold (true env) reward over n_episodes episodes."""
    actor.eval()
    total = 0.0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                dist = actor(obs_t)
                action = dist.mean.clamp(-1, 1).cpu().numpy()[0]
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += r
        total += ep_reward
    actor.train()
    return total / n_episodes


# ── remedy application ─────────────────────────────────────────────────────────

def _retrain_rm_on_fresh_prefs(rm_template, inputs, targets, seg_len, n_pairs,
                                teacher_eps, rng, device, max_epochs=300,
                                use_deployed_teacher=False):
    """Build fresh preference pairs and retrain a fresh RM."""
    if use_deployed_teacher:
        seg1, seg2, lbl = _build_teacher_pref_pairs(
            inputs, targets, seg_len, n_pairs, rng, rm_template=rm_template)
    elif teacher_eps > 0:
        seg1, seg2, lbl = _build_teacher_pref_pairs(
            inputs, targets, seg_len, n_pairs, rng,
            rm_template=None, teacher_eps_mistake=teacher_eps)
    else:
        seg1, seg2, lbl = _build_clean_pref_pairs(
            inputs, targets, seg_len, n_pairs, rng)

    if seg1 is None:
        print("  Warning: not enough on-policy data for remedy; skipping refit")
        return rm_template

    return _train_fresh_rm(rm_template, seg1, seg2, lbl,
                           max_epochs=max_epochs, device=device)


def _continue_training(env, actor, critic, rm, replay_buffer, k_steps, device) -> float:
    """
    Run k_steps of SAC training against rm and return the gold eval reward at the end.

    Uses a minimal SAC update loop (no Hydra, no logger).  This is intentionally
    lean — enough to measure Δgold, not to reproduce the full training fidelity.
    """
    from agent.sac import SACAgent

    # relabel replay buffer with new RM
    replay_buffer.relabel_with_predictor(rm)

    # minimal SAC update parameters
    actor_lr = 1e-4
    critic_lr = 1e-4
    alpha_lr = 1e-4
    batch_size = min(1024, len(replay_buffer))
    if batch_size < 64:
        return float('nan')

    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    log_alpha = torch.tensor(np.log(0.1), dtype=torch.float32,
                             requires_grad=True, device=device)
    alpha_opt = torch.optim.Adam([log_alpha], lr=alpha_lr)
    target_entropy = -env.action_space.shape[0]
    discount = 0.99
    tau = 0.005

    import copy
    critic_target = copy.deepcopy(critic).to(device)

    obs = env.reset()
    for step in range(k_steps):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            dist = actor(obs_t)
            action = dist.sample().clamp(-1, 1).cpu().numpy()[0]
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = float(terminated or truncated)

        rm_reward = rm.r_hat(np.concatenate([obs, action], axis=-1))
        replay_buffer.add(obs, action, rm_reward, next_obs, done,
                          0.0 if truncated and not terminated else done)
        obs = next_obs if not (terminated or truncated) else env.reset()

        # SAC critic update
        batch = replay_buffer.sample(batch_size)
        obs_b, act_b, rew_b, next_obs_b, not_done_b, _ = batch
        alpha = log_alpha.exp().detach()
        with torch.no_grad():
            dist_next = actor(next_obs_b)
            next_act = dist_next.rsample()
            next_lp = dist_next.log_prob(next_act).sum(-1, keepdim=True)
            q1_t, q2_t = critic_target(next_obs_b, next_act)
            q_target = rew_b + discount * not_done_b * (torch.min(q1_t, q2_t) - alpha * next_lp)
        q1, q2 = critic(obs_b, act_b)
        critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()
        critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

        # SAC actor update
        dist_b = actor(obs_b)
        act_new = dist_b.rsample()
        lp = dist_b.log_prob(act_new).sum(-1, keepdim=True)
        q1_a, q2_a = critic(obs_b, act_new)
        actor_loss = (alpha * lp - torch.min(q1_a, q2_a)).mean()
        actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

        # alpha update
        alpha_loss = (-log_alpha.exp() * (lp + target_entropy).detach()).mean()
        alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

        # soft update critic target
        if step % 2 == 0:
            for p, p_t in zip(critic.parameters(), critic_target.parameters()):
                p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

    return evaluate_gold(env, actor, n_episodes=10, device=device)


# ── main payoff test ──────────────────────────────────────────────────────────

def run_payoff(ckpt_dir: str, k_steps: int = 20000,
               n_labels_remedy1: int = 1000, n_labels_remedy2: int = 5000,
               out_dir: str = 'results/axis2_tier_b/payoff'):

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    print(f"\nLoading checkpoint: {ckpt_dir}")
    env, actor, critic, rm, replay_buffer, meta, device = load_checkpoint(ckpt_dir)
    cfg = meta['cfg']
    seg_len = cfg['segment']
    teacher_eps = cfg['teacher_eps_mistake']

    # baseline gold at checkpoint
    gold_baseline = evaluate_gold(env, actor, n_episodes=10, device=device)
    print(f"Checkpoint gold: {gold_baseline:.2f}")

    recent_inputs = rm.inputs[-50:] if rm.inputs else []
    recent_targets = rm.targets[-50:] if rm.targets else []

    results = {'ckpt_dir': ckpt_dir, 'gold_baseline': gold_baseline, 'remedies': {}}

    import copy

    for remedy_id, label in [(1, 'relabel'), (2, 'more_clean'), (3, 'bigger_rm')]:
        print(f"\nApplying remedy {remedy_id}: {label}")
        actor_r = copy.deepcopy(actor)
        critic_r = copy.deepcopy(critic)
        rb_r = copy.deepcopy(replay_buffer)
        rm_r = copy.deepcopy(rm)

        if remedy_id == 1:
            # Relabel at theta with the deployed teacher (BT/noise settings), NOT clean.
            rm_r = _retrain_rm_on_fresh_prefs(
                rm, recent_inputs, recent_targets, seg_len,
                n_pairs=n_labels_remedy1,
                teacher_eps=teacher_eps,
                rng=rng, device=str(device),
                use_deployed_teacher=True)

        elif remedy_id == 2:
            # More clean labels (eps=0)
            rm_r = _retrain_rm_on_fresh_prefs(
                rm, recent_inputs, recent_targets, seg_len,
                n_pairs=n_labels_remedy2,
                teacher_eps=0.0,            # clean labels
                rng=rng, device=str(device))

        elif remedy_id == 3:
            # Bigger RM: full 256×3 architecture, clean labels, all available data
            rm_full_template = RewardModel(
                ds=rm.ds, da=rm.da,
                ensemble_size=rm.de,
                size_segment=seg_len,
                hidden_dim=256, num_layers=3, output_activation='tanh',
                teacher_eps_mistake=teacher_eps,
            )
            n_all = max(len(recent_inputs) * (200 // seg_len), 500)
            rm_r = _retrain_rm_on_fresh_prefs(
                rm_full_template, recent_inputs, recent_targets, seg_len,
                n_pairs=n_all,
                teacher_eps=0.0,            # clean labels for bigger RM
                rng=rng, device=str(device), max_epochs=400)

        gold_after = _continue_training(env, actor_r, critic_r, rm_r, rb_r,
                                         k_steps=k_steps, device=device)
        delta_gold = gold_after - gold_baseline if not (gold_after != gold_after) else float('nan')
        results['remedies'][remedy_id] = {
            'label': label, 'gold_after': gold_after, 'delta_gold': delta_gold,
        }
        print(f"  Gold after: {gold_after:.2f}  Δgold: {delta_gold:.2f}")

    # save results
    out_path = os.path.join(out_dir, 'payoff_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    _plot_payoff(results, out_dir)
    return results


def _plot_payoff(results: dict, out_dir: str):
    """
    Generate 3×2 scatter grid: 3 instruments × remedies 1 and 2 (off-diagonal).
    Also plots 3 instruments × remedy 3 (matched-diagonal expectation).
    Total: 9 sub-plots (3 instruments × 3 remedies).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    remedies = results['remedies']
    remedy_labels = {1: 'Remedy 1\n(Relabel at θ)', 2: 'Remedy 2\n(More clean labels)',
                     3: 'Remedy 3\n(Bigger RM)'}
    delta_golds = {r: remedies[r]['delta_gold'] for r in remedies}

    # Payoff summary table (single-checkpoint, no instrument scatter yet)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    table_data = [['Remedy', 'Δgold']] + [
        [f"R{r}: {remedy_labels[r].replace(chr(10), ' ')}", f"{delta_golds[r]:.2f}"]
        for r in sorted(remedies)
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0], loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    ax.set_title(f"H2.5 Payoff — {os.path.basename(results['ckpt_dir'])}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'payoff_table.png'), dpi=120)
    plt.close()
    print(f"Saved payoff_table.png to {out_dir}")


# ── aggregate analysis across checkpoints ─────────────────────────────────────

def aggregate_payoff_results(results_root: str, out_dir: str):
    """
    Aggregate payoff_results.json across all conditions/seeds and plot
    instrument vs Δgold scatter (6 plots = 3 instruments × remedies 1 and 2 showing ≈0 for mis).
    Reads axis2_metrics.csv for instrument readings at the checkpoint step.
    """
    import glob
    import pandas as pd

    payoff_files = glob.glob(os.path.join(results_root, '**/payoff_results.json'),
                             recursive=True)
    records = []
    for pf in payoff_files:
        with open(pf) as f:
            pr = json.load(f)
        ckpt_dir = pr['ckpt_dir']
        # find axis2_metrics.csv in the run dir (parent of ckpt dir)
        run_dir = os.path.dirname(ckpt_dir)
        csv_path = os.path.join(run_dir, 'axis2_metrics.csv')
        if not os.path.exists(csv_path):
            continue
        df_m = pd.read_csv(csv_path)
        ckpt_step = int(os.path.basename(ckpt_dir).replace('ckpt_', ''))
        row_m = df_m[df_m['step'] == ckpt_step]
        if row_m.empty:
            row_m = df_m.iloc[[-1]]  # use last row if exact step not found

        instr = row_m.iloc[0]
        for r_id in [1, 2, 3]:
            r = pr['remedies'].get(str(r_id)) or pr['remedies'].get(r_id)
            if r is None:
                continue
            records.append({
                'kappa_hat': instr.get('kappa_hat', float('nan')),
                'ensemble_spread': instr.get('ensemble_spread', float('nan')),
                'gof_inf': instr.get('gof_inf', float('nan')),
                'remedy': r_id,
                'delta_gold': r['delta_gold'],
                'condition': _infer_condition(run_dir),
            })

    if not records:
        print("No payoff records found — run payoff.run_payoff() first")
        return

    df = pd.DataFrame(records)
    _plot_instrument_vs_delta_gold(df, out_dir)


def _infer_condition(run_dir: str) -> str:
    for cond in ['control', 'shift', 'epi', 'mis']:
        if cond in run_dir:
            return cond
    return 'unknown'


def _plot_instrument_vs_delta_gold(df, out_dir: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        return

    os.makedirs(out_dir, exist_ok=True)
    instruments = ['kappa_hat', 'ensemble_spread', 'gof_inf']
    instr_labels = ['κ̂', 'Ensemble spread', 'GoF (ψ_inf)']
    remedies = [1, 2, 3]
    remedy_titles = ['Remedy 1: Relabel', 'Remedy 2: More clean', 'Remedy 3: Bigger RM']
    colors = {'control': 'C0', 'shift': 'C1', 'epi': 'C2', 'mis': 'C3', 'unknown': 'C4'}

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ri, (r, rt) in enumerate(zip(remedies, remedy_titles)):
        sub = df[df['remedy'] == r]
        for ii, (inst, il) in enumerate(zip(instruments, instr_labels)):
            ax = axes[ii, ri]
            for cond, grp in sub.groupby('condition'):
                ax.scatter(grp[inst], grp['delta_gold'],
                           label=cond, color=colors.get(cond, 'grey'), alpha=0.7)
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.set_xlabel(il)
            ax.set_ylabel('Δgold' if ri == 0 else '')
            ax.set_title(rt if ii == 0 else '')
            if ri == 2 and ii == 0:
                ax.legend(fontsize=7)

    plt.suptitle('H2.5: Instrument vs Δgold per remedy\n'
                 '(off-diagonal entries should be ≈0 for mis condition)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'payoff_instrument_vs_delta_gold.png'), dpi=150)
    plt.close()
    print(f"Saved payoff_instrument_vs_delta_gold.png to {out_dir}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='H2.5 Payoff Test')
    parser.add_argument('--ckpt_dir', required=True,
                        help='Path to checkpoint directory (e.g. exp/.../ckpt_500000)')
    parser.add_argument('--k_steps', type=int, default=20000,
                        help='Steps of continued training per remedy')
    parser.add_argument('--n_remedy_labels', type=int, nargs=2, default=[1000, 5000],
                        metavar=('R1', 'R2'),
                        help='Label budgets for remedy 1 (relabel) and remedy 2 (more clean)')
    parser.add_argument('--out_dir', default='results/axis2_tier_b/payoff',
                        help='Output directory for results and plots')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate all payoff_results.json under --out_dir and plot')
    args = parser.parse_args()

    if args.aggregate:
        aggregate_payoff_results(args.ckpt_dir, args.out_dir)
    else:
        run_payoff(
            ckpt_dir=args.ckpt_dir,
            k_steps=args.k_steps,
            n_labels_remedy1=args.n_remedy_labels[0],
            n_labels_remedy2=args.n_remedy_labels[1],
            out_dir=args.out_dir,
        )
