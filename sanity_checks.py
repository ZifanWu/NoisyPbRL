"""Tandem-experiment sanity checks.

Three families of checks, matching the advice in the paper plan:

SC1  identical-init replay
     If the tandem is initialized with the same seed as the baseline and run in
     all_passive with the baseline RM (instead of its own), gradients and RM
     predictions must be bit-identical.  Not a runtime check — this is validated
     by compare_rm_predictions().

SC2  independent-init data-stream integrity
     The consumed (obs, action) stream in replay_buffer must be exactly the
     baseline stream (for passive-policy conditions).

SC3  condition-specific invariants
     One quantity that MUST match baseline, one that MUST differ, per condition.
"""

import numpy as np


# ── tolerances ───────────────────────────────────────────────────────────────
_ABS = 1e-5   # for float32 arrays coming off GPU, relabelled with RM
_REL = 1e-4


# ────────────────────────────────────────────────────────────────────────────
# SC2 / SC3-i,ii  replay buffer contains baseline transitions
# ────────────────────────────────────────────────────────────────────────────

def assert_replay_matches_baseline(reader, replay_buffer, up_to_step: int,
                                   label: str = ""):
    """Assert that replay_buffer[0:up_to_step] == baseline_log[0:up_to_step].

    Checks obs, actions, next_obs, and dones.  Does NOT check rewards because
    those are relabelled by the tandem RM and will differ from the logged
    env_rewards.

    Called from the main loop under `sanity_mode=true` for conditions (i) and
    (ii) after every few thousand steps.
    """
    rb  = reader.f["replay_buffer"]
    idx = min(up_to_step, replay_buffer.idx)
    if idx == 0:
        return

    bl_obs     = rb["obs"][:idx]
    bl_act     = rb["actions"][:idx]
    bl_nobs    = rb["next_obs"][:idx]
    bl_dones   = rb["dones"][:idx]

    ta_obs     = replay_buffer.obses[:idx]
    ta_act     = replay_buffer.actions[:idx]
    ta_nobs    = replay_buffer.next_obses[:idx]
    # not_dones is shape (N, 1); squeeze to (N,) for element-wise comparison
    ta_dones   = replay_buffer.not_dones[:idx].squeeze(-1)

    prefix = f"[{label}] " if label else ""
    if not np.allclose(bl_obs, ta_obs, atol=_ABS):
        bad = np.where(~np.isclose(bl_obs, ta_obs, atol=_ABS))
        raise AssertionError(
            f"{prefix}replay obs mismatch at indices {bad[0][:5]} "
            f"(max diff {np.abs(bl_obs - ta_obs).max():.3e})")

    if not np.allclose(bl_act, ta_act, atol=_ABS):
        raise AssertionError(
            f"{prefix}replay action mismatch "
            f"(max diff {np.abs(bl_act - ta_act).max():.3e})")

    if not np.allclose(bl_nobs, ta_nobs, atol=_ABS):
        raise AssertionError(
            f"{prefix}replay next_obs mismatch "
            f"(max diff {np.abs(bl_nobs - ta_nobs).max():.3e})")

    bl_not_done = (1.0 - bl_dones)
    if not np.allclose(bl_not_done, ta_dones, atol=_ABS):
        bad_idx = np.where(~np.isclose(bl_not_done, ta_dones, atol=_ABS))[0]
        raise AssertionError(
            f"{prefix}replay done flags mismatch at steps {bad_idx[:5].tolist()} "
            f"(baseline not_done={bl_not_done[bad_idx[:5]]}, "
            f"tandem not_done={ta_dones[bad_idx[:5]]})")


# ────────────────────────────────────────────────────────────────────────────
# SC3-iii  preference buffer identical to baseline (passive RM conditions)
# ────────────────────────────────────────────────────────────────────────────

def assert_pref_matches_baseline(reward_model, reader, up_to_event: int,
                                  label: str = ""):
    """Assert that every preference pair in reward_model matches the baseline log.

    For conditions (i) all_passive and (iii) active_pol_passive_rm: the tandem
    RM trains on baseline pairs injected directly via put_queries(), so the
    preference buffer must be identical to the baseline's.

    Checks that reward_model.buffer_seg1/2/label up to buffer_index match the
    concatenated pairs from query events 0..up_to_event-1.
    """
    prefix = f"[{label}] " if label else ""
    if up_to_event == 0:
        return

    # Reconstruct what the preference buffer should contain
    seg1_chunks, seg2_chunks, label_chunks = [], [], []
    for ev_idx in range(up_to_event):
        ev = reader.get_query_event(ev_idx)
        if len(ev["labels"]) > 0:
            seg1_chunks.append(ev["sa_t_1"])
            seg2_chunks.append(ev["sa_t_2"])
            label_chunks.append(ev["labels"])

    if not seg1_chunks:
        return

    expected_seg1  = np.concatenate(seg1_chunks, axis=0)
    expected_seg2  = np.concatenate(seg2_chunks, axis=0)
    expected_labels = np.concatenate(label_chunks, axis=0)
    n = expected_seg1.shape[0]

    actual_seg1   = reward_model.buffer_seg1[:n]
    actual_seg2   = reward_model.buffer_seg2[:n]
    actual_labels = reward_model.buffer_label[:n]

    if not np.allclose(expected_seg1, actual_seg1, atol=_ABS):
        raise AssertionError(
            f"{prefix}preference seg1 mismatch after {up_to_event} events "
            f"(max diff {np.abs(expected_seg1 - actual_seg1).max():.3e})")

    if not np.allclose(expected_seg2, actual_seg2, atol=_ABS):
        raise AssertionError(
            f"{prefix}preference seg2 mismatch "
            f"(max diff {np.abs(expected_seg2 - actual_seg2).max():.3e})")

    if not np.allclose(expected_labels, actual_labels, atol=_ABS):
        raise AssertionError(f"{prefix}preference labels mismatch")


# ────────────────────────────────────────────────────────────────────────────
# SC3-ii  preference buffer DIFFERS from baseline (active RM conditions)
# ────────────────────────────────────────────────────────────────────────────

def assert_pref_differs_from_baseline(reward_model, reader, event_idx: int,
                                       label: str = ""):
    """Assert that the pairs added at event_idx do NOT match the baseline's pairs.

    Compares the TAIL of the preference buffer (the most recently added n pairs)
    against the baseline's pairs for that event.  Checking the tail rather than
    the head is necessary because earlier events may have used baseline pairs
    (e.g. first_flag=1), so the head of the buffer is legitimately identical to
    the baseline and must not be checked here.

    A soft check: skipped if the buffer does not yet contain at least n pairs.
    """
    prefix = f"[{label}] " if label else ""
    ev = reader.get_query_event(event_idx)
    n  = len(ev["labels"])
    if n == 0 or reward_model.buffer_index < n:
        return

    # Most recently added n pairs sit at [buffer_index-n : buffer_index]
    tail_start    = reward_model.buffer_index - n
    actual_seg1   = reward_model.buffer_seg1[tail_start : reward_model.buffer_index]
    baseline_seg1 = ev["sa_t_1"][:n]

    max_diff = np.abs(actual_seg1 - baseline_seg1).max()
    if max_diff < _ABS:
        raise AssertionError(
            f"{prefix}pairs added at event {event_idx} are identical to baseline "
            f"(max diff {max_diff:.3e}) — routing may be injecting baseline pairs "
            f"instead of tandem segments")


# ────────────────────────────────────────────────────────────────────────────
# SC1  RM prediction consistency
# ────────────────────────────────────────────────────────────────────────────

def compare_rm_predictions(rm_a, rm_b, obs_batch, action_batch, label: str = ""):
    """Assert that two RM instances predict identically on the given batch.

    SC1 usage: rm_a = baseline RM (loaded from checkpoint), rm_b = tandem RM
    (same seed, trained on same pairs).  If the routing is correct and seeds
    match, predictions should agree to within floating-point precision.

    Returns max absolute difference (even if within tolerance) for logging.
    """
    import numpy as np
    import torch
    sa = np.concatenate([obs_batch, action_batch], axis=-1)
    pred_a = rm_a.r_hat_batch(sa)
    pred_b = rm_b.r_hat_batch(sa)
    max_diff = float(np.abs(pred_a - pred_b).max())
    prefix = f"[{label}] " if label else ""
    if max_diff > _ABS:
        raise AssertionError(
            f"{prefix}RM prediction mismatch (max diff {max_diff:.3e}); "
            f"check that both RMs were initialized with the same seed and "
            f"trained on identical preference pairs")
    return max_diff


# ────────────────────────────────────────────────────────────────────────────
# Convenience: run all applicable checks for a given condition
# ────────────────────────────────────────────────────────────────────────────

def run_checks(mode: str, reader, reward_model, replay_buffer,
               step: int, rm_update_idx: int, check_interval: int = 2000):
    """Dispatch the right assertions for the current tandem_mode.

    Call this at the end of every `check_interval` steps in the training loop
    when `sanity_mode=true`.  Each check prints a one-line OK/FAIL summary.
    """
    if step % check_interval != 0 or step == 0:
        return

    results = []

    if mode in ("all_passive", "passive_pol_active_rm"):
        try:
            assert_replay_matches_baseline(reader, replay_buffer, step, label=mode)
            results.append(f"  OK  replay_matches_baseline  (step {step})")
        except AssertionError as e:
            results.append(f"  FAIL replay_matches_baseline: {e}")

    if mode in ("all_passive", "active_pol_passive_rm"):
        try:
            assert_pref_matches_baseline(reward_model, reader, rm_update_idx, label=mode)
            results.append(f"  OK  pref_matches_baseline  (events 0..{rm_update_idx-1})")
        except AssertionError as e:
            results.append(f"  FAIL pref_matches_baseline: {e}")

    if mode in ("passive_pol_active_rm", "active_pol_passive_query", "active_pol_passive_dist"):
        # Event 0 is seeded with baseline pairs (first_flag=1) for all these conditions;
        # only start checking from event 1 onwards.
        if rm_update_idx > 1:
            try:
                assert_pref_differs_from_baseline(
                    reward_model, reader, rm_update_idx - 1, label=mode)
                results.append(f"  OK  pref_differs_from_baseline  (event {rm_update_idx-1})")
            except AssertionError as e:
                results.append(f"  FAIL pref_differs_from_baseline: {e}")

    for line in results:
        print(f"[sanity] {line}")
