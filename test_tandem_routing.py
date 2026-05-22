"""Unit tests for tandem data-routing logic.

No RL environment or GPU training needed.  Creates a synthetic HDF5 baseline
log with known data, then exercises every routing path (TandemLogger,
TandemReader, and the three new RewardModel sampling methods) and asserts the
SC2/SC3 invariants.

Run with:
    conda run -n bpref python test_tandem_routing.py
or
    conda run -n bpref pytest test_tandem_routing.py -v
"""

import os
import tempfile

import numpy as np
import pytest

from tandem_logger import TandemLogger
from tandem_reader import TandemReader
from reward_model import RewardModel
from sanity_checks import (
    assert_replay_matches_baseline,
    assert_pref_matches_baseline,
    assert_pref_differs_from_baseline,
    compare_rm_predictions,
)

# ── Synthetic problem dimensions ─────────────────────────────────────────────
OBS_DIM   = 6
ACT_DIM   = 3
SEG_SIZE  = 5
MAX_STEPS = 600     # 6 episodes × 100 steps
N_EPS     = 6
EP_LEN    = 100
N_EVENTS  = 2       # RM update events
PAIRS_PER_EVENT = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rng(seed=0):
    return np.random.default_rng(seed)


def _make_log(path: str):
    """Write a fully-specified synthetic baseline log to *path* and return the
    raw arrays so tests can compare against them."""
    rng = _rng(42)

    obs     = rng.standard_normal((MAX_STEPS, OBS_DIM)).astype(np.float32)
    actions = rng.standard_normal((MAX_STEPS, ACT_DIM)).astype(np.float32)
    rewards = rng.standard_normal(MAX_STEPS).astype(np.float32)
    next_obs = rng.standard_normal((MAX_STEPS, OBS_DIM)).astype(np.float32)
    dones   = np.zeros(MAX_STEPS, dtype=np.float32)

    # Episode boundaries: every EP_LEN steps
    for i in range(N_EPS):
        dones[(i + 1) * EP_LEN - 1] = 1.0
    dones_no_max = dones.copy()

    logger = TandemLogger(path, OBS_DIM, ACT_DIM, max_steps=MAX_STEPS)

    ep_id = 0
    ep_start = 0
    for t in range(MAX_STEPS):
        logger.log_transition(t, obs[t], actions[t], float(rewards[t]),
                              next_obs[t], float(dones[t]), float(dones_no_max[t]))
        if dones[t]:
            logger.log_episode_end(ep_id, ep_start, t - ep_start + 1)
            ep_id    += 1
            ep_start  = t + 1

    # Synthetic query events
    query_events = []
    for ev_idx in range(N_EVENTS):
        env_step = (ev_idx + 1) * (MAX_STEPS // (N_EVENTS + 1))
        # Episodes in pool: last 3 completed ones before this event
        ep_cutoff = env_step // EP_LEN
        pool_ids  = list(range(max(0, ep_cutoff - 3), ep_cutoff))

        # Segments: random windows into pool episodes
        sa1 = rng.standard_normal((PAIRS_PER_EVENT, SEG_SIZE, OBS_DIM + ACT_DIM)).astype(np.float32)
        sa2 = rng.standard_normal((PAIRS_PER_EVENT, SEG_SIZE, OBS_DIM + ACT_DIM)).astype(np.float32)
        r1  = rng.standard_normal((PAIRS_PER_EVENT, SEG_SIZE, 1)).astype(np.float32)
        r2  = rng.standard_normal((PAIRS_PER_EVENT, SEG_SIZE, 1)).astype(np.float32)
        labels = rng.integers(0, 2, size=(PAIRS_PER_EVENT, 1)).astype(np.float32)
        scores = rng.standard_normal(PAIRS_PER_EVENT).astype(np.float32)

        # Fake RM (random ensemble weights) — needs a live RewardModel to save checkpoints
        dummy_rm = RewardModel(OBS_DIM, ACT_DIM, ensemble_size=2,
                               size_segment=SEG_SIZE, activation='tanh', lr=3e-4,
                               mb_size=PAIRS_PER_EVENT, large_batch=1, capacity=100)

        logger.log_query_event_start(ev_idx, env_step, pool_ids, dummy_rm)
        logger.log_query_event_pairs(ev_idx, sa1, sa2, r1, r2, labels, scores)
        query_events.append(dict(env_step=env_step, pool_ids=pool_ids,
                                 sa1=sa1, sa2=sa2, r1=r1, r2=r2,
                                 labels=labels, scores=scores))

    logger.close()
    return dict(obs=obs, actions=actions, rewards=rewards,
                next_obs=next_obs, dones=dones, dones_no_max=dones_no_max,
                query_events=query_events)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def log_file():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    ground_truth = _make_log(path)
    yield path, ground_truth
    os.unlink(path)


@pytest.fixture(scope="module")
def reader(log_file):
    path, _ = log_file
    r = TandemReader(path)
    yield r
    r.close()


# ── Reader tests ──────────────────────────────────────────────────────────────

def test_reader_transition(log_file, reader):
    """TandemReader.get_transition returns exact stored values."""
    _, gt = log_file
    for step in [0, 50, 99, 200, MAX_STEPS - 1]:
        obs, act, rew, nobs, done, _ = reader.get_transition(step)
        assert np.allclose(obs,  gt["obs"][step],     atol=1e-6), f"obs mismatch at step {step}"
        assert np.allclose(act,  gt["actions"][step], atol=1e-6), f"action mismatch at step {step}"
        assert np.isclose(rew,   gt["rewards"][step], atol=1e-6), f"reward mismatch at step {step}"
        assert np.allclose(nobs, gt["next_obs"][step], atol=1e-6)
        assert np.isclose(done,  gt["dones"][step],   atol=1e-6)


def test_reader_num_events(log_file, reader):
    assert reader.num_query_events == N_EVENTS


def test_reader_query_event_pairs(log_file, reader):
    """Query event data round-trips through HDF5 exactly."""
    _, gt = log_file
    for ev_idx in range(N_EVENTS):
        ev  = reader.get_query_event(ev_idx)
        ref = gt["query_events"][ev_idx]
        assert np.allclose(ev["sa_t_1"], ref["sa1"], atol=1e-6), f"sa1 mismatch event {ev_idx}"
        assert np.allclose(ev["sa_t_2"], ref["sa2"], atol=1e-6)
        assert np.allclose(ev["labels"], ref["labels"], atol=1e-6)
        assert np.allclose(ev["disagree_scores"], ref["scores"], atol=1e-6)


def test_reader_episode_meta(log_file, reader):
    """Episode metadata is complete and self-consistent."""
    assert len(reader.episode_meta) == N_EPS
    for ep_id, (start, length) in reader.episode_meta.items():
        assert length == EP_LEN, f"ep {ep_id}: expected length {EP_LEN}, got {length}"
        assert start  == ep_id * EP_LEN, f"ep {ep_id}: expected start {ep_id*EP_LEN}, got {start}"


def test_reader_pool_reconstruction(log_file, reader):
    """reconstruct_episode_pool reproduces the correct (obs,action) content."""
    _, gt = log_file
    ev = reader.get_query_event(0)
    pool_ids = ev["pool_episode_ids"]
    inputs, targets = reader.reconstruct_episode_pool(pool_ids)

    assert len(inputs)  == len(pool_ids)
    assert len(targets) == len(pool_ids)
    for i, ep_id in enumerate(pool_ids):
        start = ep_id * EP_LEN
        expected_sa = np.concatenate(
            [gt["obs"][start:start+EP_LEN], gt["actions"][start:start+EP_LEN]], axis=-1)
        assert np.allclose(inputs[i], expected_sa, atol=1e-6), \
            f"pool ep {ep_id}: (obs,action) content mismatch"
        expected_r = gt["rewards"][start:start+EP_LEN].reshape(-1, 1)
        assert np.allclose(targets[i], expected_r, atol=1e-6), \
            f"pool ep {ep_id}: reward content mismatch"


def test_load_baseline_rm(log_file, reader):
    """load_baseline_rm changes weights in-place without error."""
    rm = RewardModel(OBS_DIM, ACT_DIM, ensemble_size=2, size_segment=SEG_SIZE,
                     activation='tanh', lr=3e-4, mb_size=PAIRS_PER_EVENT,
                     large_batch=1, capacity=100)
    weights_before = [m[-2].weight.data.clone() for m in rm.ensemble]
    reader.load_baseline_rm(0, rm)
    weights_after  = [m[-2].weight.data for m in rm.ensemble]
    # weights should have changed (baseline has different random init)
    changed = any(not (wb == wa).all()
                  for wb, wa in zip(weights_before, weights_after))
    assert changed, "load_baseline_rm did not change any weights"


# ── RewardModel routing-method tests ─────────────────────────────────────────

def _make_rm(seed=0):
    """Small deterministic RewardModel for routing tests."""
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)
    return RewardModel(OBS_DIM, ACT_DIM, ensemble_size=2, size_segment=SEG_SIZE,
                       activation='tanh', lr=3e-4,
                       mb_size=PAIRS_PER_EVENT, large_batch=2, capacity=200)


def _fill_rm_pool(rm, n_eps=4, rng=None):
    """Populate rm.inputs/targets with synthetic episodes."""
    if rng is None:
        rng = _rng(99)
    for ep in range(n_eps):
        for t in range(EP_LEN):
            obs = rng.standard_normal(OBS_DIM).astype(np.float32)
            act = rng.standard_normal(ACT_DIM).astype(np.float32)
            rew = float(rng.standard_normal())
            done = float(t == EP_LEN - 1)
            rm.add_data(obs, act, rew, done)


def test_episode_tracking_in_add_data():
    """add_data correctly tracks episode IDs and step counter."""
    rm = _make_rm(1)
    rng = _rng(1)
    step = 0
    for ep in range(3):
        for t in range(EP_LEN):
            done = float(t == EP_LEN - 1)
            rm.add_data(rng.standard_normal(OBS_DIM).astype(np.float32),
                        rng.standard_normal(ACT_DIM).astype(np.float32),
                        float(rng.standard_normal()), done)
            step += 1

    # After 3 complete episodes: inputs has 3 complete + 1 empty pending
    # episode IDs should be [0, 1, 2, 3]
    assert rm._ep_counter == 3,      f"expected ep_counter=3, got {rm._ep_counter}"
    assert rm._global_step == step,  f"global step mismatch"
    assert len(rm.input_episode_ids) == len(rm.inputs), \
        "input_episode_ids length must match inputs length"


def test_disagreement_sampling_with_details():
    """disagreement_sampling_with_details returns same number of pairs as disagreement_sampling."""
    rm = _make_rm(2)
    _fill_rm_pool(rm, n_eps=4)
    n1 = rm.disagreement_sampling()
    # Reset preference buffer
    rm.buffer_index = 0; rm.buffer_full = False

    n2, sa1, sa2, r1, r2, labels, scores = rm.disagreement_sampling_with_details()
    assert n1 == n2, "pair count mismatch between normal and _with_details"
    if n2 > 0:
        assert sa1.shape[0] == n2
        assert sa2.shape[0] == n2
        assert len(labels)  == n2
        assert len(scores)  == n2


def test_external_scorer_uses_different_rm():
    """disagreement_sampling_external_scorer uses scorer_rm, not self, to rank pairs.

    If scorer_rm has different weights from self, the selected pair indices
    will differ (with high probability on random data).
    """
    rm      = _make_rm(seed=3)
    scorer  = _make_rm(seed=77)   # deliberately different weights
    _fill_rm_pool(rm,     n_eps=5, rng=_rng(3))
    _fill_rm_pool(scorer, n_eps=5, rng=_rng(3))  # same pool, different RM

    # Run normal disagreement (self-scored) and record which pairs end up in buffer
    import copy
    rm_copy = copy.deepcopy(rm)
    rm_copy.disagreement_sampling()
    self_seg1 = rm_copy.buffer_seg1[:rm_copy.buffer_index].copy()

    # Run external-scorer variant
    rm.disagreement_sampling_external_scorer(scorer)
    ext_seg1  = rm.buffer_seg1[:rm.buffer_index].copy()

    # With different scorer weights, selected pairs should differ
    # (they sample from the same pool but score differently)
    # This is probabilistic — seed chosen so they differ
    if rm.buffer_index > 0 and rm_copy.buffer_index > 0:
        n = min(rm.buffer_index, rm_copy.buffer_index)
        differ = not np.allclose(self_seg1[:n], ext_seg1[:n], atol=1e-6)
        assert differ, (
            "external scorer selected identical pairs as self — "
            "either weights are the same (unlikely) or the scorer is not being used")


def test_external_pool_uses_different_episodes():
    """disagreement_sampling_external_pool queries from ext_inputs, not self.inputs.

    After the swap-and-restore, self.inputs must be unchanged.
    """
    rm = _make_rm(seed=4)
    _fill_rm_pool(rm, n_eps=4, rng=_rng(4))

    # Build an external pool from completely different data
    ext_rng = _rng(999)
    ext_inputs  = [ext_rng.standard_normal((EP_LEN, OBS_DIM + ACT_DIM)).astype(np.float32)
                   for _ in range(3)]
    ext_targets = [ext_rng.standard_normal((EP_LEN, 1)).astype(np.float32)
                   for _ in range(3)]

    orig_inputs_id  = id(rm.inputs[0])   # identity of first episode array
    orig_pool_len   = len(rm.inputs)

    rm.disagreement_sampling_external_pool(ext_inputs, ext_targets)

    # Pool must be restored
    assert len(rm.inputs) == orig_pool_len, "inputs list length changed after external_pool call"
    assert id(rm.inputs[0]) == orig_inputs_id, "inputs[0] identity changed — pool not restored"


def test_external_pool_self_inputs_unchanged():
    """After disagreement_sampling_external_pool, rm.inputs content is bit-identical."""
    rm = _make_rm(seed=5)
    _fill_rm_pool(rm, n_eps=3, rng=_rng(5))
    saved = [ep.copy() for ep in rm.inputs if len(ep) > 0]

    ext_inputs  = [np.ones((EP_LEN, OBS_DIM + ACT_DIM), dtype=np.float32)]
    ext_targets = [np.ones((EP_LEN, 1), dtype=np.float32)]
    rm.disagreement_sampling_external_pool(ext_inputs, ext_targets)

    for i, (orig, curr) in enumerate(zip(saved, rm.inputs)):
        if len(curr) > 0:
            assert np.array_equal(orig, curr), f"inputs[{i}] content changed"


# ── SC3 assertion-function tests ─────────────────────────────────────────────

class _FakeReplayBuffer:
    """Minimal stub that mimics the fields checked by assert_replay_matches_baseline."""
    def __init__(self, obs, actions, next_obs, dones):
        self.obses      = obs
        self.actions    = actions
        self.next_obses = next_obs
        self.not_dones  = (1.0 - dones).reshape(-1, 1)  # matches ReplayBuffer shape (N, 1)
        self.idx        = len(obs)


def test_assert_replay_matches_baseline_pass(log_file, reader):
    """SC2: replay buffer holding baseline data passes the check."""
    _, gt = log_file
    n = 300
    rb = _FakeReplayBuffer(gt["obs"][:n], gt["actions"][:n],
                           gt["next_obs"][:n], gt["dones"][:n])
    assert_replay_matches_baseline(reader, rb, n)  # must not raise


def test_assert_replay_matches_baseline_fail(log_file, reader):
    """SC2: replay buffer with corrupted obs raises AssertionError."""
    _, gt = log_file
    n = 300
    bad_obs = gt["obs"][:n] + 100.0   # intentionally wrong
    rb = _FakeReplayBuffer(bad_obs, gt["actions"][:n],
                           gt["next_obs"][:n], gt["dones"][:n])
    with pytest.raises(AssertionError, match="replay obs mismatch"):
        assert_replay_matches_baseline(reader, rb, n)


def test_assert_pref_matches_baseline_pass(log_file, reader):
    """SC3-iii: preference buffer holding baseline pairs passes."""
    _, gt = log_file
    rm = _make_rm(6)
    for ev_idx in range(N_EVENTS):
        ev = gt["query_events"][ev_idx]
        rm.put_queries(ev["sa1"], ev["sa2"], ev["labels"])
    assert_pref_matches_baseline(rm, reader, N_EVENTS)   # must not raise


def test_assert_pref_matches_baseline_fail(log_file, reader):
    """SC3-iii: preference buffer with wrong pairs raises AssertionError."""
    _, gt = log_file
    rm = _make_rm(7)
    rng = _rng(7)
    fake_sa1 = rng.standard_normal((PAIRS_PER_EVENT, SEG_SIZE, OBS_DIM + ACT_DIM)).astype(np.float32)
    fake_sa2 = rng.standard_normal((PAIRS_PER_EVENT, SEG_SIZE, OBS_DIM + ACT_DIM)).astype(np.float32)
    fake_lbl = rng.integers(0, 2, size=(PAIRS_PER_EVENT, 1)).astype(np.float32)
    rm.put_queries(fake_sa1, fake_sa2, fake_lbl)
    with pytest.raises(AssertionError, match="preference seg1 mismatch"):
        assert_pref_matches_baseline(rm, reader, 1)


def test_assert_pref_differs_from_baseline_pass(log_file, reader):
    """SC3-ii: preference buffer with different data passes the 'differs' check."""
    _, gt = log_file
    rm = _make_rm(8)
    rng = _rng(8)
    # Inject clearly different data
    diff_sa1 = (rng.standard_normal((PAIRS_PER_EVENT, SEG_SIZE, OBS_DIM + ACT_DIM)) * 100).astype(np.float32)
    diff_sa2 = (rng.standard_normal((PAIRS_PER_EVENT, SEG_SIZE, OBS_DIM + ACT_DIM)) * 100).astype(np.float32)
    diff_lbl = rng.integers(0, 2, size=(PAIRS_PER_EVENT, 1)).astype(np.float32)
    rm.put_queries(diff_sa1, diff_sa2, diff_lbl)
    assert_pref_differs_from_baseline(rm, reader, 0)  # must not raise


def test_assert_pref_differs_from_baseline_fail(log_file, reader):
    """SC3-ii: preference buffer holding IDENTICAL data fails the 'differs' check."""
    _, gt = log_file
    rm = _make_rm(9)
    ev = gt["query_events"][0]
    rm.put_queries(ev["sa1"], ev["sa2"], ev["labels"])   # same as baseline → should fail
    with pytest.raises(AssertionError, match="identical to baseline"):
        assert_pref_differs_from_baseline(rm, reader, 0)


# ── SC1  RM prediction consistency ───────────────────────────────────────────

def test_compare_rm_predictions_same_seed():
    """SC1: two RMs with same seed predict identically on fresh data."""
    rm_a = _make_rm(seed=10)
    rm_b = _make_rm(seed=10)
    rng  = _rng(10)
    obs  = rng.standard_normal((32, OBS_DIM)).astype(np.float32)
    act  = rng.standard_normal((32, ACT_DIM)).astype(np.float32)
    max_diff = compare_rm_predictions(rm_a, rm_b, obs, act)
    assert max_diff == 0.0, f"same-seed RMs differ: max_diff={max_diff}"


def test_compare_rm_predictions_different_seed():
    """SC1 inverse: different seeds produce different predictions (sanity on the checker)."""
    rm_a = _make_rm(seed=11)
    rm_b = _make_rm(seed=99)
    rng  = _rng(11)
    obs  = rng.standard_normal((32, OBS_DIM)).astype(np.float32)
    act  = rng.standard_normal((32, ACT_DIM)).astype(np.float32)
    with pytest.raises(AssertionError):
        compare_rm_predictions(rm_a, rm_b, obs, act)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
