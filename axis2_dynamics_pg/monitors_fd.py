"""FD ρ / κ_relabel monitor for the dynamics experiment.

Thin wrapper around perf_correction_fd.compute_rho_kappa_fd, with the same
"snapshot → refit → measure → restore" flow that Tier B's monitor uses, and
matching CSV/W&B logging conventions.
"""
from typing import Optional

import numpy as np


def compute_rho_kappa_fd_monitor(
    agent,
    reward_model,
    ds: int,
    device,
    pref_batch_size: int = 256,
    refit_epochs: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Wrapper that puts the actor and RM ensemble into eval mode before
    differentiating (matching Tier B monitor's pattern), and restores train
    mode after.  Returns the same dict as compute_rho_kappa_fd.
    """
    from axis2_dynamics_pg.perf_correction_fd import compute_rho_kappa_fd

    agent.actor.eval()
    for m in reward_model.ensemble:
        m.eval()
    try:
        out = compute_rho_kappa_fd(
            actor=agent.actor,
            reward_model=reward_model,
            ds=ds,
            device=device,
            pref_batch_size=pref_batch_size,
            refit_epochs=refit_epochs,
            rng=rng,
        )
    finally:
        agent.actor.train()
        for m in reward_model.ensemble:
            m.train()
    return out
