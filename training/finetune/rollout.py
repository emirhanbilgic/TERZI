"""K-step Euler rollout of the clean-latent estimate for the FT-pen penalty (finetune_main.py / finetune_masked.py --pen-rollout K).

K = 1 is the old one-step estimate x0hat = x_t - t v. For K > 1 the CURRENT model integrates the flow from t to 0 on the
sampler's own time grid (TRELLIS.2 FlowEuler: t' = r s / (1 + (r - 1) s), r = rescale_t = 3 for the shape SLat), so the
penalty sees (nearly) what 12-step sampling would produce from this x_t. The first velocity is the one the caller already
computed (grad-live); every later step is a fresh forward, activation-checkpointed so memory stays ~one forward per step.
No CFG is needed: the shape-SLat sampler applies guidance only for t in [0.6, 1.0] and the penalty only acts for
t <= pen_tmax = 0.6.
grad = "all": back-propagate through every step (the penalty shapes the whole remaining trajectory);
grad = "last": earlier steps are detached, only the last velocity carries gradient (cheap ablation).
"""
import torch
from torch.utils.checkpoint import checkpoint


def time_grid(t, k, rescale=3.0):
    """k+1 times from t down to 0, spaced like the sampler's rescaled schedule over [0, t]."""
    s = torch.linspace(1.0, 0.0, k + 1).tolist()
    return [t * (rescale * x / (1 + (rescale - 1) * x)) for x in s]


def rollout_x0(vel, x_t, t, v0, k, rescale=3.0, grad="all"):
    """vel(feats, t) -> velocity (float tensor) of the current model; x_t, v0: feats at t and the velocity there.
    Returns the K-step Euler estimate of x_0."""
    if k <= 1:
        return x_t - t * v0
    ts = time_grid(t, k, rescale)
    x = x_t - (ts[0] - ts[1]) * v0
    for i in range(1, k):
        if grad == "last" and i < k - 1:
            with torch.no_grad():
                v = vel(x.detach(), ts[i])
            x = (x - (ts[i] - ts[i + 1]) * v).detach() if i < k - 1 else x - (ts[i] - ts[i + 1]) * v
            continue
        v = checkpoint(vel, x, ts[i], use_reentrant=False)
        x = x - (ts[i] - ts[i + 1]) * v
    return x
