"""Label-free calibration: per-window local divergence kappa_k(b) and downstream influence iota_k, from a few unlabeled prompts.

  p_fp is the full-precision reference model (`fp_model`); p_b the quantized model with b-bit activations.
  kappa_k(b) = mean over prompts of sum_{t in window k} sum_{j committed at t} KL(p_fp(.|x_t)_j || p_b(.|x_t)_j)
               (every state x_t of the full-precision trajectory replayed at bit-width b; the state is held fixed)
  iota_k     = mean over prompts of sum_{j still masked after window k} KL(p_fp(.|x_after)_j || p_fp(.|x_before)_j)   (same states)
  g_k(b)     = iota_k * [kappa_k(b0) - kappa_k(b)]   (predicted gain of moving window k from b0 to b)
Only the rows that enter a KL are kept (committed positions, still-masked positions), so memory stays at one batch of logits. When the
sampler bans end-of-sequence (eos="ban"), the KLs are taken over the vocabulary without that column, the support the sampler uses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .sampler import EOS_ID, MASK_ID, generate


def _kl(lp, lq):   # KL(p || q) per row from log-probabilities (clamped at 0: float rounding can give -1e-6 when p == q)
    return (lp.exp() * (lp - lq)).sum(-1).clamp_min(0)


@torch.no_grad()
def _gen_logits(model, prompt, states, eos="last"):
    """logits of the generation region for a batch of states [n, G] -> [n, G, V] (model dtype, on the model's device); with eos="ban"
    the end-of-sequence column is removed."""
    dev = next(model.parameters()).device; S = states.to(dev)
    lg = model(torch.cat([prompt.to(dev).expand(len(S), -1), S], 1)).logits[:, prompt.shape[1]:]
    return torch.cat([lg[..., :EOS_ID], lg[..., EOS_ID + 1:]], -1) if eos == "ban" else lg


@torch.no_grad()
def calibrate(model, quant, fp_model, prompts, levels=(3, 4, 5, 6, 7, 8), b0=4, gen_len=256, steps=64, windows=8, batch=8,
              trajectory="fp", eos="last", log=print):
    """prompts: list of [1, P] id tensors. kappa sums over every position committed in the window. trajectory: "fp" (default) replays
    the full-precision trajectory, "target" the uniform b0 one. Returns dict(kappa {b: [K]}, iota [K], gain {b: [K]}, traces)."""
    spw = steps // windows; kappa = {b: torch.zeros(windows) for b in levels}; iota = torch.zeros(windows); traces = []
    for n, prompt in enumerate(prompts):
        sched = None if trajectory == "fp" else [b0] * windows
        _, tr = (generate(fp_model, None, prompt, gen_len, steps, None, windows, record=True, eos=eos) if trajectory == "fp"
                 else generate(model, quant, prompt, gen_len, steps, sched, windows, record=True, eos=eos))       # reference trajectory
        traces.append(tr); S = tr["states"]
        for i in range(0, steps, batch):                                                                # kappa: replay each state
            ts = range(i, min(i + batch, steps)); rows = [(r, t, tr["commits"][t]) for r, t in enumerate(ts)]
            lg = _gen_logits(fp_model, prompt, S[i:i + len(ts)], eos)
            fp = {t: F.log_softmax(lg[r, idx.to(lg.device)].float(), -1) for r, t, idx in rows}; del lg
            for b in levels:
                quant.bits = b; lg = _gen_logits(model, prompt, S[i:i + len(ts)], eos); quant.bits = None
                for r, t, idx in rows:
                    if len(idx):
                        kappa[b][t // spw] += _kl(fp[t], F.log_softmax(lg[r, idx.to(lg.device)].float(), -1)).sum().cpu() / len(prompts)
                del lg
        ends = torch.cat([S[::spw][1:], tr["final"][None]])                                             # x_after of every window
        for k in range(windows):                                                                        # iota: fp beliefs before / after
            still = (ends[k] == MASK_ID)
            if not still.any():
                continue
            lg = _gen_logits(fp_model, prompt, torch.stack([S[k * spw], ends[k]]), eos); lg = lg[:, still.to(lg.device)]
            iota[k] += _kl(F.log_softmax(lg[1].float(), -1), F.log_softmax(lg[0].float(), -1)).sum().cpu() / len(prompts)
            del lg
        log(f"calibration prompt {n + 1}/{len(prompts)} done")
    gain = {b: iota * (kappa[b0] - kappa[b]) for b in levels}
    return {"kappa": kappa, "iota": iota, "gain": gain, "traces": traces}
