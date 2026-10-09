"""LLaDA sampler (one block, greedy, low-confidence remasking) with a per-step precision schedule and trajectory recording.

A step-by-step re-implementation of LLaDA's reference `generate` (temperature 0, no classifier-free guidance, one block), so that the
precision of every denoising step can be set and every intermediate state recorded; it reproduces the reference token for token
(checked by scripts/check_sampler_parity.py). At each of T steps the model predicts every masked position and the most confident
predictions are committed (G / T per step, the remainder spread over the first steps). End-of-sequence handling, as in LLaDA's
evaluation settings:
  eos="last"  the confidence of end-of-sequence / end-of-turn predictions is set to 0 (reasoning tasks: EOS is committed last);
  eos="ban"   end-of-sequence can never be predicted (code tasks);
  eos="none"  no special handling.
`schedule` maps a window index (K windows of T / K steps) to an activation bit-width (None = no activation quantization); `quant`
may be None for a model without activation quantization (e.g. the full-precision reference).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MASK_ID, EOS_ID, EOT_ID = 126336, 126081, 126348
EOS_IDS = (EOS_ID, EOT_ID)


def transfer_counts(n_masked: int, steps: int) -> list[int]:
    """Tokens committed at each step: n // steps each, the remainder added to the first steps (LLaDA's get_num_transfer_tokens)."""
    base, rem = divmod(n_masked, steps)
    return [base + (1 if i < rem else 0) for i in range(steps)]


@torch.no_grad()
def generate(model, quant, prompt_ids: torch.Tensor, gen_len: int = 256, steps: int = 64, schedule=None, windows: int = 8,
             record: bool = False, eos: str = "last"):
    """prompt_ids [1, P] -> (generated ids [G], trace). trace = {'states': [T, G] pre-commit states, 'commits': list of T index
    tensors (generation-relative positions committed at each step), 'final': [G]}."""
    dev = next(model.parameters()).device; P = prompt_ids.shape[1]
    x = torch.cat([prompt_ids.to(dev), torch.full((1, gen_len), MASK_ID, device=dev, dtype=prompt_ids.dtype)], 1)
    counts = transfer_counts(gen_len, steps); states, commits = [], []
    for t in range(steps):
        if quant is not None:
            quant.bits = None if schedule is None else schedule[t * windows // steps]
        mask = x == MASK_ID
        if record:
            states.append(x[0, P:].clone())
        logits = model(x).logits
        if eos == "ban":
            logits[:, :, EOS_ID] = -torch.inf
        x0 = torch.argmax(logits, dim=-1)
        if eos == "last":                                   # after the argmax, as in the reference: EOS / EoT get probability 0
            logits[:, :, EOS_ID] = -torch.inf; logits[:, :, EOT_ID] = -torch.inf
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        x0 = torch.where(mask, x0, x)
        conf = torch.where(mask, x0_p, torch.tensor(-float("inf"), device=dev, dtype=x0_p.dtype))
        idx = torch.topk(conf[0], k=counts[t]).indices
        x[0, idx] = x0[0, idx]
        if record:
            commits.append((idx - P).cpu())
    if quant is not None:
        quant.bits = None
    trace = {"states": torch.stack(states).cpu(), "commits": commits, "final": x[0, P:].cpu()} if record else None
    return x[0, P:].cpu(), trace
