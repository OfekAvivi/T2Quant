"""LLaDA sampler (one block, greedy, low-confidence remasking) with a per-step precision schedule and trajectory recording.

At each of T steps the model predicts every masked position; the `G / T` most confident predictions are committed. Positions whose
prediction is end-of-sequence / end-of-turn get confidence 0, so they are committed last (LLaDA's evaluation rule for reasoning
tasks). `schedule` maps a window index (K windows of T / K steps) to an activation bit-width (None = no activation quantization);
`quant` may be None for a model without activation quantization (e.g. the full-precision reference).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MASK_ID, EOS_IDS = 126336, (126081, 126348)


@torch.no_grad()
def generate(model, quant, prompt_ids: torch.Tensor, gen_len: int = 256, steps: int = 64, schedule=None, windows: int = 8,
             record: bool = False):
    """prompt_ids [1, P] -> (generated ids [G], trace). trace = {'states': [T, G] pre-commit states, 'commits': list of T index tensors}."""
    dev = next(model.parameters()).device; P = prompt_ids.shape[1]
    x = torch.cat([prompt_ids.to(dev), torch.full((1, gen_len), MASK_ID, device=dev)], 1)
    per_step = gen_len // steps; states, commits = [], []
    for t in range(steps):
        if quant is not None:
            quant.bits = None if schedule is None else schedule[t * windows // steps]
        mask = x[0, P:] == MASK_ID
        if record:
            states.append(x[0, P:].clone())
        logits = model(x).logits[0, P:].float()
        x0 = logits.argmax(-1)
        conf = F.softmax(logits, -1).gather(-1, x0[:, None])[:, 0]
        conf = torch.where(torch.isin(x0, torch.tensor(EOS_IDS, device=dev)), torch.zeros_like(conf), conf)   # EOS / EoT last
        conf = torch.where(mask, conf, torch.full_like(conf, -float("inf")))
        k = per_step if t < steps - 1 else int(mask.sum())
        idx = conf.topk(k).indices
        x[0, P + idx] = x0[idx]
        if record:
            commits.append(idx.cpu())
    if quant is not None:
        quant.bits = None
    trace = {"states": torch.stack(states).cpu(), "commits": commits, "final": x[0, P:].cpu()} if record else None
    return x[0, P:].cpu(), trace
