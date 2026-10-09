"""Checks on the real model that the rotation leaves the full-precision function unchanged (up to bf16 rounding): logits on partially
masked GSM8K inputs before vs after `apply_rotation`, reported as top-1 agreement and the mean KL between the two predictive
distributions at masked positions.  Usage: python scripts/check_rotation.py [--n 8]"""
import argparse, sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2quant import apply_rotation, data  # noqa: E402
from t2quant.models import load_fp, load_tokenizer  # noqa: E402
from t2quant.sampler import MASK_ID  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
a = ap.parse_args(); tok = load_tokenizer(a.model); model = load_fp(a.model); g = torch.Generator().manual_seed(0)
X = []
for r in data.load("test", a.n):
    ids = data.prompt_ids(tok, r["question"])[0]; gen = torch.full((256,), MASK_ID)
    keep = torch.rand(256, generator=g) < 0.5; gen[keep] = torch.randint(100, 30000, (int(keep.sum()),), generator=g)   # half the positions filled
    X.append(torch.cat([ids, gen]))
with torch.no_grad():
    before = [model(x[None].cuda()).logits[0].float().cpu() for x in X]
    apply_rotation(model, seed=0)
    after = [model(x[None].cuda()).logits[0].float().cpu() for x in X]
agree = kl = n = 0
for x, b, c in zip(X, before, after):
    m = x == MASK_ID; lb, lc = F.log_softmax(b[m], -1), F.log_softmax(c[m], -1)
    agree += int((lb.argmax(-1) == lc.argmax(-1)).sum()); kl += float((lb.exp() * (lb - lc)).sum(-1).sum()); n += int(m.sum())
print(f"masked positions: {n}; top-1 agreement {100 * agree / n:.2f}%; mean KL(before || after) {kl / n:.2e} nats")
