"""GPTQ 4-bit weight quantization for LLaDA (Frantar et al., 2023), as used for the quantized base model.

Per linear layer, second-order error feedback with the Hessian H = sum x x^T of that layer's inputs; transformer blocks are processed
in order, so every Hessian sees the already-quantized earlier blocks. Per-output-channel asymmetric grid with a min-MSE clip search
and act-order. Calibration inputs are WikiText-103 windows masked at a random ratio per sequence, so the Hessians match the inputs
the denoising sampler feeds the model. Run after `apply_rotation` (the Hessian hooks see the rotated, Hadamard-transformed inputs).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .rotation import transformer
from .sampler import MASK_ID

GROUPS = (("q_proj", "k_proj", "v_proj"), ("attn_out",), ("ff_proj", "up_proj"), ("ff_out",))   # quantized in this order within a block
SHRINK = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)


def calibration_windows(tok, n: int = 128, seq: int = 512, n_lines: int = 4000, seed: int = 0) -> torch.Tensor:
    """[n, seq] windows of the first `n_lines` non-empty WikiText-103 training lines, each masked at its own ratio t ~ U(0, 1)."""
    from datasets import load_dataset
    lines = [t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"] if t.strip()][:n_lines]
    ids = torch.tensor([i for t in lines for i in tok(t, add_special_tokens=False)["input_ids"]], dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, ids.numel() - seq - 1, (n,), generator=g)
    X = torch.stack([ids[s:s + seq] for s in starts.tolist()]).clone()
    X[torch.rand(n, seq, generator=g) < torch.rand(n, 1, generator=g)] = MASK_ID
    return X


def _grid(w, bits):
    """Per-row (scale, zero) of the asymmetric grid with the smallest squared error over the shrink factors."""
    qmax = 2 ** bits - 1; lo0, hi0 = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True); best = None
    for f in SHRINK:
        s = ((hi0 - lo0) * f / qmax).clamp_min(1e-8); z = torch.round(-lo0 * f / s)
        err = ((torch.clamp(torch.round(w / s) + z, 0, qmax) - z) * s - w).pow(2).sum(-1, keepdim=True)
        if best is None:
            best = [s, z, err]
        else:
            m = err < best[2]; best = [torch.where(m, s, best[0]), torch.where(m, z, best[1]), torch.where(m, err, best[2])]
    return best[0], best[1]


@torch.no_grad()
def gptq_layer(W: torch.Tensor, H: torch.Tensor, bits: int = 4, blocksize: int = 128, damp: float = 0.01) -> torch.Tensor:
    """W [out, in], H [in, in] -> quantized W (same dtype)."""
    dtype = W.dtype; W = W.float().clone(); H = H.clone(); n = W.shape[1]; qmax = 2 ** bits - 1
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0.0
    perm = torch.argsort(torch.diag(H), descending=True); W = W[:, perm]; H = H[perm][:, perm]   # act-order
    H += torch.eye(n, device=H.device) * damp * torch.mean(torch.diag(H))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    s, z = _grid(W, bits); Q = torch.zeros_like(W)
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n); W1 = W[:, i1:i2].clone(); Err = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]; q = ((torch.clamp(torch.round(w[:, None] / s) + z, 0, qmax) - z) * s)[:, 0]
            Q[:, i1 + i] = q; e = (w - q) / Hi[i, i]
            W1[:, i:] -= e[:, None] @ Hi[i, i:][None, :]; Err[:, i] = e
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, torch.argsort(perm)].to(dtype)


class _Hessian:
    def __init__(self, lin: nn.Linear):
        self.n = 0; self.H = torch.zeros(lin.in_features, lin.in_features, device=lin.weight.device)
        self.h = lin.register_forward_pre_hook(self._pre)

    def _pre(self, _mod, args):
        x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
        self.H *= self.n / (self.n + x.shape[0]); self.n += x.shape[0]; x = math.sqrt(2 / self.n) * x; self.H += x.T @ x


class _Stop(Exception):
    pass


@torch.no_grad()
def apply_gptq(model: nn.Module, calib: torch.Tensor, bits: int = 4, batch: int = 8, log=print) -> None:
    """Quantize every linear of every transformer block in place (block by block)."""
    tf = transformer(model); dev = tf.wte.weight.device; calib = calib.to(dev); xs = []

    def grab(_mod, args, _kw):
        xs.append(args[0]); raise _Stop
    h = tf.blocks[0].register_forward_pre_hook(grab, with_kwargs=True)
    for i in range(0, len(calib), batch):
        try:
            model(calib[i:i + batch])
        except _Stop:
            pass
    h.remove(); X = torch.cat(xs)
    for bi, block in enumerate(tf.blocks):
        for group in GROUPS:
            hs = {n: _Hessian(getattr(block, n)) for n in group}
            for i in range(0, len(X), batch):
                block(X[i:i + batch])
            for n, hh in hs.items():
                hh.h.remove(); lin = getattr(block, n); lin.weight.data.copy_(gptq_layer(lin.weight.data, hh.H, bits))
        X = torch.cat([block(X[i:i + batch])[0] for i in range(0, len(X), batch)])
        if bi % 8 == 7 or bi == len(tf.blocks) - 1:
            log(f"GPTQ: block {bi + 1}/{len(tf.blocks)} done")
