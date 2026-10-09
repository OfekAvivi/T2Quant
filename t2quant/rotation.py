"""QuaRot-style rotation for LLaDA (Llama-type blocks), so that per-token 4-bit activations are usable.

Offline and exact up to rounding: fuse the RMSNorm weights into the linears they feed, rotate the residual stream by a random-sign
Hadamard Q, and rotate the value -> attention-output path per head. Online: a Hadamard transform on the FFN intermediate before the
down projection. The network function is unchanged in full precision; only the activations seen by the quantizer are rotated.
Row-vector convention: y = x W^T with W of shape [out, in].
"""
from __future__ import annotations

import math

import torch
from torch import nn

_H12 = torch.tensor([   # 12 x 12 Hadamard (Paley construction)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1], [1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1],
    [1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1], [1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1], [1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1],
    [1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1], [1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1], [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1],
    [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], [1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1], [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1],
], dtype=torch.float64)


def transformer(model: nn.Module) -> nn.Module:
    return model.model.transformer if hasattr(model, "model") and hasattr(model.model, "transformer") else model.transformer


def _sylvester(k: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(k):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def _factor(n: int) -> tuple[int, int]:
    """n -> (m, p), p a power of two and m in {1, 12}, so that H_n = H_m (x) H_p."""
    k, m = 0, n
    while m % 2 == 0 and m > 1:
        m //= 2; k += 1
    if m == 1:
        return 1, n
    if m == 3 and k >= 2:
        return 12, n // 12
    raise ValueError(f"no Hadamard construction for n={n}")


def hadamard(n: int) -> torch.Tensor:
    """Orthonormal Hadamard-type matrix of size n = 2^k or 12 * 2^k (float64)."""
    m, p = _factor(n)
    h = _sylvester(p.bit_length() - 1) if m == 1 else torch.kron(_H12, _sylvester(p.bit_length() - 1))
    return h / math.sqrt(n)


def random_hadamard(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return hadamard(n) * (torch.randint(0, 2, (n,), generator=g).double() * 2 - 1)[None, :]


@torch.no_grad()
def _right(w, m):   # W <- W M
    w.data.copy_((w.data.double().to(m.device) @ m).to(w.dtype))


@torch.no_grad()
def _left(w, m):    # W <- M W
    w.data.copy_((m @ w.data.double().to(m.device)).to(w.dtype))


class _OnlineFFNHadamard:
    """u' = u H before the down projection (W_down' = W_down H folded offline); Kronecker form avoids the n x n matrix."""

    def __init__(self, model: nn.Module):
        tf = transformer(model); n = tf.blocks[0].ff_out.in_features; dev = tf.wte.weight.device
        m, p = _factor(n)
        self.Hm = (_H12 / math.sqrt(12)).to(dev) if m == 12 else torch.ones(1, 1, dtype=torch.float64, device=dev)
        self.Hp = _sylvester(p.bit_length() - 1).to(dev) / math.sqrt(p)
        H = torch.kron(self.Hm, self.Hp)
        for block in tf.blocks:
            _right(block.ff_out.weight, H)
        self._c = None
        self.handles = [block.ff_out.register_forward_pre_hook(self._pre) for block in tf.blocks]

    def _pre(self, _mod, args):
        u = args[0]
        if self._c is None or self._c[0].dtype != u.dtype:
            self._c = (self.Hm.to(u), self.Hp.to(u))
        Hm, Hp = self._c; shp = u.shape
        x = u.reshape(*shp[:-1], Hm.shape[0], Hp.shape[0])
        x = torch.einsum("...np,nm->...mp", torch.einsum("...mp,pq->...mq", x, Hp), Hm)
        return (x.reshape(shp),) + tuple(args[1:])


@torch.no_grad()
def apply_rotation(model: nn.Module, seed: int = 0) -> None:
    """Fuse norms, rotate the residual stream and the per-head value path, install the online FFN Hadamard (in place)."""
    tf = transformer(model)
    for block in tf.blocks:                                            # 1. fuse RMSNorm weights into their consumers
        for norm, consumers in ((block.attn_norm, (block.q_proj, block.k_proj, block.v_proj)), (block.ff_norm, (block.ff_proj, block.up_proj))):
            g = norm.weight.data.double()
            for lin in consumers:
                lin.weight.data.copy_((lin.weight.data.double() * g[None, :]).to(lin.weight.dtype))
            norm.weight.data.fill_(1.0)
    g = tf.ln_f.weight.data.double()
    tf.ff_out.weight.data.copy_((tf.ff_out.weight.data.double() * g[None, :]).to(tf.ff_out.weight.dtype)); tf.ln_f.weight.data.fill_(1.0)
    Q = random_hadamard(tf.wte.weight.shape[1], seed).to(tf.wte.weight.device)   # 2. rotate the residual stream
    _right(tf.wte.weight, Q)
    for block in tf.blocks:
        for lin in (block.q_proj, block.k_proj, block.v_proj, block.ff_proj, block.up_proj):
            _right(lin.weight, Q)
        for lin in (block.attn_out, block.ff_out):
            _left(lin.weight, Q.T)
    _right(tf.ff_out.weight, Q)
    H = int(model.config.n_heads); hd = int(model.config.d_model) // H                # 3. per-head Hadamard on v -> o
    IH = torch.block_diag(*([hadamard(hd).to(Q.device)] * H))
    for block in tf.blocks:
        _left(block.v_proj.weight, IH); _right(block.attn_out.weight, IH)
    model._tq_online = _OnlineFFNHadamard(model)                                     # 4. online FFN Hadamard
