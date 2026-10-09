"""Dynamic per-token activation fake-quantization with a switchable bit-width.

The four activation tensors of every transformer block are quantized: the inputs of the q/k/v projections, of the attention
output projection, of the FFN up/gate projections and of the FFN down projection (after the online Hadamard). Per token (row):
asymmetric min/max range, shrunk by a clip ratio below 8 bits, rounded to 2^b levels. `bits = None` disables quantization, so
the same model object serves as the full-precision reference. T²Quant only changes `bits` between denoising steps.
"""
from __future__ import annotations

import torch
from torch import nn

from .rotation import transformer


def fake_quant(x: torch.Tensor, bits: int, clip: float = 0.7) -> torch.Tensor:
    xf = x.float(); a = xf.amin(-1, keepdim=True); b = xf.amax(-1, keepdim=True)
    if clip < 1.0 and bits < 8:
        a, b = a * clip, b * clip
    qmax = float(2 ** bits - 1); s = ((b - a) / qmax).clamp_min(1e-12); z = torch.round(-a / s)
    return ((torch.clamp(torch.round(xf / s) + z, 0.0, qmax) - z) * s).to(x.dtype)


class ActQuant:
    """Installs the quantization hooks; set `.bits` (int or None) before each forward."""

    def __init__(self, model: nn.Module, clip: float = 0.7):
        self.bits: int | None = None; self.clip = clip; self.handles = []
        for block in transformer(model).blocks:
            self.handles.append(block.attn_norm.register_forward_hook(self._post))   # q/k/v input
            self.handles.append(block.ff_norm.register_forward_hook(self._post))     # up/gate input
            self.handles.append(block.attn_out.register_forward_pre_hook(self._pre)) # attention-output input
            self.handles.append(block.ff_out.register_forward_pre_hook(self._pre))   # down input (after the online Hadamard)

    def _post(self, _mod, _args, out):
        return out if self.bits is None else fake_quant(out, self.bits, self.clip)

    def _pre(self, _mod, args):
        return args if self.bits is None else (fake_quant(args[0], self.bits, self.clip),) + tuple(args[1:])
