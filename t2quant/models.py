"""Builds the full-precision reference and the quantized base (rotation + GPTQ weights + switchable activation quantization), with an
optional on-disk cache of the GPTQ weights so that several processes (e.g. evaluation shards) do not each re-run GPTQ."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from .actquant import ActQuant
from .gptq import apply_gptq, calibration_windows
from .rotation import apply_rotation, transformer

LINEARS = ("q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out")


def load_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def load_fp(name: str, device: str = "cuda"):
    return AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()


def load_quantized(name: str, tok, weight_bits: int = 4, clip: float = 0.7, seed: int = 0, cache_dir: str | None = None, device: str = "cuda",
                   log=print):
    """Rotated model with GPTQ `weight_bits` weights (None = keep bf16) and activation quantization hooks -> (model, ActQuant)."""
    model = load_fp(name, device); apply_rotation(model, seed=seed)
    if weight_bits:
        key = hashlib.sha1(f"{name}|w{weight_bits}|seed{seed}|gptq-wikitext103-128x512".encode()).hexdigest()[:12]
        path = Path(cache_dir) / f"gptq_{key}.pt" if cache_dir else None
        blocks = transformer(model).blocks
        if path is not None and path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            for i, b in enumerate(blocks):
                for n in LINEARS:
                    getattr(b, n).weight.data.copy_(saved[f"{i}.{n}"].to(device))
            log(f"GPTQ weights loaded from {path}")
        else:
            apply_gptq(model, calibration_windows(tok, seed=seed), bits=weight_bits, log=log)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(f".tmp{os.getpid()}")
                torch.save({f"{i}.{n}": getattr(b, n).weight.data.cpu() for i, b in enumerate(blocks) for n in LINEARS}, tmp)
                os.replace(tmp, path); log(f"GPTQ weights cached at {path}")
    return model, ActQuant(model, clip=clip)
