"""Checks that t2quant.generate reproduces LLaDA's reference sampler (generate.py of the official LLaDA repository) token for token,
on the full-precision model and on the quantized base at uniform A4, for both end-of-sequence rules.
Usage: LLADA_REPO=/path/to/LLaDA python scripts/check_sampler_parity.py [--n 5] [--gen_len 256] [--steps 64]"""
import argparse, os, sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from t2quant import ActQuant, apply_gptq, apply_rotation, calibration_windows, generate  # noqa: E402
from t2quant import data  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5); ap.add_argument("--gen_len", type=int, default=256)
ap.add_argument("--steps", type=int, default=64); ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
ap.add_argument("--quantized", action="store_true", help="also check the rotated W4 (GPTQ) model at uniform A4 (takes ~10 more minutes)")
a = ap.parse_args()
sys.path.insert(0, os.environ.get("LLADA_REPO", "../LLaDA")); from generate import generate as reference  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModel.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
prompts = [data.prompt_ids(tok, r["question"]) for r in data.load("test", a.n)]
cases = [("full precision", None)]
if a.quantized:
    apply_rotation(model, seed=0); apply_gptq(model, calibration_windows(tok)); q = ActQuant(model); cases.append(("W4A4", q))
ok = True
for name, q in cases:
    for eos in ("last", "ban"):
        same = 0
        for ids in prompts:
            if q is not None:
                q.bits = 4
            ref = reference(model, ids.cuda(), steps=a.steps, gen_length=a.gen_len, block_length=a.gen_len, temperature=0.0, cfg_scale=0.0,
                            remasking="low_confidence", logits_eos_inf=eos == "ban", confidence_eos_eot_inf=eos == "last")[0, ids.shape[1]:].cpu()
            ours, _ = generate(model, q, ids, a.gen_len, a.steps, None if q is None else [4] * 8, 8, eos=eos)
            same += int(torch.equal(ref, ours))
        print(f"{name:15s} eos={eos:4s}: {same}/{len(prompts)} token-identical", flush=True); ok &= same == len(prompts)
print("PARITY OK" if ok else "PARITY MISMATCH")
sys.exit(0 if ok else 1)
