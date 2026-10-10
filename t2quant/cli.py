"""Command-line pipeline for larger runs (several tasks, more problems, several GPUs).

  python -m t2quant.cli calibrate --task gsm8k --out runs/gsm8k [--shard i/n]  # calibration + DP -> calibration.pt, schedule.json, calibration.png
  python -m t2quant.cli evaluate  --task gsm8k --out runs/gsm8k --config {fp,uniform,t2quant} [--shard i/n]
  python -m t2quant.cli report    --out runs/gsm8k                       # accuracy with 95% CI, paired comparison against uniform

Calibration shards (prompts i, i+n, ...) and evaluation shards (problems i, i+n, ...) can run in parallel on several GPUs; the last
calibration shard to finish merges the shards and runs the DP, `report` merges the evaluation shards. `--cache_dir` stores the GPTQ
weights after the first run so that later processes load them instead of re-running GPTQ. Calibration prompts never overlap the
evaluation problems (GSM8K: training questions; HumanEval: a fixed shuffle of the problems, the first n_calib calibrate and the rest
evaluate).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import data
from .allocate import allocate
from .calibrate import calibrate
from .models import load_fp, load_quantized, load_tokenizer
from .sampler import generate


def _common(ap):
    ap.add_argument("--task", default="gsm8k", choices=("gsm8k", "humaneval")); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct"); ap.add_argument("--gen_len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=64); ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--weight_bits", type=int, default=4); ap.add_argument("--clip", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0, help="rotation seed of the quantized base")
    ap.add_argument("--b0", type=int, default=4)
    ap.add_argument("--n_calib", type=int, default=None, help="calibration prompts (default: 16 for GSM8K, 50 for HumanEval)")
    ap.add_argument("--cache_dir", default=None)


def cmd_calibrate(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tok = load_tokenizer(a.model)
    T = data.task(a.task, tok, a.n_calib, 1); levels = tuple(int(b) for b in a.levels.split(",")); i, n = (int(v) for v in a.shard.split("/"))
    fp = load_fp(a.model); model, quant = load_quantized(a.model, tok, a.weight_bits, a.clip, a.seed, a.cache_dir)
    prompts = T["calib"][i::n]
    cal = calibrate(model, quant, fp, prompts, levels, a.b0, a.gen_len, a.steps, a.windows,
                    trajectory=a.trajectory, eos=T["eos"])
    part = out / f"calibration.shard{i}of{n}.pt"; tmp = part.with_suffix(".tmp")      # sums over this shard's prompts
    torch.save({"kappa": {b: v * len(prompts) for b, v in cal["kappa"].items()}, "iota": cal["iota"] * len(prompts), "n": len(prompts)}, tmp)
    tmp.replace(part)
    parts = [out / f"calibration.shard{j}of{n}.pt" for j in range(n)]
    if all(p.exists() for p in parts):
        _finish(a, out, levels, [torch.load(p, weights_only=False) for p in parts])
    else:
        print(f"calibration shard {i}/{n} done; the DP runs when all {n} shards are in {out}")


def _finish(a, out, levels, parts):
    """Merges the calibration shards (prompt means), runs the DP, writes calibration.pt, schedule.json and calibration.png."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    N = sum(p["n"] for p in parts); kappa = {b: sum(p["kappa"][b] for p in parts) / N for b in levels}; iota = sum(p["iota"] for p in parts) / N
    gain = {b: iota * (kappa[a.b0] - kappa[b]) for b in levels}; sched = allocate(gain, levels, a.b0)
    torch.save({"kappa": kappa, "iota": iota, "gain": gain, "n_prompts": N}, out / "calibration.pt")
    meta = {k: v for k, v in vars(a).items() if k not in ("func", "shard")}; meta["n_calib"] = N
    (out / "schedule.json").write_text(json.dumps({"schedule": [int(b) for b in sched], "average_bits": float(np.mean(sched)), **meta}, indent=1))
    K = a.windows; x = np.arange(1, K + 1); fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    for b in levels:
        ax[0].plot(x, kappa[b], marker="o", label=f"A{b}")
        if b != a.b0:
            ax[2].plot(x, gain[b], marker="o", label=f"A{a.b0}->A{b}")
    ax[0].set(title="local divergence kappa_k(b)", xlabel="window"); ax[0].legend(fontsize=7)
    ax[1].bar(x, iota, color="grey"); ax[1].set(title="influence iota_k", xlabel="window")
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set(title="predicted gain g_k(b)", xlabel="window"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out / "calibration.png", dpi=150)
    print(f"{a.task}: schedule {sched} (average {np.mean(sched):.2f} bits, {N} calibration prompts) -> {out / 'schedule.json'}")


def cmd_evaluate(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tok = load_tokenizer(a.model)
    if a.n_calib is None and (out / "schedule.json").exists():      # evaluate on the problems the calibration did not use
        a.n_calib = json.loads((out / "schedule.json").read_text())["n_calib"]
    T = data.task(a.task, tok, a.n_calib, a.n_eval); i, n = (int(v) for v in a.shard.split("/")); K = a.windows
    if a.config == "fp":
        model, quant, sched = load_fp(a.model), None, None
    else:
        model, quant = load_quantized(a.model, tok, a.weight_bits, a.clip, a.seed, a.cache_dir)
        sched = [a.b0] * K if a.config == "uniform" else json.loads((out / "schedule.json").read_text())["schedule"]
    path = out / f"preds_{a.config}.shard{i}of{n}.jsonl"
    done = {json.loads(l)["idx"] for l in open(path)} if path.exists() else set()
    with open(path, "a") as f:
        for idx in range(i, len(T["eval"]), n):
            if idx in done:
                continue
            ids, score = T["eval"][idx]
            gen, _ = generate(model, quant, ids, a.gen_len, a.steps, sched, K, eos=T["eos"]); text = tok.decode(gen, skip_special_tokens=True)
            f.write(json.dumps({"idx": idx, "ok": bool(score(text)), "response": text}) + "\n"); f.flush()
    print(f"{a.task} {a.config} shard {i}/{n} done -> {path}")


def _load(out, config):
    rows = {}
    for p in sorted(Path(out).glob(f"preds_{config}.shard*of*.jsonl")):
        for l in open(p):
            r = json.loads(l); rows[r["idx"]] = r["ok"]
    return rows


def cmd_report(a):
    rng = np.random.default_rng(0); R = {c: _load(a.out, c) for c in ("fp", "uniform", "t2quant")}; R = {c: v for c, v in R.items() if v}
    common = sorted(set.intersection(*(set(v) for v in R.values()))) if R else []
    lines = [f"{len(common)} problems evaluated by every configuration"]
    for c, v in R.items():
        x = np.array([v[i] for i in common], float); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        lines.append(f"{c:9s} {100 * x.mean():5.1f}%  (95% CI {100 * np.percentile(bs, 2.5):.1f}-{100 * np.percentile(bs, 97.5):.1f})")
    if "uniform" in R and "t2quant" in R:
        up = sum(R["t2quant"][i] and not R["uniform"][i] for i in common); dn = sum(R["uniform"][i] and not R["t2quant"][i] for i in common)
        k = up + dn; p = min(1.0, 2 * sum(math.comb(k, j) for j in range(max(up, dn), k + 1)) / 2 ** k) if k else 1.0
        lines.append(f"T2Quant vs uniform: fixed {up}, broke {dn} (exact sign test p = {p:.3f})")
    print("\n".join(lines)); (Path(a.out) / "report.txt").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(required=True)
    c = sub.add_parser("calibrate"); _common(c); c.add_argument("--levels", default="3,4,5,6,7,8")
    c.add_argument("--trajectory", default="fp", choices=("fp", "target"))
    c.add_argument("--shard", default="0/1", help="i/n: calibrate on prompts i, i+n, ... (merged when all n shards exist)")
    c.set_defaults(func=cmd_calibrate)
    e = sub.add_parser("evaluate"); _common(e); e.add_argument("--config", required=True, choices=("fp", "uniform", "t2quant"))
    e.add_argument("--n_eval", type=int, default=None); e.add_argument("--shard", default="0/1"); e.set_defaults(func=cmd_evaluate)
    r = sub.add_parser("report"); r.add_argument("--out", required=True); r.set_defaults(func=cmd_report)
    a = ap.parse_args(argv); a.func(a)


if __name__ == "__main__":
    main()
