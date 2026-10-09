"""Writes T2Quant.ipynb (and run.py, the same code as a script) from the cells below, so the two never drift apart.
Usage: python build_notebook.py"""
import json

CELLS = [
("md", """# T²Quant

Masked-diffusion LMs (here **LLaDA-8B-Instruct**) generate by iteratively unmasking tokens over many denoising steps. Under 4-bit
activation quantization, *when* during denoising precision matters depends on the task. T²Quant finds this out **without
labels** and reallocates activation precision over the denoising trajectory at a **fixed average bit budget**:

1. **Calibration** (a few unlabeled prompts): for each denoising window, the local divergence $\\kappa_k(b)$ between full-precision
   and $b$-bit predictions at the positions the window commits, and the window's downstream influence $\\iota_k$ on positions not yet generated.
2. **Predicted gain** of moving window $k$ from the target precision $b_0$ to $b$: $g_k(b) = \\iota_k\\,[\\kappa_k(b_0) - \\kappa_k(b)]$.
3. **Allocation**: an exact dynamic program picks one precision per window, maximizing $\\sum_k g_k(b_k)$ at the budget of uniform $b_0$.
4. **Deployment**: each denoising step looks up its window's precision.

This notebook runs the full pipeline on GSM8K on one GPU (≥ 24 GB, roughly 25–30 minutes). Its settings differ from the paper's in a
few places, listed in the last cell."""),
("code", """import torch, numpy as np, matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from t2quant import ActQuant, allocate, apply_rotation, calibrate, generate
from t2quant import data

CFG = dict(
    model="GSAI-ML/LLaDA-8B-Instruct",
    gen_len=256, steps=64, windows=8,     # 4 tokens per step; 8 windows of 8 steps
    levels=(3, 4, 8), b0=4,               # A_down / A_target / A_up activation bits; budget = uniform A_target
    clip=0.7,                             # activation clip ratio below 8 bits
    n_calib=8,                            # unlabeled calibration prompts (GSM8K train questions; answers never read)
    n_eval=50,                            # GSM8K test problems for the evaluation
)
torch.manual_seed(0)"""),
("md", """## 0. Model: full precision and quantized in one object
A QuaRot-style rotation (exact in full precision) makes per-token 4-bit activations usable. The activation quantizer is a set of hooks
whose bit-width can be changed between forward passes (`quant.bits = None` gives the full-precision model). Weights stay in bf16."""),
("code", """tok = AutoTokenizer.from_pretrained(CFG["model"], trust_remote_code=True)
model = AutoModel.from_pretrained(CFG["model"], trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
apply_rotation(model, seed=0)
quant = ActQuant(model, clip=CFG["clip"])

q = data.load("test", 1)[0]; ids = data.prompt_ids(tok, q["question"])
for name, sched in (("full precision", None), ("uniform A4", [CFG["b0"]] * CFG["windows"])):
    out, _ = generate(model, quant, ids, CFG["gen_len"], CFG["steps"], sched, CFG["windows"])
    print(f"--- {name}:", tok.decode(out, skip_special_tokens=True)[:400])"""),
("md", """## 1. Label-free calibration
Run the uniform-$A_\\mathrm{target}$ model on a few unlabeled prompts, record every intermediate state, and replay each state at every
precision $b$ (the state is held fixed, so only the precision differs). $\\kappa_k(b)$ sums KL(full precision ‖ $b$-bit) over the
positions committed in window $k$; $\\iota_k$ measures how much window $k$'s commitments change the full-precision model's beliefs
about the positions still masked afterwards. No answers, labels or task metrics are used."""),
("code", """calib = [data.prompt_ids(tok, r["question"]) for r in data.load("train", CFG["n_calib"])]
cal = calibrate(model, quant, calib, CFG["levels"], CFG["b0"], CFG["gen_len"], CFG["steps"], CFG["windows"])

K = CFG["windows"]; x = np.arange(1, K + 1)
fig, ax = plt.subplots(1, 3, figsize=(12, 3))
for b in CFG["levels"]:
    ax[0].plot(x, cal["kappa"][b], marker="o", label=f"A{b}")
ax[0].set(title="local divergence $\\\\kappa_k(b)$", xlabel="window"); ax[0].legend()
ax[1].bar(x, cal["iota"], color="grey"); ax[1].set(title="influence $\\\\iota_k$", xlabel="window")
for b in CFG["levels"]:
    if b != CFG["b0"]:
        ax[2].bar(x + (0.2 if b > CFG["b0"] else -0.2), cal["gain"][b], width=0.4, label=f"A{CFG['b0']} -> A{b}")
ax[2].axhline(0, color="k", lw=0.6); ax[2].set(title="predicted gain $g_k(b)$", xlabel="window"); ax[2].legend()
plt.tight_layout(); plt.show()"""),
("md", """## 2. Exact budget-constrained allocation
One precision per window, maximizing the total predicted gain at exactly the average bits of uniform $A_\\mathrm{target}$
(each upgraded window must be paid for by downgrading others)."""),
("code", """schedule = allocate(cal["gain"], CFG["levels"], CFG["b0"])
print("T²Quant schedule (bits per window):", schedule, "| average bits:", np.mean(schedule))"""),
("md", """## 3. Evaluation on held-out problems
Full precision, uniform $A_\\mathrm{target}$ and the T²Quant schedule (same average bits) on the first `n_eval` GSM8K test
problems. With 50 problems the accuracy differences are noisy (one problem = 2 points); the paired win / loss counts are more informative."""),
("code", """evalset = data.load("test", CFG["n_eval"])
def evaluate(sched):
    ok = []
    for r in evalset:
        out, _ = generate(model, quant, data.prompt_ids(tok, r["question"]), CFG["gen_len"], CFG["steps"], sched, CFG["windows"])
        ok.append(data.correct(tok.decode(out, skip_special_tokens=True), r["gold"]))
    return np.array(ok)

res = {"full precision": evaluate(None), f"uniform A{CFG['b0']}": evaluate([CFG["b0"]] * K), "T²Quant": evaluate(schedule)}
for name, ok in res.items():
    print(f"{name:16s} {100 * ok.mean():5.1f}%")
u, s = res[f"uniform A{CFG['b0']}"], res["T²Quant"]
print(f"T²Quant vs uniform: fixed {int((s & ~u).sum())} problems, broke {int((u & ~s).sum())}")
plt.figure(figsize=(4, 3)); plt.bar(list(res), [100 * v.mean() for v in res.values()], color=["k", "grey", "tab:blue"])
plt.ylabel("GSM8K accuracy (%)"); plt.xticks(rotation=15); plt.tight_layout(); plt.show()"""),
("md", """## Differences from the paper's setup
* **Weights** stay in bf16 (activation-only quantization); the paper's base also quantizes weights to 4 bits with GPTQ.
* **Scale**: 256 tokens / 64 steps / 8 windows, 8 calibration prompts and 50 test problems, versus 512 tokens / 128 steps / 16 windows
  and hundreds of problems in the paper; one task, one rotation seed.
* **Task**: GSM8K only; other tasks need their prompt format and scorer in `t2quant/data.py`. Everything else (`calibrate`, `allocate`,
  `generate`) is task-agnostic.
* Change `levels` / `b0` in `CFG` to try other precision sets (e.g. `(4, 8)` with a larger budget passed to `allocate`)."""),
]


def main():
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    script = ['"""Headless version of T2Quant.ipynb (generated by build_notebook.py): python run.py"""',
              "import matplotlib; matplotlib.use('Agg')"]
    for i, (kind, src) in enumerate(CELLS):
        lines = src.splitlines(keepends=True)
        if kind == "md":
            nb["cells"].append({"cell_type": "markdown", "id": f"c{i}", "metadata": {}, "source": lines})
            script.append("\n# " + "\n# ".join(src.splitlines()))
        else:
            nb["cells"].append({"cell_type": "code", "id": f"c{i}", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
            script.append(src.replace("plt.show()", "plt.savefig(f'fig{plt.gcf().number}.png')"))
    json.dump(nb, open("T2Quant.ipynb", "w"), indent=1)
    open("run.py", "w").write("\n".join(script) + "\n")
    print("wrote T2Quant.ipynb and run.py")


if __name__ == "__main__":
    main()
