# T²Quant

Label-free, task-specific allocation of activation precision over the denoising trajectory of a masked-diffusion language model
(LLaDA-8B-Instruct), at a fixed average bit budget.

`T2Quant.ipynb` runs the method end to end on GSM8K:

| step | what happens | code |
|---|---|---|
| 0 | load LLaDA-8B-Instruct, apply a QuaRot-style rotation, install switchable per-token activation quantization | `t2quant/rotation.py`, `t2quant/actquant.py` |
| 1 | **calibration** on a few *unlabeled* prompts: per-window local divergence $\kappa_k(b)$ and downstream influence $\iota_k$ | `t2quant/calibrate.py` |
| 2 | predicted gain $g_k(b) = \iota_k[\kappa_k(b_0) - \kappa_k(b)]$ and **exact dynamic-programming allocation** at the uniform budget | `t2quant/allocate.py` |
| 3 | evaluation: full precision vs uniform $A_\mathrm{target}$ vs the T²Quant schedule on held-out problems | `t2quant/sampler.py`, `t2quant/data.py` |

## Requirements
* One CUDA GPU with at least 24 GB of memory (the model runs in bf16). The notebook takes roughly 25–30 minutes.
* Python 3.10+, then `pip install -r requirements.txt`.
* The model (`GSAI-ML/LLaDA-8B-Instruct`) and GSM8K are downloaded from the Hugging Face Hub on first use.

## Run
```bash
pip install -r requirements.txt
jupyter notebook T2Quant.ipynb     # or, headless: python run.py  (figures saved as fig*.png)
python -m pytest tests -q          # CPU tests: rotation exactness, quantizer, DP optimality, sampler, calibration (seconds)
```
`T2Quant.ipynb` and `run.py` are generated from the same cells by `build_notebook.py`.

## Configuration
All settings are in the `CFG` dictionary of the second cell: generation length / steps / windows, the precision levels
(`levels`, default `(3, 4, 8)` = $A_\mathrm{down}$ / $A_\mathrm{target}$ / $A_\mathrm{up}$) and target `b0`, the number of
calibration prompts and of evaluation problems.

## Differences from the paper's setup
* Weights stay in bf16 (activation-only quantization); the paper additionally quantizes weights to 4 bits with GPTQ.
* 256 generated tokens / 64 steps / 8 windows, 8 calibration prompts and 50 test problems (paper: 512 tokens / 128 steps /
  16 windows and hundreds of problems); one task, one rotation seed. With 50 problems, accuracy differences of a few points are
  within noise.
* The sampler re-implements LLaDA's low-confidence remasking (greedy, one block, end-of-sequence committed last).
