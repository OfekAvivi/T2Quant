# T²Quant

Label-free, task-specific allocation of activation precision over the denoising trajectory of a masked-diffusion language model
(LLaDA-8B-Instruct), at a fixed average bit budget.

`T2Quant.ipynb` runs the method end to end on GSM8K:

| step | what happens | code |
|---|---|---|
| 0 | load LLaDA-8B-Instruct twice: a full-precision reference, and the quantized base (QuaRot-style rotation, GPTQ 4-bit weights, switchable per-token activation quantization) | `t2quant/rotation.py`, `t2quant/gptq.py`, `t2quant/actquant.py` |
| 1 | **calibration** on a few *unlabeled* prompts: on the full-precision trajectory, per-window local divergence $\kappa_k(b)$ at score-relevant committed positions and downstream influence $\iota_k$ | `t2quant/calibrate.py`, `t2quant/data.py` |
| 2 | predicted gain $g_k(b) = \iota_k[\kappa_k(b_0) - \kappa_k(b)]$ and **exact dynamic-programming allocation** at the uniform budget | `t2quant/allocate.py` |
| 3 | evaluation: full precision vs uniform $A_\mathrm{target}$ vs the T²Quant schedule on held-out problems | `t2quant/sampler.py`, `t2quant/data.py` |

## Requirements
* One CUDA GPU with at least 40 GB of memory (the full-precision reference and the quantized model are both loaded in bf16). The
  notebook takes roughly 1.5 hours (GPTQ about 10–15 minutes, calibration about 15 minutes, evaluation of three configurations on
  200 problems about an hour).
* Python 3.10+, then `pip install -r requirements.txt`.
* The model (`GSAI-ML/LLaDA-8B-Instruct`), GSM8K, HumanEval and WikiText-103 (GPTQ calibration) are downloaded from the Hugging Face
  Hub on first use.

## Run
```bash
pip install -r requirements.txt
jupyter notebook T2Quant.ipynb     # or, headless: python run.py  (figures saved as fig*.png)
python -m pytest tests -q          # CPU tests: rotation exactness, GPTQ, quantizer, DP optimality, sampler, calibration (seconds)
```
`T2Quant.ipynb` and `run.py` are generated from the same cells by `build_notebook.py`.

The command line runs the same method on GSM8K or HumanEval at the paper's setting (512 tokens / 128 steps / 16 windows), with
calibration and evaluation split into shards that run in parallel, one per GPU:
```bash
P="--task humaneval --out runs/humaneval --gen_len 512 --steps 128 --windows 16 --cache_dir cache"   # or --task gsm8k
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python -m t2quant.cli calibrate $P --shard $i/4 & done; wait
for c in fp uniform t2quant; do
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python -m t2quant.cli evaluate $P --config $c --shard $i/4 & done; wait
done
python -m t2quant.cli report --out runs/humaneval       # accuracy with 95% bootstrap CI, fixed/broken problems, exact sign test
```
The last calibration shard to finish merges the shards and runs the allocation (`schedule.json`, `calibration.png`). Each process
that finds no cached GPTQ weights in `--cache_dir` computes them (10–15 minutes) and caches them for later runs. Calibration
prompts never overlap the evaluation problems: GSM8K calibrates on training questions (answers are never read); HumanEval, which has no
training split, uses a fixed shuffle of its 164 problems, the first 50 for calibration and the rest for evaluation (the first 100 of
them are the paper's test split).

## Configuration
All settings are in the `CFG` dictionary of the notebook's second cell (and the matching command-line flags): generation length /
steps / windows, weight bits, the activation precision levels (`levels`, default `(3, 4, 5, 6, 7, 8)`, the per-window choices) and
target `b0` (default 4), the number of calibration prompts (GSM8K 16, HumanEval 50) and of evaluation problems, and two calibration
choices:
* `trajectory`: the states replayed for $\kappa$ and $\iota$ — the full-precision trajectory (`"fp"`, default, as in the paper) or
  the quantized model's own uniform $A_\mathrm{target}$ trajectory (`"target"`);
* `kappa_positions`: the committed positions $\kappa$ sums over — the score-relevant ones (`"relevant"`, default, Eq. 3 of the paper;
  GSM8K: the answer text before any follow-up "Question:", HumanEval: the code inside the python fence) or every committed position
  (`"all"`).

## Results
LLaDA-8B-Instruct, 4-bit GPTQ weights, 4.00 average activation bits (levels 3–8, target 4), 512 tokens / 128 steps / 16 windows,
default calibration (full-precision trajectory, score-relevant positions). Fixed / broken: problems T²Quant solves and uniform W4A4
does not, and the reverse; p: exact two-sided sign test on those problems.

| task (evaluation problems) | calibration prompts | T²Quant schedule (bits per window) | full precision | uniform W4A4 | T²Quant | fixed / broken | p |
|---|---|---|---|---|---|---|---|
| GSM8K (first 500 test) | 16 train questions | 8 5 5 5 5 4 4 4 3 3 3 3 3 3 3 3 | 51.2 | 46.6 | **51.4** | 84 / 60 | 0.055 |
| HumanEval (100, the paper's split) | 50 problems | 4 4 4 4 4 4 5 5 4 5 5 4 3 3 3 3 | 25.0 | 23.0 | **25.0** | 7 / 5 | 0.77 |

The two tasks get different schedules from the same procedure: GSM8K moves precision to the first windows, HumanEval to the middle of
the trajectory, where the code is written.

Calibration choices (same setting; T²Quant accuracy, uniform W4A4 as above):

| trajectory | positions | GSM8K | HumanEval |
|---|---|---|---|
| full precision | score-relevant (default) | 51.4 | **25.0** |
| full precision | all committed | 51.4 (same schedule) | 18.0 (schedule 5 4 4 … 4 3) |
| uniform $A_\mathrm{target}$ | score-relevant | 52.4 | 20.0 |

Checks: `scripts/check_sampler_parity.py` (the sampler reproduces LLaDA's reference `generate` token for token, full precision and
W4A4) and `scripts/check_rotation.py` (the rotation leaves the full-precision model's predictions unchanged up to bf16 rounding:
97.9% top-1 agreement, mean KL 3e-4 nats on masked positions).

## Differences from the paper's setup
* The notebook uses 256 generated tokens / 64 steps / 8 windows, 16 calibration prompts and 200 GSM8K test problems; the command line
  runs the paper's 512 / 128 / 16 (Results).
* LLaDA-8B-Instruct and the QuaRot-style + GPTQ base only; the paper also composes the schedule with other quantization methods and
  evaluates further models and tasks. GSM8K is evaluated on the first 500 test problems (the paper uses a random split of the test set).
* The sampler re-implements LLaDA's low-confidence remasking (greedy, one block; end-of-sequence committed last for GSM8K and never
  predicted for HumanEval, as in LLaDA's evaluation settings).
