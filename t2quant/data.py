"""GSM8K prompts (LLaDA's 2-shot chat format) and scorer. Calibration uses training questions only (no answers are read);
evaluation uses the first test problems."""
from __future__ import annotations

import re

SHOTS = [
    ("Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 "
     "chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of "
     "their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to "
     "study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for "
     "lunch each day?", "The answer is 4"),
    ("Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the "
     "3 pointers and free throws.  What's the total number of points scored by both teams added together?", "The answer is 201"),
]


def prompt_ids(tok, question: str):
    msgs = []
    for q, a in SHOTS:
        msgs += [{"role": "user", "content": f"Question: {q}\nLet's think step by step\nAnswer:"}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"Question: {question}\nLet's think step by step\nAnswer:"})
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


def load(split: str, n: int):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": r["question"], "gold": r["answer"].split("####")[-1].replace(",", "").strip()} for r in ds.select(range(n))]


def correct(text: str, gold: str) -> bool:
    text = text.split("Question:")[0]
    m = re.search(r"[Tt]he [Aa]nswer is[:\s]*(-?\d+(?:\.\d+)?)", text)
    pred = m.group(1) if m else (re.findall(r"-?\d+\.?\d*", text.replace(",", "")) or ["NULL"])[-1]
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


# ---- HumanEval (code; zero-shot, LLaDA's prompt; end-of-sequence banned as in LLaDA's code evaluation) ----------------------------
def humaneval_load(n: int | None = None):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = [{"question": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]} for r in ds]
    return rows[:n] if n else rows


def humaneval_prompt_ids(tok, prompt: str):
    text = tok.apply_chat_template([{"role": "user", "content": "Complete the following python code:\n" + prompt}],
                                   add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


def _code(prompt: str, text: str) -> str:
    import re as _re
    m = _re.search(r"```(?:python)?\n(.*?)```", text, _re.S); body = m.group(1) if m else text
    return body if body.strip().startswith("def") else prompt + body


def humaneval_correct(text: str, row: dict, timeout: float = 10.0) -> bool:
    """Runs the completed function against the benchmark's tests in a separate Python process (pass@1)."""
    import os, subprocess, sys, tempfile
    code = _code(row["question"], text) + "\n" + row["test"] + f"\ncheck({row['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        return subprocess.run([sys.executable, path], capture_output=True, timeout=timeout, cwd=tempfile.gettempdir()).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


# ---- task registry: calibration prompts never overlap the evaluation problems ---------------------------------------------------
N_CALIB = {"gsm8k": 16, "humaneval": 50}       # default number of calibration prompts per task


def task(name: str, tok, n_calib: int | None, n_eval: int | None):
    """-> dict(calib=[prompt ids], eval=[(prompt ids, scorer(text) -> bool)], eos=str)."""
    n_calib = N_CALIB[name] if n_calib is None else n_calib
    if name == "gsm8k":
        cal = [prompt_ids(tok, r["question"]) for r in load("train", n_calib)]
        ev = [(prompt_ids(tok, r["question"]), (lambda text, g=r["gold"]: correct(text, g))) for r in load("test", n_eval or 1319)]
        return dict(calib=cal, eval=ev, eos="last")
    if name == "humaneval":     # no training split: a fixed shuffle (seed 0) of the 164 problems; the first n_calib calibrate, the next evaluate
        import random
        rows = humaneval_load(); order = list(range(len(rows))); random.Random(0).shuffle(order)
        cal_rows = [rows[i] for i in order[:n_calib]]; ev_rows = [rows[i] for i in order[n_calib:]][:n_eval]
        return dict(calib=[humaneval_prompt_ids(tok, r["question"]) for r in cal_rows],
                    eval=[(humaneval_prompt_ids(tok, r["question"]), (lambda text, r=r: humaneval_correct(text, r))) for r in ev_rows],
                    eos="ban")
    raise ValueError(f"unknown task {name!r} (gsm8k, humaneval)")
