"""CPU tests: rotation leaves the network function unchanged, fake-quantization has 2^b levels, the DP is exactly optimal,
and the sampler commits every position with the requested per-step budget. Run: python -m pytest tests -q"""
import itertools
from types import SimpleNamespace

import torch
from torch import nn

from t2quant import ActQuant, allocate, apply_rotation, fake_quant, generate
from t2quant.sampler import MASK_ID


class _Norm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.rand(d) + 0.5)

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class _Block(nn.Module):
    def __init__(self, d, h, f):
        super().__init__(); self.h = h
        self.attn_norm, self.ff_norm = _Norm(d), _Norm(d)
        self.q_proj, self.k_proj, self.v_proj, self.attn_out = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.ff_proj, self.up_proj, self.ff_out = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

    def forward(self, x):
        B, L, D = x.shape; hd = D // self.h
        a = self.attn_norm(x); sp = lambda t: t.view(B, L, self.h, hd).transpose(1, 2)  # noqa: E731
        q, k, v = sp(self.q_proj(a)), sp(self.k_proj(a)), sp(self.v_proj(a))
        att = torch.softmax(q @ k.transpose(-1, -2) / hd ** 0.5, -1) @ v
        x = x + self.attn_out(att.transpose(1, 2).reshape(B, L, D))
        m = self.ff_norm(x)
        return x + self.ff_out(torch.nn.functional.silu(self.ff_proj(m)) * self.up_proj(m)), None   # (hidden, cache) like LLaDA


class TinyLLaDA(nn.Module):
    """LLaDA-shaped mock (model.transformer.{wte, blocks, ln_f, ff_out}, config.n_heads / d_model) for CPU tests."""

    def __init__(self, V=130000, d=16, h=2, f=48, n=2):
        super().__init__(); torch.manual_seed(0)
        self.config = SimpleNamespace(n_heads=h, d_model=d)
        self.transformer = nn.Module(); self.transformer.wte = nn.Embedding(V, d)
        self.transformer.blocks = nn.ModuleList([_Block(d, h, f) for _ in range(n)])
        self.transformer.ln_f = _Norm(d); self.transformer.ff_out = nn.Linear(d, V, bias=False)

    def forward(self, ids):
        x = self.transformer.wte(ids)
        for b in self.transformer.blocks:
            x = b(x)[0]
        return SimpleNamespace(logits=self.transformer.ff_out(self.transformer.ln_f(x)))


def test_rotation_is_exact():
    m = TinyLLaDA().double(); ids = torch.randint(0, 1000, (1, 12))
    ref = m(ids).logits.clone(); apply_rotation(m, seed=0)
    assert torch.allclose(m(ids).logits, ref, atol=1e-8)


def test_fake_quant_levels():
    x = torch.randn(4, 64)
    for b in (3, 4, 8):
        assert all(len(torch.unique(r)) <= 2 ** b for r in fake_quant(x, b))


def test_dp_matches_brute_force():
    torch.manual_seed(1); K, levels, b0 = 6, (3, 4, 8), 4
    gain = {3: -torch.rand(K), 4: torch.zeros(K), 8: torch.rand(K) * 2}
    sched = allocate(gain, levels, b0)
    assert sum(sched) == K * b0
    best = max((s for s in itertools.product(levels, repeat=K) if sum(s) == K * b0), key=lambda s: sum(float(gain[b][k]) for k, b in enumerate(s)))
    assert abs(sum(float(gain[b][k]) for k, b in enumerate(sched)) - sum(float(gain[b][k]) for k, b in enumerate(best))) < 1e-9


def test_sampler_commits_everything():
    m = TinyLLaDA(); q = ActQuant(m); out, tr = generate(m, q, torch.randint(0, 1000, (1, 5)), gen_len=32, steps=8, schedule=[4] * 4, windows=4, record=True)
    assert (out != MASK_ID).all() and len(tr["commits"]) == 8 and all(len(c) == 4 for c in tr["commits"])
    assert ((tr["states"] == MASK_ID).sum(1) == torch.arange(32, 0, -4)).all()


def test_calibrate_and_allocate_end_to_end():
    from t2quant import calibrate
    m = TinyLLaDA(); q = ActQuant(m); prompts = [torch.randint(0, 1000, (1, 5)) for _ in range(2)]
    cal = calibrate(m, q, TinyLLaDA(), prompts, levels=(3, 4, 8), b0=4, gen_len=32, steps=8, windows=4, batch=4, log=lambda *_: None)
    assert cal["iota"][-1] == 0 and (cal["iota"][:-1] >= 0).all()          # nothing is masked after the last window
    assert all((cal["kappa"][b] >= 0).all() for b in (3, 4, 8)) and (cal["gain"][4] == 0).all()
    sched = allocate(cal["gain"], (3, 4, 8), 4)
    assert len(sched) == 4 and sum(sched) == 16


def test_gptq_quantizes_to_4_bit_grid_and_keeps_function_close():
    from t2quant import apply_gptq
    m = TinyLLaDA().float(); ids = torch.randint(0, 1000, (16, 12)); ref = m(ids[:2]).logits.clone()
    apply_rotation(m, seed=0); apply_gptq(m, ids, bits=4, batch=8, log=lambda *_: None)
    for b in m.transformer.blocks:
        for n in ("q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"):
            assert all(len(torch.unique(r)) <= 16 for r in getattr(b, n).weight.data)
    out = m(ids[:2]).logits
    assert torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0) > 0.9


class _CharTok:
    """Tokenizer stub: token i is the character chr(i) (special ids >= 126000 decode to nothing)."""
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i < 126000)


def test_relevant_positions_stop_at_question():
    from t2quant.data import relevant_positions
    text = "The answer is 3 Question: x"; ids = [ord(c) for c in text] + [126081] * 3
    R = relevant_positions(ids, _CharTok())
    assert R == set(range(text.index("Question:")))


def test_calibrate_with_relevance_and_fp_trajectory():
    from t2quant import calibrate
    m = TinyLLaDA(); q = ActQuant(m); prompts = [torch.randint(0, 1000, (1, 5)) for _ in range(2)]
    for kw in (dict(relevant=lambda final: set(range(8))), dict(trajectory="fp")):
        cal = calibrate(m, q, TinyLLaDA(), prompts, levels=(3, 4, 8), b0=4, gen_len=32, steps=8, windows=4, batch=4, log=lambda *_: None, **kw)
        assert all((cal["kappa"][b] >= 0).all() for b in (3, 4, 8)) and (cal["gain"][4] == 0).all()


def test_transfer_counts_match_llada():
    from t2quant.sampler import transfer_counts
    assert transfer_counts(256, 64) == [4] * 64
    c = transfer_counts(10, 4); assert c == [3, 3, 2, 2] and sum(c) == 10


def test_humaneval_scoring_and_relevance():
    from t2quant import data
    row = {"question": "def add(a, b):\n    \"\"\"add\"\"\"\n", "test": "def check(f):\n    assert f(1, 2) == 3\n", "entry_point": "add"}
    assert data.humaneval_correct("Sure:\n```python\ndef add(a, b):\n    return a + b\n```", row)
    assert data.humaneval_correct("    return a + b", row) and not data.humaneval_correct("    return a - b", row)
    text = "x ```python\ndef f():\n    pass\n``` done"; ids = [ord(c) for c in text]
    R = data.humaneval_relevant_positions(ids, _CharTok()); s = text.index("def"); e = text.index("```", s)
    assert R == set(range(s, e))


def test_report_statistics(tmp_path):
    import json
    from t2quant.cli import cmd_report
    from types import SimpleNamespace
    ok = {"fp": [1, 1, 1, 0], "uniform": [1, 0, 0, 0], "t2quant": [1, 1, 0, 1]}
    for c, v in ok.items():
        (tmp_path / f"preds_{c}.shard0of1.jsonl").write_text("".join(json.dumps({"idx": i, "ok": bool(x)}) + "\n" for i, x in enumerate(v)))
    cmd_report(SimpleNamespace(out=str(tmp_path)))
    rep = (tmp_path / "report.txt").read_text()
    assert "4 problems" in rep and "fixed 2, broke 0" in rep and "75.0%" in rep
