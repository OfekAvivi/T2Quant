"""Exact budget-constrained allocation by dynamic programming.

Choose one bit-width b_k in `levels` per window to maximise sum_k gain[b_k][k] subject to sum_k cost(b_k) == budget (default:
the cost of the uniform b0 schedule, i.e. the same average activation bits). With integer costs, F_k(z) = max_b F_{k-1}(z - c(b)) + g_k(b)
is exact (principle of optimality); backtracking from F_K(budget) gives the schedule. The uniform schedule is always feasible.
"""
from __future__ import annotations


def allocate(gain: dict, levels, b0: int, cost=lambda b: b, budget: int | None = None) -> list[int]:
    K = len(next(iter(gain.values()))); budget = K * cost(b0) if budget is None else budget
    NEG = float("-inf"); F = [{0: 0.0}]; arg = [{}]
    for k in range(K):
        Fk, Ak = {}, {}
        for z, v in F[k].items():
            for b in levels:
                z2 = z + cost(b); val = v + float(gain[b][k])
                if z2 <= budget and val > Fk.get(z2, NEG) + 1e-12:
                    Fk[z2], Ak[z2] = val, (b, z)
        F.append(Fk); arg.append(Ak)
    if budget not in F[K]:
        raise ValueError("infeasible budget")
    sched, z = [], budget
    for k in range(K, 0, -1):
        b, z = arg[k][z]; sched.append(b)
    return sched[::-1]
