"""Uncertainty and sensitivity analysis for the live-agent results.

Sampling unit: one episode inside a fixed (model, task, environment) cell. The nine
tasks are fixed templates, not a sample from a task population, so every interval
here is conditional on this task set; nothing below supports claims about other tasks.

    PYTHONPATH=src python -m finsandbox.stats [--boot 5000] [--out results/stats.json]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from .experiments import kendall_tau_b
from .llm_analysis import AFFECTED, MODEL_ORDER, load

RULE_TASKS = ["freeze_fraud", "drain_queue_after_freeze", "payee_change", "payment_retry"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def cells(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r["model"], r["task"], r["variant"])].append(r)
    return out


def model_scores(cell_map, variant, models, tasks, key):
    """Task-weighted success per model: mean over tasks of the per-task rate."""
    return {m: sum(sum(e[key] for e in cell_map[(m, t, variant)]) / len(cell_map[(m, t, variant)])
                   for t in tasks) / len(tasks) for m in models}


def tau(cell_map, variant, models, tasks, key="grader_pass"):
    meas = model_scores(cell_map, variant, models, tasks, key)
    true = model_scores(cell_map, variant, models, tasks, "oracle_pass")
    return kendall_tau_b([meas[m] for m in models], [true[m] for m in models]), meas, true


def bootstrap_tau(cell_map, variant, models, tasks, key, n_boot, seed):
    rng = random.Random(seed)
    taus, gpt5_below, inversions = [], 0, []
    for _ in range(n_boot):
        res = {}
        for m in models:
            for t in tasks:
                eps = cell_map[(m, t, variant)]
                res[(m, t, variant)] = [rng.choice(eps) for _ in eps]
        tb, meas, true = tau(res, variant, models, tasks, key)
        if not math.isnan(tb):
            taus.append(tb)
        inv = sum(1 for a, b in combinations(models, 2) if (true[a] - true[b]) * (meas[a] - meas[b]) < 0)
        inversions.append(inv)
        if "gpt-5" in models:
            better = sum(1 for m in models if m != "gpt-5" and meas[m] > meas["gpt-5"])
            gpt5_below += better >= 2 and all(true["gpt-5"] >= true[m] for m in models)
    taus.sort()
    q = lambda p: round(taus[min(len(taus) - 1, int(p * len(taus)))], 3)
    return {"median": q(0.5), "ci95": [q(0.025), q(0.975)],
            "p_tau_below_0.5": round(sum(t < 0.5 for t in taus) / len(taus), 3),
            "p_at_least_one_inversion": round(sum(i > 0 for i in inversions) / len(inversions), 3),
            "p_gpt5_true_best_but_ranked_below_two": round(gpt5_below / n_boot, 3) if "gpt-5" in models else None}


def clean(x):
    """Replace NaN (undefined tau or empty interval) with None so the output is valid JSON."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("results/llm"))
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/stats.json"))
    args = ap.parse_args()
    rows = load(args.dir)
    cm = cells(rows)
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    out = {"sampling_unit": "episode within a fixed (model, task, environment) cell; intervals are "
                            "conditional on the nine fixed tasks", "rates": {}, "ranking": {}}

    # Pooled grader-error rates on affected tasks, with Wilson intervals.
    for v, rule in AFFECTED.items():
        eps = [r for r in rows if r["variant"] == v and r["rule"] == rule]
        correct = [e for e in eps if e["oracle_pass"]]
        wrong = [e for e in eps if not e["oracle_pass"]]
        for g, err in (("outcome", "grader_error"), ("state_match", "match_error")):
            ff = sum(e[err] == "false_failure" for e in correct)
            fp = sum(e[err] == "false_pass" for e in wrong)
            silent = sum(e["env_violation"] and e["grader_pass" if g == "outcome" else "match_pass"] for e in eps)
            out["rates"][f"{v}/{g}"] = {
                "false_failure": [ff, len(correct), wilson(ff, len(correct))],
                "false_pass": [fp, len(wrong), wilson(fp, len(wrong))],
                "silent": [silent, len(eps), wilson(silent, len(eps))]}
        # Per-model breakdown, so no model dominates the pooled number unseen.
        out["rates"][f"{v}/per_model_outcome_ff"] = {
            m: [sum(e["grader_error"] == "false_failure" for e in correct if e["model"] == m),
                sum(1 for e in correct if e["model"] == m)] for m in models}

    # Ranking agreement: point estimate, episode bootstrap, leave-one-out sensitivity.
    for v in ["reference", *AFFECTED]:
        for g, key in (("outcome", "grader_pass"), ("state_match", "match_pass")):
            tb, meas, true = tau(cm, v, models, RULE_TASKS, key)
            entry = {"tau_b": round(tb, 3) if not math.isnan(tb) else None,
                     "measured": {m: round(x, 3) for m, x in meas.items()},
                     "true": {m: round(x, 3) for m, x in true.items()},
                     "bootstrap": bootstrap_tau(cm, v, models, RULE_TASKS, key, args.boot, args.seed),
                     "leave_one_task_out": {t: round(tau(cm, v, models, [x for x in RULE_TASKS if x != t], key)[0], 3)
                                            for t in RULE_TASKS},
                     "leave_one_model_out": {m: round(tau(cm, v, [x for x in models if x != m], RULE_TASKS, key)[0], 3)
                                             for m in models}}
            out["ranking"][f"{v}/{g}"] = entry
    text = json.dumps(clean(out), indent=1)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
