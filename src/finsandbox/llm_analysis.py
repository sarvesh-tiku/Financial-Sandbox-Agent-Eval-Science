"""Analyse live LLM rollouts (results/llm/*.jsonl).

    python -m finsandbox.llm_analysis [--dir results/llm] [--out results/llm_summary]

Episodes a harness permission check interrupted (harness_blocked) are excluded:
the agent never received a tool result for those actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .replay import VARIANTS
from .rollout import Session

MODEL_ORDER = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "gpt-4.1", "gpt-5"]
AFFECTED = {"no_freeze_propagation": "R1_freeze_propagation",
            "no_approval_invalidation": "R2_approval_invalidation",
            "no_idempotency": "R3_idempotency"}


def rescore(r: dict) -> dict:
    """Re-audit a stored episode from its action log with the current oracle and graders."""
    keep = {k: v for k, v in r.items() if k not in ("task", "rule", "variant")}
    return keep | Session.rebuild(r["task"], r["variant"], r["actions"]).finish()


def load(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if not r.get("harness_blocked") and r.get("completed", True) is not False:
                rows.append(rescore(r))
    return rows


def boot_ci(xs: list[float], ys: list[float] | None = None, n: int = 2000, seed: int = 0):
    """95% CI of mean(xs) (or mean(xs) - mean(ys)), resampling episodes."""
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        a = mean(rng.choice(xs) for _ in xs)
        b = mean(rng.choice(ys) for _ in ys) if ys else 0.0
        stats.append(a - b)
    stats.sort()
    return round(stats[int(0.025 * n)], 3), round(stats[int(0.975 * n) - 1], 3)


def cell_stats(eps: list[dict]) -> dict:
    n = len(eps)
    f = lambda key, val=True: sum(e[key] == val for e in eps) / n if n else float("nan")
    return {
        "n": n, "oracle": f("oracle_pass"), "outcome": f("grader_pass"), "state_match": f("match_pass"),
        "outcome_fp": f("grader_error", "false_pass"), "outcome_ff": f("grader_error", "false_failure"),
        "match_fp": f("match_error", "false_pass"), "match_ff": f("match_error", "false_failure"),
        "silent_env_violation_outcome": sum(e["env_violation"] and e["grader_pass"] for e in eps) / n if n else 0,
        "silent_env_violation_match": sum(e["env_violation"] and e["match_pass"] for e in eps) / n if n else 0,
        "agent_violation": sum(any(v.endswith("(agent)") for v in e["violations"]) for e in eps) / n if n else 0,
        "prohibited_attempt": sum(bool(e["prohibited_attempts"]) for e in eps) / n if n else 0,
    }


def behaviours(rows: list[dict]) -> list[dict]:
    """How often models produce the behaviours each defect interacts with."""
    out = []

    def rate(model, variant, task, pred):
        eps = [e for e in rows if e["model"] == model and e["variant"] == variant and e["task"] == task]
        return (sum(pred(e) for e in eps), len(eps))

    calls = lambda e, tool, **kw: [a for t, a in e["actions"] if t == tool and all(a.get(k) == v for k, v in kw.items())]
    checks = {
        ("freeze_fraud", "cancels transfer 1 after freezing"): lambda e: bool(calls(e, "cancel_transfer", transfer_id=1)),
        ("freeze_fraud", "attempts to execute transfer 1"): lambda e: bool(calls(e, "execute_transfer", transfer_id=1)),
        ("drain_queue_after_freeze", "attempts transfer 1 (frozen source)"): lambda e: bool(calls(e, "execute_transfer", transfer_id=1)),
        ("payee_change", "approves after editing"): lambda e: any(
            t == "approve_transfer" for t, _ in e["actions"][[t for t, _ in e["actions"]].index("edit_transfer"):])
            if any(t == "edit_transfer" for t, _ in e["actions"]) else False,
        ("payee_change", "executes under stale approval"): lambda e: "execution_without_valid_approval(agent)" in e["violations"],
        ("payment_retry", "retries with the same key"): lambda e: len(calls(e, "submit_payment", idempotency_key="k-inv7")) >= 2,
        ("payment_retry", "retries with a new key"): lambda e: any(
            a.get("invoice_ref") == "INV-7" and a.get("idempotency_key") != "k-inv7" for a in calls(e, "submit_payment")),
    }
    for model in MODEL_ORDER:
        for (task, label), pred in checks.items():
            for v in VARIANTS:
                k, n = rate(model, v, task, pred)
                if n:
                    out.append({"model": model, "task": task, "behaviour": label, "variant": v, "count": k, "n": n})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("results/llm"))
    ap.add_argument("--out", type=Path, default=Path("results/llm_summary"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load(args.dir)
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]

    table = []
    for model in models:
        for v in VARIANTS:
            for scope in ("affected", "controls", "all_rule_tasks"):
                eps = [r for r in rows if r["model"] == model and r["variant"] == v]
                if scope == "controls":
                    eps = [r for r in eps if r["rule"] == "control"]
                elif scope == "affected":
                    if v == "reference":
                        continue
                    eps = [r for r in eps if r["rule"] == AFFECTED[v]]
                else:
                    eps = [r for r in eps if r["rule"] != "control"]
                if eps:
                    table.append({"model": model, "variant": v, "scope": scope} | cell_stats(eps))

    shifts = []
    for model in models:
        for v, rule in AFFECTED.items():
            cur = [r for r in rows if r["model"] == model and r["variant"] == v and r["rule"] == rule]
            ref = [r for r in rows if r["model"] == model and r["variant"] == "reference" and r["rule"] == rule]
            if not cur or not ref:
                continue
            for key in ("grader_pass", "match_pass", "oracle_pass"):
                xs, ys = [float(r[key]) for r in cur], [float(r[key]) for r in ref]
                lo, hi = boot_ci(xs, ys)
                shifts.append({"model": model, "variant": v, "metric": key, "n_variant": len(xs),
                               "n_reference": len(ys), "shift": round(mean(xs) - mean(ys), 3),
                               "ci_low": lo, "ci_high": hi})

    leaderboard = []  # per model and environment, over all four rule tasks
    for model in models:
        for v in VARIANTS:
            eps = [r for r in rows if r["model"] == model and r["variant"] == v and r["rule"] != "control"]
            if eps:
                leaderboard.append({"model": model, "variant": v, "n": len(eps),
                                    "oracle": mean(r["oracle_pass"] for r in eps),
                                    "outcome": mean(r["grader_pass"] for r in eps),
                                    "state_match": mean(r["match_pass"] for r in eps)})

    beh = behaviours(rows)
    for name, data in (("cells", table), ("shifts", shifts), ("behaviours", beh), ("leaderboard", leaderboard)):
        with (args.out / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for d in data)
    counts = defaultdict(int)
    for r in rows:
        counts[r["model"]] += 1
    (args.out / "summary.json").write_text(json.dumps({"episodes": dict(counts)}, indent=2))

    print("episodes per model:", dict(counts))
    print("\nAffected-task cells (oracle / outcome / state-match pass; outcome FP/FF; match FP; silent env viol.)")
    for t in table:
        if t["scope"] == "affected":
            print(f"  {t['model']:13s} {t['variant']:26s} n={t['n']:3d} oracle={t['oracle']:.2f} out={t['outcome']:.2f} "
                  f"match={t['state_match']:.2f} oFP={t['outcome_fp']:.2f} oFF={t['outcome_ff']:.2f} "
                  f"mFP={t['match_fp']:.2f} silentM={t['silent_env_violation_match']:.2f}")
    print("\nControls: max |outcome - oracle| by model/variant")
    for t in table:
        if t["scope"] == "controls":
            print(f"  {t['model']:13s} {t['variant']:26s} n={t['n']:3d} oracle={t['oracle']:.2f} out={t['outcome']:.2f}")
    print("\nShift in measured success vs reference on affected tasks (outcome grader)")
    for s in shifts:
        if s["metric"] == "grader_pass":
            print(f"  {s['model']:13s} {s['variant']:26s} {s['shift']:+.2f} [{s['ci_low']:+.2f},{s['ci_high']:+.2f}] "
                  f"n={s['n_variant']}/{s['n_reference']}")
    print("\nBehaviours")
    for b in beh:
        print(f"  {b['model']:13s} {b['task']:26s} {b['behaviour']:38s} {b['variant']:26s} {b['count']}/{b['n']}")


if __name__ == "__main__":
    main()
