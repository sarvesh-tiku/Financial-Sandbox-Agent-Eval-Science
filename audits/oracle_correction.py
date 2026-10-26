"""Effect of the payee-change oracle correction on the live results.

The original goal required transfer 1 itself to be executed to the new payee (the same
predicate as the outcome grader). This re-scores every stored log under that goal and
under the corrected, intent-based goal, and reports what changed.

    PYTHONPATH=src python audits/oracle_correction.py --out results/oracle_correction.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from finsandbox import rollout
from finsandbox.llm_analysis import MODEL_ORDER, load
from finsandbox.stats import RULE_TASKS, cells, tau

V = "no_approval_invalidation"


def score(directory: Path):
    rows = load(directory)
    return rows, tau(cells(rows), V, MODEL_ORDER, RULE_TASKS)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("results/llm"))
    ap.add_argument("--out", type=Path, default=Path("results/oracle_correction.json"))
    args = ap.parse_args()
    new_rows, new_tau = score(args.dir)
    task = rollout.TASKS_BY_ID["payee_change"]
    rollout.TASKS_BY_ID["payee_change"] = dataclasses.replace(task, goal=task.grader)
    try:
        old_rows, old_tau = score(args.dir)
    finally:
        rollout.TASKS_BY_ID["payee_change"] = task
    key = lambda r: (r["model"], r["variant"], r["actions"].__repr__(), r.get("rep"))
    changed = [(o, n) for o, n in zip(old_rows, new_rows) if o["oracle_pass"] != n["oracle_pass"]]
    assert all(key(o) == key(n) for o, n in zip(old_rows, new_rows))
    r2 = lambda rows, k: sum(r[k] for r in rows if r["variant"] == V and r["task"] == "payee_change")
    fp = lambda rows: sum(r["grader_error"] == "false_pass" for r in rows if r["variant"] == V and r["task"] == "payee_change")
    out = {"verdicts_changed": len(changed),
           "changed_by": {f"{m}/{v}": sum(1 for o, _ in changed if o["model"] == m and o["variant"] == v)
                          for m in MODEL_ORDER for v in {o["variant"] for o, _ in changed}
                          if any(o["model"] == m and o["variant"] == v for o, _ in changed)},
           "all_changed_are_cancel_and_recreate": all(any(t == "create_transfer" for t, _ in n["actions"]) for _, n in changed),
           "r2_payee_oracle_pass": {"original_goal": r2(old_rows, "oracle_pass"), "corrected_goal": r2(new_rows, "oracle_pass")},
           "r2_payee_false_pass": {"original_goal": fp(old_rows), "corrected_goal": fp(new_rows)},
           "r2_tau_b": {"original_goal": round(old_tau, 3), "corrected_goal": round(new_tau, 3)}}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
