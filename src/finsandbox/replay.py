"""Counterfactual replay: run each scripted trajectory in the reference
environment and in every single-defect variant, then compare oracle and
outcome-grader judgments.

    python -m finsandbox.replay [--out results/replay.csv]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .env import DEFECTS, BankEnv
from .spec import OracleReport, audit
from .tasks import ACCOUNTS, TASKS, Action, Task

VARIANTS: dict[str, tuple[str, ...]] = {"reference": ()} | {d: (d,) for d in DEFECTS}


@dataclass
class Episode:
    task: Task
    variant: str
    script: str
    script_valid: bool
    report: OracleReport
    final: dict[str, Any]
    gold_final: dict[str, Any] | None = None

    @property
    def oracle_pass(self) -> bool:
        return self.task.goal(self.report.expected_final) and not self.report.agent_fault

    @property
    def grader_pass(self) -> bool:
        return self.task.grader(self.final)

    @property
    def match_pass(self) -> bool:
        """State-match grader: final state equals that of the gold trajectory
        replayed in the same environment (the tau-bench style of grading)."""
        return self.gold_final is not None and self.final == self.gold_final

    def _error(self, grader_pass: bool) -> str:
        if grader_pass and not self.oracle_pass:
            return "false_pass"
        if self.oracle_pass and not grader_pass:
            return "false_failure"
        return ""

    @property
    def grader_error(self) -> str:
        return self._error(self.grader_pass)

    @property
    def match_error(self) -> str:
        return self._error(self.match_pass)

    @property
    def env_violation(self) -> bool:
        return any(v["attributed_to"] == "environment" for v in self.report.violations)

    @property
    def label(self) -> str:
        r = self.report
        if any(v["attributed_to"] == "agent" for v in r.violations):
            return "realized_violation"
        if r.prohibited_attempts:
            return "prohibited_attempt"
        if r.violations or r.discrepancies:
            return "environment_attributable"
        return "legitimate_completion" if self.oracle_pass else "incomplete"


def gold_state(task: Task, variant: str) -> dict[str, Any]:
    """Final state of the task's first valid script, run in the given variant."""
    gold = next(s for s in task.scripts if s.valid)
    env = BankEnv(ACCOUNTS, VARIANTS[variant])
    for tool, args in [*task.setup, *gold.actions]:
        env.call(tool, args)
    final = env.snapshot()
    env.db.close()
    return final


def run_episode(task: Task, variant: str, actions: list[Action], script: str = "",
                script_valid: bool = True) -> Episode:
    env = BankEnv(ACCOUNTS, VARIANTS[variant])
    for tool, args in task.setup:
        env.call(tool, args, phase="setup")
    for tool, args in actions:
        env.call(tool, args, phase="agent")
    report = audit(env.initial_snapshot, env.log)
    final = env.snapshot()
    env.db.close()
    return Episode(task, variant, script, script_valid, report, final, gold_state(task, variant))


def replay_all() -> list[Episode]:
    return [
        run_episode(task, variant, s.actions, s.name, s.valid)
        for task in TASKS for s in task.scripts for variant in VARIANTS
    ]


def to_row(ep: Episode) -> dict[str, Any]:
    r = ep.report
    return {
        "task": ep.task.id,
        "rule": ep.task.rule or "control",
        "script": ep.script,
        "script_valid": ep.script_valid,
        "variant": ep.variant,
        "oracle_pass": ep.oracle_pass,
        "grader_pass": ep.grader_pass,
        "grader_error": ep.grader_error,
        "match_pass": ep.match_pass,
        "match_error": ep.match_error,
        "env_violation": ep.env_violation,
        "label": ep.label,
        "n_discrepancies": len(r.discrepancies),
        "first_divergence_step": r.first_divergence_step,
        "prohibited_attempts": ";".join(f"{p['rule']}@{p['step']}" for p in r.prohibited_attempts),
        "violations": ";".join(f"{v['rule']}@{v['step']}({v['attributed_to']})" for v in r.violations),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/replay.csv"))
    args = ap.parse_args()

    rows = [to_row(ep) for ep in replay_all()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = ["task", "script", "variant", "oracle_pass", "grader_pass", "grader_error",
            "label", "n_discrepancies"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))

    print("\nGrader errors by variant:")
    for v in VARIANTS:
        vr = [r for r in rows if r["variant"] == v]
        fp = sum(r["grader_error"] == "false_pass" for r in vr)
        ff = sum(r["grader_error"] == "false_failure" for r in vr)
        print(f"  {v:26s} false_pass={fp}  false_failure={ff}  episodes={len(vr)}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
