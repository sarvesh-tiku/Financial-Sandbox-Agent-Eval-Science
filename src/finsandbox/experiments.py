"""Experiments that need no model API.

E1  Counterfactual replay: every scripted trajectory in every variant.
E2  Behaviour profiles: deterministic agents that are competent on a subset of
    skills. Measures how each defect moves measured (grader) success away from
    true (oracle) success, and whether it reorders agents.
E3  Stochastic paired agents: profiles that slip at random, run with the same
    random draws in every variant (paired), to estimate per-defect shifts in
    measured success with bootstrap confidence intervals.

    python -m finsandbox.experiments [--out results] [--episodes 50] [--seed 0]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from .replay import VARIANTS, replay_all, to_row
from .tasks import TASKS

SKILLS = ("freeze_awareness", "approval_discipline", "retry_discipline")
DEFECT_OF_VARIANT = {"no_freeze_propagation": "R1_freeze_propagation",
                     "no_approval_invalidation": "R2_approval_invalidation",
                     "no_idempotency": "R3_idempotency"}


GRADERS = ("outcome", "state_match")


def outcome_table(episodes) -> dict[tuple[str, str, bool, str], tuple[bool, bool]]:
    """(grader, task, script_valid, variant) -> (grader_pass, oracle_pass)."""
    table = {}
    for e in episodes:
        for g, passed in (("outcome", e.grader_pass), ("state_match", e.match_pass)):
            table[(g, e.task.id, e.script_valid, e.variant)] = (passed, e.oracle_pass)
    return table


def _outcome(table, grader: str, task, valid: bool, variant: str) -> tuple[bool, bool]:
    if (grader, task.id, valid, variant) not in table:  # task has no invalid script
        valid = True
    return table[(grader, task.id, valid, variant)]


# -- E1 ---------------------------------------------------------------------


def e1_summary(episodes) -> list[dict[str, Any]]:
    ref = {(e.task.id, e.script): e for e in episodes if e.variant == "reference"}
    out = []
    for v in VARIANTS:
        for scope in ("rule_tasks", "controls"):
            eps = [e for e in episodes if e.variant == v and (e.task.rule is None) == (scope == "controls")]
            row = {"variant": v, "scope": scope, "episodes": len(eps),
                   "episodes_with_discrepancy": sum(bool(e.report.discrepancies) for e in eps),
                   "env_violations": sum(e.env_violation for e in eps)}
            for g, err, passed in (("outcome", "grader_error", "grader_pass"),
                                   ("state_match", "match_error", "match_pass")):
                row[f"{g}_false_pass"] = sum(getattr(e, err) == "false_pass" for e in eps)
                row[f"{g}_false_failure"] = sum(getattr(e, err) == "false_failure" for e in eps)
                row[f"{g}_silent_env_violation"] = sum(getattr(e, passed) and e.env_violation for e in eps)
                row[f"{g}_changed_vs_reference"] = sum(
                    getattr(e, passed) != getattr(ref[(e.task.id, e.script)], passed) for e in eps)
            out.append(row)
    return out


# -- E2 ---------------------------------------------------------------------


def profiles() -> dict[str, frozenset[str]]:
    out = {}
    for k in range(len(SKILLS) + 1):
        for combo in itertools.combinations(SKILLS, k):
            name = "+".join(s.split("_")[0] for s in combo) or "none"
            out[name] = frozenset(combo)
    return out


def e2_profiles(table) -> list[dict[str, Any]]:
    rows = []
    for grader in GRADERS:
        for name, skills in profiles().items():
            for v in VARIANTS:
                g = o = 0
                for task in TASKS:
                    valid = task.skill is None or task.skill in skills
                    gp, op = _outcome(table, grader, task, valid, v)
                    g += gp
                    o += op
                rows.append({"grader": grader, "profile": name, "variant": v,
                             "measured": g, "true": o, "n_tasks": len(TASKS)})
    return rows


def kendall_tau_b(xs: list[float], ys: list[float]) -> float:
    c = d = tx = ty = 0
    for i, j in itertools.combinations(range(len(xs)), 2):
        dx, dy = xs[i] - xs[j], ys[i] - ys[j]
        c += dx * dy > 0
        d += dx * dy < 0
        tx += dx == 0 and dy != 0
        ty += dy == 0 and dx != 0
    denom = ((c + d + tx) * (c + d + ty)) ** 0.5
    return (c - d) / denom if denom else float("nan")


def e2_rankings(rows) -> list[dict[str, Any]]:
    out = []
    names = list(profiles())
    for grader, v in itertools.product(GRADERS, VARIANTS):
        by = {r["profile"]: r for r in rows if r["variant"] == v and r["grader"] == grader}
        m = [by[n]["measured"] for n in names]
        t = [by[n]["true"] for n in names]
        inversions, ties = [], []
        for a, b in itertools.combinations(names, 2):
            dt = by[a]["true"] - by[b]["true"]
            dm = by[a]["measured"] - by[b]["measured"]
            if dt and dm and (dt > 0) != (dm > 0):
                inversions.append(f"{a} vs {b}")
            elif dt and not dm:
                ties.append(f"{a} vs {b}")
        distinct = sum(by[a]["true"] != by[b]["true"] for a, b in itertools.combinations(names, 2))
        out.append({"grader": grader, "variant": v, "kendall_tau_b": round(kendall_tau_b(m, t), 3),
                    "n_pairs_true_distinct": distinct,
                    "correctly_ordered": distinct - len(inversions) - len(ties),
                    "inverted_pairs": inversions, "collapsed_pairs": ties})
    return out


def e2_skill_credit(rows) -> list[dict[str, Any]]:
    """Average gain in measured/true success from having a skill, over profiles."""
    by = {(r["grader"], r["profile"], r["variant"]): r for r in rows}
    prof = profiles()
    out = []
    for grader, skill in itertools.product(GRADERS, SKILLS):
        pairs = [(a, b) for a, sa in prof.items() for b, sb in prof.items() if sa == sb | {skill} and skill not in sb]
        for v in VARIANTS:
            out.append({
                "grader": grader, "skill": skill, "variant": v,
                "measured_credit": mean(by[(grader, a, v)]["measured"] - by[(grader, b, v)]["measured"] for a, b in pairs),
                "true_credit": mean(by[(grader, a, v)]["true"] - by[(grader, b, v)]["true"] for a, b in pairs),
            })
    return out


# -- E3 ---------------------------------------------------------------------


def e3_paired(table, episodes_per_task: int, seed: int, p_skilled: float = 0.95,
              p_unskilled: float = 0.3, n_boot: int = 2000) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for name, skills in profiles().items():
        # One draw per (task, episode), shared by every variant: the pairing.
        draws = {
            (task.id, k): rng.random() < (1.0 if task.skill is None else
                                          p_skilled if task.skill in skills else p_unskilled)
            for task in TASKS for k in range(episodes_per_task)
        }
        for grader, v in itertools.product(GRADERS, VARIANTS):
            for scope in ("rule_tasks", "controls"):
                tasks = [t for t in TASKS if (t.rule is None) == (scope == "controls")]
                if v != "reference" and scope == "rule_tasks":
                    tasks = [t for t in tasks if t.rule == DEFECT_OF_VARIANT[v]]
                diffs, meas, true = [], [], []
                for t in tasks:
                    for k in range(episodes_per_task):
                        valid = draws[(t.id, k)]
                        g, o = _outcome(table, grader, t, valid, v)
                        g_ref, _ = _outcome(table, grader, t, valid, "reference")
                        diffs.append(int(g) - int(g_ref))
                        meas.append(int(g))
                        true.append(int(o))
                boot = sorted(
                    mean(rng.choice(diffs) for _ in diffs) for _ in range(n_boot)
                )
                rows.append({
                    "grader": grader, "profile": name, "variant": v, "scope": scope, "n": len(diffs),
                    "measured_success": round(mean(meas), 3), "true_success": round(mean(true), 3),
                    "shift_vs_reference": round(mean(diffs), 3),
                    "ci_low": round(boot[int(0.025 * n_boot)], 3),
                    "ci_high": round(boot[int(0.975 * n_boot) - 1], 3),
                })
    return rows


# -- driver -----------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    episodes = replay_all()
    table = outcome_table(episodes)
    write_csv(args.out / "e1_replay.csv", [to_row(e) for e in episodes])
    e1 = e1_summary(episodes)
    write_csv(args.out / "e1_summary.csv", e1)
    e2 = e2_profiles(table)
    write_csv(args.out / "e2_profiles.csv", e2)
    e2r = e2_rankings(e2)
    write_csv(args.out / "e2_rankings.csv", e2r)
    e2c = e2_skill_credit(e2)
    write_csv(args.out / "e2_skill_credit.csv", e2c)
    e3 = e3_paired(table, args.episodes, args.seed)
    write_csv(args.out / "e3_paired.csv", e3)
    (args.out / "summary.json").write_text(json.dumps(
        {"e1": e1, "e2_rankings": e2r, "e2_skill_credit": e2c, "config": vars(args) | {"out": str(args.out)}}, indent=2))

    names = list(profiles())
    print("E1 counterfactual replay  (FP/FF/silent = false pass / false failure / passed despite env violation)")
    for r in e1:
        print(f"  {r['variant']:26s} {r['scope']:10s} disc={r['episodes_with_discrepancy']:2d}/{r['episodes']:2d} "
              f"envviol={r['env_violations']}  outcome FP={r['outcome_false_pass']} FF={r['outcome_false_failure']} "
              f"silent={r['outcome_silent_env_violation']}  state_match FP={r['state_match_false_pass']} "
              f"FF={r['state_match_false_failure']} silent={r['state_match_silent_env_violation']}")
    for grader in GRADERS:
        print(f"\nE2 [{grader}] profiles: measured/true successes out of {len(TASKS)} tasks")
        print("  " + "profile".ljust(24) + "".join(v[:14].ljust(16) for v in VARIANTS))
        for n in names:
            cells = [next(r for r in e2 if r["grader"] == grader and r["profile"] == n and r["variant"] == v)
                     for v in VARIANTS]
            print("  " + n.ljust(24) + "".join(f"{c['measured']}/{c['true']}".ljust(16) for c in cells))
        for r in e2r:
            if r["grader"] == grader:
                print(f"  {r['variant']:26s} tau_b={r['kendall_tau_b']} correct={r['correctly_ordered']}"
                      f"/{r['n_pairs_true_distinct']} inverted={len(r['inverted_pairs'])} "
                      f"collapsed={len(r['collapsed_pairs'])}")
        print("  skill credit (measured / true):")
        for skill in SKILLS:
            print(f"    {skill:20s}" + "".join(
                f"{c['measured_credit']:+.0f}/{c['true_credit']:+.0f}".ljust(16)
                for c in e2c if c["skill"] == skill and c["grader"] == grader))
    print(f"\nE3 paired shift in measured success vs reference, {args.episodes} episodes/task "
          "(rule tasks of that defect | controls)")
    for grader in GRADERS:
        for n in ("freeze+approval+retry", "none"):
            for v in list(VARIANTS)[1:]:
                a, c = (next(r for r in e3 if r["grader"] == grader and r["profile"] == n
                             and r["variant"] == v and r["scope"] == sc) for sc in ("rule_tasks", "controls"))
                print(f"  {grader:12s} {n:22s} {v:26s} {a['shift_vs_reference']:+.3f} "
                      f"[{a['ci_low']:+.3f},{a['ci_high']:+.3f}] | {c['shift_vs_reference']:+.3f} "
                      f"[{c['ci_low']:+.3f},{c['ci_high']:+.3f}]")
    print(f"\nwrote {args.out}/")


if __name__ == "__main__":
    main()
