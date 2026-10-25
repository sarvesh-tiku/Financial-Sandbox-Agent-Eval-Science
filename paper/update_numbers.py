"""Recompute the live-run numbers quoted in main.tex from results/llm and rewrite
the macro block at the top of the paper.

    PYTHONPATH=src python paper/update_numbers.py
"""

import glob
import json
from pathlib import Path

from finsandbox.experiments import kendall_tau_b
from finsandbox.llm_analysis import AFFECTED, MODEL_ORDER, load

ROOT = Path(__file__).resolve().parents[1]
rows = load(ROOT / "results" / "llm")


def affected(variant):
    return [r for r in rows if r["variant"] == variant and r["rule"] != "control"
            and (variant == "reference" or r["rule"] == AFFECTED[variant])]


def count(eps, f):
    return sum(1 for e in eps if f(e))


ref, r1, r2, r3 = (affected(v) for v in ("reference", "no_freeze_propagation",
                                         "no_approval_invalidation", "no_idempotency"))
cost = sum(json.loads(line).get("cost_usd", 0)
           for f in glob.glob(str(ROOT / "results" / "llm" / "openai_*.jsonl")) for line in open(f))
gpt5_payee = [r for r in rows if r["model"] == "gpt-5" and r["variant"] == "no_approval_invalidation"
              and r["task"] == "payee_change"]


def tau(variant):
    true, meas = [], []
    for m in MODEL_ORDER:
        eps = [r for r in rows if r["model"] == m and r["variant"] == variant and r["rule"] != "control"]
        true.append(sum(r["oracle_pass"] for r in eps) / len(eps))
        meas.append(sum(r["grader_pass"] for r in eps) / len(eps))
    return kendall_tau_b(meas, true)


r1_ff = count(r1, lambda e: e["grader_error"] == "false_failure")
assert r1_ff == count(r1, lambda e: e["oracle_pass"]), "R1: every correct episode should be false-failed"
assert count(r3, lambda e: e["grader_error"] == "false_failure") == len(r3) == count(r3, lambda e: e["oracle_pass"])
values = {
    "NEpis": f"{len(rows):,}".replace(",", "{,}"),
    "Cost": f"{cost:.2f}",
    "NRefRule": len(ref),
    "NRefOracle": count(ref, lambda e: e["oracle_pass"]),
    "NRoneN": len(r1),
    "NRoneOracle": r1_ff,
    "NRoneGenuine": len(r1) - r1_ff,
    "NRtwoN": len(r2),
    "NRtwoOracle": count(r2, lambda e: e["oracle_pass"]),
    "NRtwoFP": count(r2, lambda e: e["grader_error"] == "false_pass"),
    "NRtwoFF": count(r2, lambda e: e["grader_error"] == "false_failure"),
    "NRtwoSilent": count(r2, lambda e: e["env_violation"] and e["grader_pass"]),
    "NRthreeN": len(r3),
    "NCtl": count(rows, lambda e: e["rule"] == "control"),
    "NRetry": count(rows, lambda e: e["task"] == "payment_retry"),
    "GfiveN": len(gpt5_payee),
    "GfiveRecreate": count(gpt5_payee, lambda e: any(t == "create_transfer" for t, _ in e["actions"])),
    "TauRtwo": f"{tau('no_approval_invalidation'):.2f}",
}
assert count(ref, lambda e: e["grader_error"] or e["match_error"]) == 0
assert count([r for r in rows if r["rule"] == "control"],
             lambda e: e["grader_error"] or e["match_error"] or e["n_discrepancies"]) == 0

lines = (ROOT / "paper" / "main.tex").read_text().splitlines(keepends=True)
for name, val in values.items():
    prefix = "\\newcommand{\\%s}{" % name
    hits = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    assert len(hits) == 1, name
    lines[hits[0]] = "%s%s}\n" % (prefix, val)
(ROOT / "paper" / "main.tex").write_text("".join(lines))
print(json.dumps(values, indent=1))
