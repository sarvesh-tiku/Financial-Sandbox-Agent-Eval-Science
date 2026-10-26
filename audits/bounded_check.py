"""Bounded exhaustive check of the environment against the specification oracle.

Enumerates every action sequence up to a given depth over a small action alphabet,
runs it in each environment variant, and replays it through the oracle. For the
reference environment, expected and implemented state must agree after every step.
For each defective variant, the first divergence must be caused by an action that
the omitted rule governs, and every such action type must diverge at least once.

    PYTHONPATH=src python audits/bounded_check.py --depth 4 --out results/bounded_check.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

from finsandbox.env import BankEnv
from finsandbox.replay import VARIANTS
from finsandbox.spec import ReferenceModel, diff_states

ACCOUNTS = [{"id": "A", "owner": "a", "balance": 1_000}, {"id": "B", "owner": "b", "balance": 1_000},
            {"id": "V", "owner": "v", "balance": 0}]
ALPHABET = [
    ("freeze_account", {"account_id": "A"}),
    ("create_transfer", {"src": "A", "dst": "V", "amount": 100, "payee": "P1"}),
    ("create_transfer", {"src": "B", "dst": "V", "amount": 100, "payee": "P1"}),
    ("edit_transfer", {"transfer_id": 1, "payee": "P2"}),
    ("edit_transfer", {"transfer_id": 2, "amount": 150}),
    ("approve_transfer", {"transfer_id": 1}),
    ("approve_transfer", {"transfer_id": 2}),
    ("cancel_transfer", {"transfer_id": 1}),
    ("execute_transfer", {"transfer_id": 1}),
    ("execute_transfer", {"transfer_id": 2}),
    ("submit_payment", {"src": "A", "dst": "V", "amount": 50, "invoice_ref": "INV-1", "idempotency_key": "k1"}),
    ("submit_payment", {"src": "A", "dst": "V", "amount": 50, "invoice_ref": "INV-1", "idempotency_key": "k2"}),
]
# Which action types each omitted rule governs.
GOVERNS = {"no_freeze_propagation": {"freeze_account"},
           "no_approval_invalidation": {"edit_transfer"},
           "no_idempotency": {"submit_payment"}}


def run(variant: str, seq) -> tuple[int | None, int]:
    """Return (index of first divergent step or None, number of steps compared)."""
    env = BankEnv(ACCOUNTS, VARIANTS[variant])
    model = ReferenceModel(env.initial_snapshot)
    first = None
    for i, (tool, args) in enumerate(seq):
        verdict = model.apply(tool, args)
        env.call(tool, args)
        ok = env.log[-1]["ok"]
        same_state = not diff_states(model.snapshot(), env.log[-1]["state"])
        # Acceptance can differ only where the oracle also flags a spec violation the
        # environment is meant to block; state must match whenever no violation is involved.
        if first is None and (not same_state or (verdict.accepted != ok and not verdict.violation)):
            first = i
            break
    env.db.close()
    return first, i + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/bounded_check.json"))
    args = ap.parse_args()
    seqs = [s for d in range(1, args.depth + 1) for s in itertools.product(ALPHABET, repeat=d)]
    summary = {"depth": args.depth, "alphabet_size": len(ALPHABET), "sequences": len(seqs), "variants": {}}
    for variant in VARIANTS:
        steps = diverged = 0
        culprits = Counter()
        for seq in seqs:
            first, n = run(variant, seq)
            steps += n
            if first is not None:
                diverged += 1
                culprits[seq[first][0]] += 1
        entry = {"steps_compared": steps, "sequences_diverging": diverged,
                 "first_divergence_by_action": dict(culprits)}
        if variant == "reference":
            entry["pass"] = diverged == 0
        else:
            entry["pass"] = diverged > 0 and set(culprits) <= GOVERNS[variant]
        summary["variants"][variant] = entry
        print(variant, json.dumps(entry), flush=True)
    args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
