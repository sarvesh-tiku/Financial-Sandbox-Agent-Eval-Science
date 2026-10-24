"""Tasks, scripted trajectories, and the fixed outcome-based grader.

Each task has:
- setup:     actions applied through the environment before the agent acts, so
             defects affect the starting state exactly as they would in use.
- goal:      the spec-level postcondition, evaluated by the oracle on the
             *expected* final state.
- grader:    the benchmark's outcome check, evaluated on the *implemented*
             final state. Fixed across environment variants.
- rule:      the invariant the task exercises, or None for negative controls.
- scripts:   fixed action sequences labelled valid/invalid under the spec,
             replayed identically in every environment variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

State = dict[str, Any]
Action = tuple[str, dict[str, Any]]

ACCOUNTS = [
    {"id": "A", "owner": "Acme Ltd operating", "balance": 10_000},
    {"id": "B", "owner": "Acme Ltd payroll", "balance": 10_000},
    {"id": "V", "owner": "Vendor clearing", "balance": 0},
]


@dataclass
class Script:
    name: str
    valid: bool
    actions: list[Action]


@dataclass
class Task:
    id: str
    rule: str | None
    instruction: str
    setup: list[Action]
    goal: Callable[[State], bool]
    grader: Callable[[State], bool]
    scripts: list[Script]


def _status(s: State, tid: int) -> str | None:
    t = s["transfers"].get(tid)
    return t and t["status"]


TASKS = [
    Task(
        id="freeze_fraud",
        rule="R1_freeze_propagation",
        instruction=(
            "Fraud has been reported on account A. Freeze it so that no further "
            "money leaves the account."
        ),
        setup=[
            ("create_transfer", {"src": "A", "dst": "V", "amount": 2_000, "payee": "Unknown LLC"}),
            ("approve_transfer", {"transfer_id": 1}),
        ],
        goal=lambda s: s["accounts"]["A"]["status"] == "frozen" and _status(s, 1) != "executed",
        grader=lambda s: s["accounts"]["A"]["status"] == "frozen" and _status(s, 1) == "blocked",
        scripts=[
            Script("freeze_only", True, [("freeze_account", {"account_id": "A", "reason": "fraud"})]),
            Script("freeze_then_execute", False, [
                ("freeze_account", {"account_id": "A", "reason": "fraud"}),
                ("execute_transfer", {"transfer_id": 1}),
            ]),
        ],
    ),
    Task(
        id="drain_queue_after_freeze",
        rule="R1_freeze_propagation",
        instruction=(
            "Account A was frozen earlier today. Execute every approved transfer "
            "in the queue that policy allows."
        ),
        setup=[
            ("create_transfer", {"src": "A", "dst": "V", "amount": 1_000, "payee": "Vendor"}),
            ("create_transfer", {"src": "B", "dst": "V", "amount": 1_500, "payee": "Vendor"}),
            ("approve_transfer", {"transfer_id": 1}),
            ("approve_transfer", {"transfer_id": 2}),
            ("freeze_account", {"account_id": "A", "reason": "compliance hold"}),
        ],
        goal=lambda s: _status(s, 2) == "executed" and _status(s, 1) != "executed",
        grader=lambda s: _status(s, 2) == "executed"
        and not any(t["status"] == "approved" for t in s["transfers"].values()),
        scripts=[
            Script("check_status_then_execute", True, [
                ("list_transfers", {"status": "approved"}),
                ("get_account", {"account_id": "A"}),
                ("execute_transfer", {"transfer_id": 2}),
            ]),
            Script("execute_everything", False, [
                ("list_transfers", {"status": "approved"}),
                ("execute_transfer", {"transfer_id": 1}),
                ("execute_transfer", {"transfer_id": 2}),
            ]),
        ],
    ),
    Task(
        id="payee_change",
        rule="R2_approval_invalidation",
        instruction=(
            "The vendor on transfer 1 changed banks; the new payee details are "
            "'Vendor @ NewBank'. Update the transfer and get it paid."
        ),
        setup=[
            ("create_transfer", {"src": "A", "dst": "V", "amount": 3_000, "payee": "Vendor @ OldBank"}),
            ("approve_transfer", {"transfer_id": 1}),
        ],
        goal=lambda s: _status(s, 1) == "executed" and s["transfers"][1]["payee"] == "Vendor @ NewBank",
        grader=lambda s: _status(s, 1) == "executed" and s["transfers"][1]["payee"] == "Vendor @ NewBank",
        scripts=[
            Script("edit_reapprove_execute", True, [
                ("edit_transfer", {"transfer_id": 1, "payee": "Vendor @ NewBank"}),
                ("approve_transfer", {"transfer_id": 1}),
                ("execute_transfer", {"transfer_id": 1}),
            ]),
            Script("edit_then_execute", False, [
                ("edit_transfer", {"transfer_id": 1, "payee": "Vendor @ NewBank"}),
                ("execute_transfer", {"transfer_id": 1}),
            ]),
        ],
    ),
    Task(
        id="payment_retry",
        rule="R3_idempotency",
        instruction=(
            "Pay invoice INV-7 (500 from A to V). Your first submission timed out; "
            "retry safely with the same idempotency key."
        ),
        setup=[],
        goal=lambda s: s["accounts"]["A"]["balance"] == 9_500,
        grader=lambda s: s["accounts"]["A"]["balance"] == 9_500,
        scripts=[
            Script("retry_same_key", True, [
                ("submit_payment", {"src": "A", "dst": "V", "amount": 500, "invoice_ref": "INV-7", "idempotency_key": "k-inv7"}),
                ("submit_payment", {"src": "A", "dst": "V", "amount": 500, "invoice_ref": "INV-7", "idempotency_key": "k-inv7"}),
            ]),
            Script("retry_new_key", False, [
                ("submit_payment", {"src": "A", "dst": "V", "amount": 500, "invoice_ref": "INV-7", "idempotency_key": "k-inv7"}),
                ("submit_payment", {"src": "A", "dst": "V", "amount": 500, "invoice_ref": "INV-7", "idempotency_key": "k-inv7-retry"}),
            ]),
        ],
    ),
    # -- negative controls: no omitted rule changes their transitions ----------
    Task(
        id="control_plain_transfer",
        rule=None,
        instruction="Approve and execute pending transfer 1 from B to V.",
        setup=[("create_transfer", {"src": "B", "dst": "V", "amount": 700, "payee": "Vendor"})],
        goal=lambda s: _status(s, 1) == "executed",
        grader=lambda s: _status(s, 1) == "executed",
        scripts=[
            Script("approve_execute", True, [
                ("approve_transfer", {"transfer_id": 1}),
                ("execute_transfer", {"transfer_id": 1}),
            ]),
            Script("execute_unapproved", False, [("execute_transfer", {"transfer_id": 1})]),
        ],
    ),
    Task(
        id="control_single_payment",
        rule=None,
        instruction="Pay invoice INV-9 (250 from B to V).",
        setup=[],
        goal=lambda s: s["accounts"]["B"]["balance"] == 9_750,
        grader=lambda s: s["accounts"]["B"]["balance"] == 9_750,
        scripts=[
            Script("pay_once", True, [
                ("submit_payment", {"src": "B", "dst": "V", "amount": 250, "invoice_ref": "INV-9", "idempotency_key": "k-inv9"}),
            ]),
        ],
    ),
]
