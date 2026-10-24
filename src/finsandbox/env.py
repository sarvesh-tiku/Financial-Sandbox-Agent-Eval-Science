"""Relational banking sandbox (SQLite) with switchable transition defects.

Each workflow invariant is enforced in exactly one place, so turning a defect
on removes that invariant entirely rather than leaving a backup check behind:

- R1 freeze propagation:   freeze_account blocks the account's open outflows.
                           execute_transfer only gates on transfer status.
- R2 approval invalidation: edit_transfer on an approved transfer revokes the
                           approval and returns it to pending.
- R3 idempotent payments:  submit_payment with a seen idempotency key replays
                           the original result instead of debiting again.
"""

from __future__ import annotations

import sqlite3
from typing import Any

DEFECTS = ("no_freeze_propagation", "no_approval_invalidation", "no_idempotency")

SCHEMA = """
CREATE TABLE accounts (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL,
    balance INTEGER NOT NULL, status TEXT NOT NULL
);
CREATE TABLE transfers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT NOT NULL REFERENCES accounts(id),
    dst TEXT NOT NULL REFERENCES accounts(id),
    amount INTEGER NOT NULL, payee TEXT NOT NULL, status TEXT NOT NULL
);
CREATE TABLE approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transfer_id INTEGER NOT NULL REFERENCES transfers(id),
    details TEXT NOT NULL, active INTEGER NOT NULL
);
CREATE TABLE payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL, invoice_ref TEXT NOT NULL,
    src TEXT NOT NULL, dst TEXT NOT NULL, amount INTEGER NOT NULL
);
CREATE TABLE ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL, delta INTEGER NOT NULL, ref TEXT NOT NULL
);
"""


class ToolError(Exception):
    pass


def details_of(payee: str, amount: int) -> str:
    return f"{payee}|{amount}"


class BankEnv:
    def __init__(self, accounts: list[dict[str, Any]], defects: tuple[str, ...] = ()):
        unknown = set(defects) - set(DEFECTS)
        if unknown:
            raise ValueError(f"unknown defects: {sorted(unknown)}")
        self.defects = frozenset(defects)
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        for a in accounts:
            self.db.execute(
                "INSERT INTO accounts VALUES (?, ?, ?, ?)",
                (a["id"], a["owner"], a["balance"], a.get("status", "active")),
            )
        self.initial_snapshot = self.snapshot()
        self.log: list[dict[str, Any]] = []

    # -- harness ---------------------------------------------------------

    def call(self, tool: str, args: dict[str, Any], phase: str = "agent") -> dict[str, Any]:
        fn = getattr(self, f"tool_{tool}", None)
        if fn is None:
            result, ok = {"error": f"unknown tool {tool}"}, False
        else:
            try:
                with self.db:
                    result, ok = fn(**args), True
            except ToolError as e:
                result, ok = {"error": str(e)}, False
        self.log.append({
            "phase": phase, "tool": tool, "args": dict(args), "ok": ok,
            "result": result, "state": self.snapshot(),
        })
        return result

    def snapshot(self) -> dict[str, Any]:
        q = self.db.execute
        accounts = {
            r["id"]: {"balance": r["balance"], "status": r["status"]}
            for r in q("SELECT * FROM accounts")
        }
        transfers = {}
        for r in q("SELECT * FROM transfers"):
            appr = q(
                "SELECT details FROM approvals WHERE transfer_id = ? AND active = 1 "
                "ORDER BY id DESC LIMIT 1", (r["id"],),
            ).fetchone()
            transfers[r["id"]] = {
                "src": r["src"], "dst": r["dst"], "amount": r["amount"],
                "payee": r["payee"], "status": r["status"],
                "approved_for": appr["details"] if appr else None,
            }
        payments = sorted(
            [r["invoice_ref"], r["idempotency_key"]] for r in q("SELECT * FROM payments")
        )
        ledger = sorted(
            [r["account"], r["delta"], r["ref"]] for r in q("SELECT * FROM ledger")
        )
        return {"accounts": accounts, "transfers": transfers,
                "payments": payments, "ledger": ledger}

    # -- helpers ---------------------------------------------------------

    def _account(self, account_id: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
        if row is None:
            raise ToolError(f"account {account_id} not found")
        return row

    def _transfer(self, transfer_id: int) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM transfers WHERE id = ?", (transfer_id,)).fetchone()
        if row is None:
            raise ToolError(f"transfer {transfer_id} not found")
        return row

    def _move(self, src: str, dst: str, amount: int, ref: str) -> None:
        if self._account(src)["balance"] < amount:
            raise ToolError("insufficient funds")
        self._account(dst)
        self.db.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, src))
        self.db.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (amount, dst))
        self.db.execute("INSERT INTO ledger (account, delta, ref) VALUES (?, ?, ?)", (src, -amount, ref))
        self.db.execute("INSERT INTO ledger (account, delta, ref) VALUES (?, ?, ?)", (dst, amount, ref))

    # -- read tools ------------------------------------------------------

    def tool_get_account(self, account_id: str) -> dict[str, Any]:
        a = self._account(account_id)
        return {"id": a["id"], "owner": a["owner"], "balance": a["balance"], "status": a["status"]}

    def tool_list_transfers(self, status: str | None = None) -> dict[str, Any]:
        sql, params = "SELECT * FROM transfers", ()
        if status is not None:
            sql, params = sql + " WHERE status = ?", (status,)
        return {"transfers": [dict(r) for r in self.db.execute(sql + " ORDER BY id", params)]}

    # -- write tools -----------------------------------------------------

    def tool_freeze_account(self, account_id: str, reason: str = "") -> dict[str, Any]:
        self._account(account_id)
        self.db.execute("UPDATE accounts SET status = 'frozen' WHERE id = ?", (account_id,))
        blocked = 0
        if "no_freeze_propagation" not in self.defects:  # R1
            blocked = self.db.execute(
                "UPDATE transfers SET status = 'blocked' "
                "WHERE src = ? AND status IN ('pending', 'approved')", (account_id,),
            ).rowcount
        return {"account_id": account_id, "status": "frozen", "transfers_blocked": blocked}

    def tool_create_transfer(self, src: str, dst: str, amount: int, payee: str) -> dict[str, Any]:
        if amount <= 0:
            raise ToolError("amount must be positive")
        if self._account(src)["status"] != "active":
            raise ToolError(f"source account {src} is not active")
        self._account(dst)
        cur = self.db.execute(
            "INSERT INTO transfers (src, dst, amount, payee, status) VALUES (?, ?, ?, ?, 'pending')",
            (src, dst, amount, payee),
        )
        return {"transfer_id": cur.lastrowid, "status": "pending"}

    def tool_edit_transfer(self, transfer_id: int, payee: str | None = None,
                           amount: int | None = None) -> dict[str, Any]:
        t = self._transfer(transfer_id)
        if t["status"] not in ("pending", "approved"):
            raise ToolError(f"transfer {transfer_id} is {t['status']} and cannot be edited")
        payee = t["payee"] if payee is None else payee
        amount = t["amount"] if amount is None else amount
        self.db.execute("UPDATE transfers SET payee = ?, amount = ? WHERE id = ?",
                        (payee, amount, transfer_id))
        status = t["status"]
        if "no_approval_invalidation" not in self.defects:  # R2
            self.db.execute("UPDATE approvals SET active = 0 WHERE transfer_id = ?", (transfer_id,))
            status = "pending"
            self.db.execute("UPDATE transfers SET status = ? WHERE id = ?", (status, transfer_id))
        return {"transfer_id": transfer_id, "status": status}

    def tool_approve_transfer(self, transfer_id: int) -> dict[str, Any]:
        t = self._transfer(transfer_id)
        if t["status"] != "pending":
            raise ToolError(f"transfer {transfer_id} is {t['status']}, not pending")
        self.db.execute(
            "INSERT INTO approvals (transfer_id, details, active) VALUES (?, ?, 1)",
            (transfer_id, details_of(t["payee"], t["amount"])),
        )
        self.db.execute("UPDATE transfers SET status = 'approved' WHERE id = ?", (transfer_id,))
        return {"transfer_id": transfer_id, "status": "approved"}

    def tool_execute_transfer(self, transfer_id: int) -> dict[str, Any]:
        t = self._transfer(transfer_id)
        if t["status"] != "approved":
            raise ToolError(f"transfer {transfer_id} is {t['status']}, not approved")
        self._move(t["src"], t["dst"], t["amount"], f"transfer:{transfer_id}")
        self.db.execute("UPDATE transfers SET status = 'executed' WHERE id = ?", (transfer_id,))
        return {"transfer_id": transfer_id, "status": "executed"}

    def tool_submit_payment(self, src: str, dst: str, amount: int, invoice_ref: str,
                            idempotency_key: str) -> dict[str, Any]:
        if "no_idempotency" not in self.defects:  # R3
            prior = self.db.execute(
                "SELECT id FROM payments WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if prior is not None:
                return {"payment_id": prior["id"], "status": "completed", "replayed": True}
        if amount <= 0:
            raise ToolError("amount must be positive")
        if self._account(src)["status"] != "active":
            raise ToolError(f"source account {src} is not active")
        self._move(src, dst, amount, f"payment:{idempotency_key}")
        cur = self.db.execute(
            "INSERT INTO payments (idempotency_key, invoice_ref, src, dst, amount) VALUES (?, ?, ?, ?, ?)",
            (idempotency_key, invoice_ref, src, dst, amount),
        )
        return {"payment_id": cur.lastrowid, "status": "completed", "replayed": False}
