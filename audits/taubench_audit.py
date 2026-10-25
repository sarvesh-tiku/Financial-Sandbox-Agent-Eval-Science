"""Audit tau-bench's retail and airline graders with the same protocol.

tau-bench scores an episode by hashing the final database and comparing it with
the hash obtained by replaying the task's gold actions in the same environment
(plus a check that required output strings were said). This script measures:

A. Attempt blindness: append one policy-prohibited action that the tool rejects
   (e.g. cancelling a delivered order) to each gold trajectory. In the correct
   environment the database is unchanged, so the episode still gets full
   database credit.
B. Seeded fidelity defects (mutation testing): remove one state transition from
   a tool. Gold replay inherits the defect, so the grader's verdict on the gold
   trajectory cannot change; a per-task differential replay against the
   unmodified environment shows which tasks the defect actually affects.
C. Tasks whose gold trajectory writes nothing, so an agent that takes no
   action receives database credit.

    python audits/taubench_audit.py --tau-bench /path/to/tau-bench --out results/taubench

Only the tool implementations, data, and task lists are imported; the LLM user
simulator is never loaded.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import types
from hashlib import sha256
from pathlib import Path


# -- loading tau-bench without its package __init__ (which imports the LLM stack)


def load_domain(root: Path, domain: str):
    for pkg in ("tau_bench", "tau_bench.envs", f"tau_bench.envs.{domain}"):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [str(root.joinpath(*pkg.split(".")))]
            sys.modules[pkg] = mod
    data = importlib.import_module(f"tau_bench.envs.{domain}.data")
    tools = importlib.import_module(f"tau_bench.envs.{domain}.tools")
    tasks = importlib.import_module(f"tau_bench.envs.{domain}.tasks_test")
    tool_map = {t.get_info()["function"]["name"]: t for t in tools.ALL_TOOLS}
    task_list = next(v for k, v in vars(tasks).items() if k.startswith("TASKS"))
    return data.load_data, tool_map, task_list


# Same hashing as tau_bench/envs/base.py (to_hashable + consistent_hash).
def to_hashable(item):
    if isinstance(item, dict):
        return tuple((k, to_hashable(v)) for k, v in sorted(item.items()))
    if isinstance(item, list):
        return tuple(to_hashable(e) for e in item)
    if isinstance(item, set):
        return tuple(sorted(to_hashable(e) for e in item))
    return item


def data_hash(data) -> str:
    return sha256(str(to_hashable(data)).encode()).hexdigest()


TERMINATE = {"transfer_to_human_agents"}


def run(load, tool_map, actions, overrides=None):
    """Execute tool actions the way Env.step does; returns (data, observations)."""
    data = load()
    tools = dict(tool_map) | (overrides or {})
    obs = []
    for name, kwargs in actions:
        if name not in tools or name in TERMINATE:
            continue
        try:
            obs.append(str(tools[name].invoke(data=data, **kwargs)))
        except Exception as e:  # noqa: BLE001 - mirrors Env.step
            obs.append(f"Error: {e}")
    return data, obs


def gold(task):
    return [(a.name, a.kwargs) for a in task.actions if a.name != "respond"]


def db_reward(load, tool_map, agent_actions, task, overrides=None) -> bool:
    agent_data, _ = run(load, tool_map, agent_actions, overrides)
    gold_data, _ = run(load, tool_map, gold(task), overrides)
    return data_hash(agent_data) == data_hash(gold_data)


# -- A. attempt blindness ------------------------------------------------------

RETAIL_GUARDS = [
    # (tool, status the policy requires, args builder)
    ("cancel_pending_order", "pending", lambda o: {"order_id": o["order_id"], "reason": "no longer needed"}),
    ("return_delivered_order_items", "delivered", lambda o: {
        "order_id": o["order_id"], "item_ids": [o["items"][0]["item_id"]],
        "payment_method_id": o["payment_history"][0]["payment_method_id"]}),
]


def retail_prohibited_attempt(data, user_id):
    """A call the policy forbids for one of the user's orders, or None."""
    for order_id in data["users"][user_id]["orders"]:
        order = data["orders"][order_id] | {"order_id": order_id}
        for tool, required, build in RETAIL_GUARDS:
            if order["status"] != required and order["items"] and order["payment_history"]:
                return tool, build(order), order["status"]
    return None


def audit_attempts(load, tool_map, tasks):
    rows = []
    base = load()
    for i, task in enumerate(tasks):
        attempt = retail_prohibited_attempt(base, task.user_id)
        if attempt is None:
            rows.append({"task": i, "attempt": None})
            continue
        tool, args, status = attempt
        traj = gold(task) + [(tool, args)]
        _, obs = run(load, tool_map, traj)
        rejected = obs[-1].startswith("Error")
        rows.append({
            "task": i, "attempt": f"{tool} on {status} order", "rejected_by_env": rejected,
            "db_reward_gold": db_reward(load, tool_map, gold(task), task),
            "db_reward_with_attempt": db_reward(load, tool_map, traj, task),
        })
    return rows


# -- B. seeded fidelity defects ------------------------------------------------


def make_mutants(tool_map):
    """Each mutant drops one state transition from a retail tool."""
    cancel = tool_map["cancel_pending_order"]
    modify = tool_map["modify_pending_order_items"]
    ret = tool_map["return_delivered_order_items"]

    class CancelNoRefund:
        @staticmethod
        def invoke(data, order_id, reason):
            balances = {pid: pm.get("balance") for u in data["users"].values()
                        for pid, pm in u["payment_methods"].items()}
            out = cancel.invoke(data=data, order_id=order_id, reason=reason)
            if not out.startswith("Error"):
                order = data["orders"][order_id]
                order["payment_history"] = [p for p in order["payment_history"]
                                            if p["transaction_type"] != "refund"]
                for u in data["users"].values():
                    for pid, pm in u["payment_methods"].items():
                        if "balance" in pm:
                            pm["balance"] = balances[pid]
            return out

    class ModifyStaysPending:
        @staticmethod
        def invoke(data, **kw):
            out = modify.invoke(data=data, **kw)
            if not out.startswith("Error"):
                data["orders"][kw["order_id"]]["status"] = "pending"
            return out

    class ReturnStaysDelivered:
        @staticmethod
        def invoke(data, **kw):
            out = ret.invoke(data=data, **kw)
            if not out.startswith("Error"):
                data["orders"][kw["order_id"]]["status"] = "delivered"
            return out

    return {
        "cancel_without_refund": {"cancel_pending_order": CancelNoRefund},
        "modify_keeps_order_pending": {"modify_pending_order_items": ModifyStaysPending},
        "return_keeps_order_delivered": {"return_delivered_order_items": ReturnStaysDelivered},
    }


def audit_mutants(load, tool_map, tasks):
    rows = []
    for name, overrides in make_mutants(tool_map).items():
        touched = differs = grader_changed = 0
        for task in tasks:
            g = gold(task)
            if not any(a in overrides for a, _ in g):
                continue
            touched += 1
            ref_data, _ = run(load, tool_map, g)
            mut_data, _ = run(load, tool_map, g, overrides)
            differs += data_hash(ref_data) != data_hash(mut_data)
            grader_changed += db_reward(load, tool_map, g, task, overrides) != db_reward(load, tool_map, g, task)
        rows.append({"mutant": name, "tasks_using_tool": touched,
                     "gold_final_state_differs_from_reference": differs,
                     "grader_verdict_on_gold_changed": grader_changed})
    return rows


# -- C. no-write tasks ---------------------------------------------------------


def audit_no_write(load, tool_map, tasks, domain):
    rows = []
    base_hash = data_hash(load())
    for i, task in enumerate(tasks):
        gold_data, _ = run(load, tool_map, gold(task))
        rows.append({"domain": domain, "task": i, "gold_writes": data_hash(gold_data) != base_hash,
                     "requires_outputs": bool(task.outputs)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tau-bench", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/taubench"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.tau_bench))

    load_r, tools_r, tasks_r = load_domain(args.tau_bench, "retail")
    attempts = audit_attempts(load_r, tools_r, tasks_r)
    mutants = audit_mutants(load_r, tools_r, tasks_r)
    no_write = audit_no_write(load_r, tools_r, tasks_r, "retail")
    load_a, tools_a, tasks_a = load_domain(args.tau_bench, "airline")
    no_write += audit_no_write(load_a, tools_a, tasks_a, "airline")

    with_attempt = [r for r in attempts if r["attempt"]]
    blind = [r for r in with_attempt if r["rejected_by_env"] and r["db_reward_with_attempt"]]
    summary = {
        "retail_test_tasks": len(tasks_r), "airline_test_tasks": len(tasks_a),
        "A_tasks_with_available_prohibited_attempt": len(with_attempt),
        "A_attempt_rejected_by_env": sum(r["rejected_by_env"] for r in with_attempt),
        "A_gold_db_reward": sum(r["db_reward_gold"] for r in with_attempt),
        "A_db_reward_kept_despite_attempt": len(blind),
        "B_mutants": mutants,
        "C": {d: {"tasks": sum(r["domain"] == d for r in no_write),
                  "gold_writes_nothing": sum(r["domain"] == d and not r["gold_writes"] for r in no_write),
                  "of_which_no_required_outputs": sum(r["domain"] == d and not r["gold_writes"]
                                                      and not r["requires_outputs"] for r in no_write)}
              for d in ("retail", "airline")},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "attempts.json").write_text(json.dumps(attempts, indent=2))
    (args.out / "no_write.json").write_text(json.dumps(no_write, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
