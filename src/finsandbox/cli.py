"""Command-line agent interface, for agents that act through a shell.

Episodes are registered by the experimenter with opaque ids; the variant is
stored in the state directory and never shown to the agent.

    python -m finsandbox.cli register --state DIR --model M --rep R   # prints episode ids
    bank start ID                # task, policy, and tool list
    bank call ID TOOL 'JSON'     # one tool call; prints the result as JSON
    bank done ID 'summary'       # end the episode
    python -m finsandbox.cli collect --state DIR --out FILE.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from .replay import VARIANTS
from .rollout import SYSTEM, TOOL_SPECS, TASKS_BY_ID, Session, append_jsonl

MAX_TOOL_CALLS = 15


def _state_dir(args) -> Path:
    return Path(args.state or os.environ["BANK_STATE"])


def _load(args) -> tuple[Path, dict]:
    path = _state_dir(args) / f"{args.id}.json"
    if not path.exists():
        sys.exit(f"unknown episode {args.id}")
    return path, json.loads(path.read_text())


def cmd_register(args) -> None:
    state = _state_dir(args)
    state.mkdir(parents=True, exist_ok=True)
    variants = args.variants or list(VARIANTS)
    for variant in variants:
        ids = []
        for task_id in TASKS_BY_ID:
            eid = secrets.token_hex(4)
            (state / f"{eid}.json").write_text(json.dumps({
                "task": task_id, "variant": variant, "model": args.model, "rep": args.rep,
                "actions": [], "done": False, "summary": "",
            }))
            ids.append(eid)
        print(json.dumps({"variant": variant, "episodes": ids}))


def cmd_start(args) -> None:
    _, ep = _load(args)
    tools = "\n".join(f"- {n}: {d}\n  args: {json.dumps(s['properties'])}" for n, d, s in TOOL_SPECS)
    print(f"{SYSTEM}\n\nTASK:\n{TASKS_BY_ID[ep['task']].instruction}\n\nTOOLS:\n{tools}\n\n"
          f"Call a tool with: bank call {args.id} <tool> '<json args>'\n"
          f"Finish with: bank done {args.id} '<one-paragraph summary>'")


def cmd_call(args) -> None:
    path, ep = _load(args)
    if ep["done"]:
        sys.exit("episode already finished")
    if len(ep["actions"]) >= MAX_TOOL_CALLS:
        sys.exit("tool-call budget exhausted; finish with bank done")
    try:
        tool_args = json.loads(args.args or "{}")
    except json.JSONDecodeError:
        print(json.dumps({"error": "args must be a JSON object"}))
        return
    session = Session.rebuild(ep["task"], ep["variant"], ep["actions"])
    result = session.call(args.tool, tool_args)
    ep["actions"] = [[t, a] for t, a in session.actions]
    path.write_text(json.dumps(ep))
    session.env.db.close()
    print(json.dumps(result))


def cmd_done(args) -> None:
    path, ep = _load(args)
    ep["done"], ep["summary"] = True, args.summary
    path.write_text(json.dumps(ep))
    print("episode finished")


def cmd_collect(args) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(_state_dir(args).glob("*.json")):
        ep = json.loads(path.read_text())
        session = Session.rebuild(ep["task"], ep["variant"], ep["actions"])
        append_jsonl(out, session.finish(model=ep["model"], rep=ep["rep"], provider="claude-subagent",
                                         episode_id=path.stem, completed=ep["done"],
                                         tool_calls=len(ep["actions"]), final_text=ep["summary"]))
        n += 1
    print(f"wrote {n} episodes to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="bank")
    ap.add_argument("--state", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("register"); p.add_argument("--model", required=True)
    p.add_argument("--rep", type=int, required=True); p.add_argument("--variants", nargs="*")
    p.set_defaults(fn=cmd_register)
    p = sub.add_parser("start"); p.add_argument("id"); p.set_defaults(fn=cmd_start)
    p = sub.add_parser("call"); p.add_argument("id"); p.add_argument("tool"); p.add_argument("args", nargs="?")
    p.set_defaults(fn=cmd_call)
    p = sub.add_parser("done"); p.add_argument("id"); p.add_argument("summary"); p.set_defaults(fn=cmd_done)
    p = sub.add_parser("collect"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_collect)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
