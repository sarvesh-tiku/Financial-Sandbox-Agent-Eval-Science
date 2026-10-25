"""Live rollouts against the OpenAI Chat Completions API, with a hard spend cap.

    python -m finsandbox.openai_agent --models gpt-4o-mini gpt-5-mini --reps 5 \
        --out results/llm/openai.jsonl --budget-usd 5

Reads the key from OPENAI_API_KEY. Episodes already present in --out are
skipped, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from .replay import VARIANTS
from .rollout import SYSTEM, TOOL_SPECS, Session, append_jsonl
from .tasks import TASKS, Task

TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": s, "strict": True}}
         for n, d, s in TOOL_SPECS]

# USD per 1M tokens (input, output). Conservative estimates used only for the cap.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60), "gpt-5-mini": (0.25, 2.00),
          "gpt-4.1": (2.00, 8.00), "gpt-5": (1.25, 10.00),
          "gpt-5-nano": (0.05, 0.40), "gpt-4.1-nano": (0.10, 0.40)}


class BudgetExceeded(RuntimeError):
    pass


class Meter:
    def __init__(self, budget: float):
        self.budget, self.spent = budget, 0.0

    def add(self, model: str, usage) -> float:
        pin, pout = PRICES[model]
        cost = (usage.prompt_tokens * pin + usage.completion_tokens * pout) / 1e6
        self.spent += cost
        if self.spent > self.budget:
            raise BudgetExceeded(f"spent ${self.spent:.2f} > budget ${self.budget:.2f}")
        return cost


def run_rollout(client: OpenAI, meter: Meter, task: Task, variant: str, model: str,
                max_tool_calls: int = 15, **meta: Any) -> dict[str, Any]:
    session = Session(task, variant)
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": task.instruction}]
    extra = {"reasoning_effort": "low"} if model.startswith("gpt-5") else {}
    tool_calls, stop, final_text, cost = 0, "budget_exhausted", "", 0.0

    while tool_calls < max_tool_calls:
        resp = client.chat.completions.create(model=model, messages=messages, tools=TOOLS, **extra)
        cost += meter.add(model, resp.usage)
        choice = resp.choices[0]
        msg = choice.message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            stop, final_text = choice.finish_reason, msg.content or ""
            break
        for tc in msg.tool_calls:
            tool_calls += 1
            try:
                args = json.loads(tc.function.arguments)
                result = session.call(tc.function.name, args)
            except json.JSONDecodeError:
                result = {"error": "invalid JSON arguments"}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    return session.finish(model=model, stop_reason=stop, tool_calls=tool_calls,
                          final_text=final_text, cost_usd=round(cost, 5), **meta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-5-mini"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/llm/openai.jsonl"))
    ap.add_argument("--budget-usd", type=float, default=5.0)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = set()  # episodes already recorded in any OpenAI results file in the same directory
    for path in args.out.parent.glob("openai_*.jsonl"):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["task"], r["variant"], r["rep"]))
    client, meter = OpenAI(), Meter(args.budget_usd)
    tasks = [t for t in TASKS if args.tasks is None or t.id in args.tasks]
    try:
        for rep in range(args.reps):
            for model in args.models:
                for task in tasks:
                    for variant in VARIANTS:
                        if (model, task.id, variant, rep) in done:
                            continue
                        rec = run_rollout(client, meter, task, variant, model, rep=rep, provider="openai")
                        append_jsonl(args.out, rec)
                        print(f"rep={rep} {model:12s} {task.id:28s} {variant:26s} oracle={rec['oracle_pass']!s:5s} "
                              f"outcome={rec['grader_pass']!s:5s} match={rec['match_pass']!s:5s} "
                              f"{rec['label']:24s} ${meter.spent:.3f}", flush=True)
    except BudgetExceeded as e:
        print(f"STOPPED: {e}")
    print(f"total spent ~${meter.spent:.3f}")


if __name__ == "__main__":
    main()
