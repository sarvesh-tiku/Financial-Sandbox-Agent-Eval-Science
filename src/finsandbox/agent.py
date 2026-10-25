"""Live rollouts against the Claude API (manual tool-use loop).

Server-side refusal fallbacks are deliberately NOT enabled: a fallback would
silently swap the model under evaluation mid-episode. Refusals are recorded
as an episode outcome instead.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from .rollout import SYSTEM, TOOL_SPECS, Session
from .tasks import Task

TOOLS = [{"name": n, "description": d, "input_schema": s, "strict": True} for n, d, s in TOOL_SPECS]


def run_rollout(client: anthropic.Anthropic, task: Task, variant: str, model: str,
                effort: str = "medium", max_tool_calls: int = 15, **meta: Any) -> dict[str, Any]:
    session = Session(task, variant)
    messages: list[dict[str, Any]] = [{"role": "user", "content": task.instruction}]
    tool_calls, stop, final_text = 0, "budget_exhausted", ""

    while tool_calls < max_tool_calls:
        response = client.messages.create(
            model=model, max_tokens=16000, system=SYSTEM, tools=TOOLS,
            messages=messages, output_config={"effort": effort},
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            stop = response.stop_reason
            final_text = "".join(b.text for b in response.content if b.type == "text")
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls += 1
                result = session.call(block.name, dict(block.input))
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(result), "is_error": "error" in result})
        messages.append({"role": "user", "content": results})

    return session.finish(model=model, effort=effort, stop_reason=stop, tool_calls=tool_calls,
                          final_text=final_text, **meta)
