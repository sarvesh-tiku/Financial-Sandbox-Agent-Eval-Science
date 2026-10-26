"""Seeded-defect audit of AppWorld's assertion-based evaluation.

Runs each task's ground-truth solution in AppWorld, unmodified and under seeded
"omitted transition" defects at the API layer: a write request of one HTTP method
returns a success response without being applied. AppWorld evaluates with
hand-written assertions over start and end database states rather than by gold
replay, so a defect can surface as a failed evaluation of the gold solution.

    APPWORLD_ROOT=/path python audits/appworld_audit.py --splits train dev --out results/appworld

Only aggregate counts and task identifiers are written: AppWorld's task data may
be redistributed only in encrypted form.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import deque
from pathlib import Path


def _key(method: str, url: str, data) -> str:
    # Access tokens differ between runs; everything else in a write request is deterministic.
    data = {k: v for k, v in (data or {}).items() if k != "access_token"} if isinstance(data, dict) else data
    return json.dumps([method, url.split("?")[0], data], sort_keys=True, default=str)


def run(task_id: str, mutant: str | None, recorded: dict | None = None) -> tuple[dict, dict]:
    """Run the ground-truth solution once.

    Reference run: every write response is recorded. Mutant run: writes of one method are
    dropped, and the caller receives the response the real API gave for that request in the
    reference run (status and body), so only the state change is missing. A request with no
    recorded counterpart gets a bare 200 "Success." and is counted as a fallback.
    """
    from appworld import AppWorld
    from appworld.requester import CustomResponse, Requester

    class Replayed(CustomResponse):
        def json(self):
            return json.loads(self.text)

    originals = {m: getattr(Requester, f"_{m}") for m in ("post", "patch", "delete")}
    recording: dict[str, list] = {}
    queues = {k: deque(v) for k, v in (recorded or {}).items()}
    fallbacks = 0

    def recorder(m):
        def call(self, url, data=None, *args, **kwargs):
            response = originals[m](self, url, data, *args, **kwargs)
            recording.setdefault(_key(m, url, data), []).append([response.status_code, response.text])
            return response
        return call

    if mutant is None:
        for m in originals:
            setattr(Requester, f"_{m}", recorder(m))
    else:
        def noop(self, url, data=None, *args, **kwargs):
            nonlocal fallbacks
            if mutant == "post" and ("auth/token" in url or "supervisor" in url):
                return originals["post"](self, url, data, *args, **kwargs)
            self.request_tracker.add_request(method=mutant, url=url, data=data)
            queue = queues.get(_key(mutant, url, data))
            if queue:
                status, text = queue.popleft()
                return Replayed(status_code=status, text=text)
            fallbacks += 1
            return CustomResponse(status_code=200, text="Success.")
        setattr(Requester, f"_{mutant}", noop)
    root = Path(os.environ["APPWORLD_ROOT"])
    code = (root / "data" / "tasks" / task_id / "ground_truth" / "compiled_solution.py").read_text()
    code += "\nsolution(apis, requester)\n"
    try:
        # Same settings as AppWorld's own ground-truth verification (appworld/verify.py).
        with AppWorld(task_id=task_id, experiment_name=f"audit_{mutant or 'reference'}",
                      raise_on_failure=False, ground_truth_mode="full") as world:
            output = str(world.execute(code))
            crashed = "Execution successful" not in output
            result = world.evaluate().to_dict()
        rec = {"task": task_id, "mutant": mutant or "reference", "success": result["success"],
               "crashed": crashed, "num_tests": result["num_tests"], "num_failures": len(result["failures"])}
        if crashed:
            # Exception type only; the message can contain task data.
            rec["crash_type"] = output.strip().splitlines()[-1].split(":")[0][:80]
        if mutant:
            rec["response_fallbacks"] = fallbacks
        return rec, recording
    except Exception:  # noqa: BLE001 - record and continue
        return {"task": task_id, "mutant": mutant or "reference", "error": traceback.format_exc(limit=1)[-300:]}, recording
    finally:
        for m, f in originals.items():
            setattr(Requester, f"_{m}", f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--out", type=Path, default=Path("results/appworld"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ["APPWORLD_ROOT"])

    jobs = []
    tasks = [t for split in args.splits for t in (root / "data" / "datasets" / f"{split}.txt").read_text().split()]
    for tid in tasks[args.shard::args.num_shards]:  # shard by task: mutants need their task's reference run
        calls = json.loads((root / "data" / "tasks" / tid / "ground_truth" / "api_calls.json").read_text())
        methods = {c["method"] for c in calls
                   if not (c["method"] == "post" and ("auth/token" in c["url"] or "supervisor" in c["url"]))}
        jobs.append((tid, None))
        jobs += [(tid, m) for m in ("post", "patch", "delete") if m in methods]
    out = args.out / f"runs_{args.shard}.jsonl"
    done = set()
    if out.exists():
        done = {(r["task"], r["mutant"]) for r in map(json.loads, out.read_text().splitlines())}
    reference: dict[str, tuple[dict, dict]] = {}
    for tid, mutant in jobs:
        if (tid, mutant or "reference") in done:
            continue
        if tid not in reference:
            reference[tid] = run(tid, None)  # also records the write responses replayed below
        rec = reference[tid][0] if mutant is None else run(tid, mutant, reference[tid][1])[0]
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    sys.exit(main())
