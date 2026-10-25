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
from pathlib import Path


def run(task_id: str, mutant: str | None) -> dict:
    from appworld import AppWorld
    from appworld.requester import CustomResponse, Requester

    originals = {m: getattr(Requester, f"_{m}") for m in ("post", "patch", "delete")}
    if mutant:
        def noop(self, url, data=None, *args, **kwargs):
            if mutant == "post" and ("auth/token" in url or "supervisor" in url):
                return originals["post"](self, url, data, *args, **kwargs)
            self.request_tracker.add_request(method=mutant, url=url, data=data)
            return CustomResponse(status_code=200, text="Success.")
        setattr(Requester, f"_{mutant}", noop)
    root = Path(os.environ["APPWORLD_ROOT"])
    code = (root / "data" / "tasks" / task_id / "ground_truth" / "compiled_solution.py").read_text()
    code += "\nsolution(apis, requester)\n"
    try:
        # Same settings as AppWorld's own ground-truth verification (appworld/verify.py).
        with AppWorld(task_id=task_id, experiment_name=f"audit_{mutant or 'reference'}",
                      raise_on_failure=False, ground_truth_mode="full") as world:
            output = world.execute(code)
            crashed = "Execution successful" not in str(output)
            result = world.evaluate().to_dict()
        return {"task": task_id, "mutant": mutant or "reference", "success": result["success"],
                "crashed": crashed, "num_tests": result["num_tests"], "num_failures": len(result["failures"])}
    except Exception:  # noqa: BLE001 - record and continue
        return {"task": task_id, "mutant": mutant or "reference", "error": traceback.format_exc(limit=1)[-300:]}
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
    for split in args.splits:
        for tid in (root / "data" / "datasets" / f"{split}.txt").read_text().split():
            calls = json.loads((root / "data" / "tasks" / tid / "ground_truth" / "api_calls.json").read_text())
            methods = {c["method"] for c in calls
                       if not (c["method"] == "post" and ("auth/token" in c["url"] or "supervisor" in c["url"]))}
            jobs.append((tid, None))
            jobs += [(tid, m) for m in ("post", "patch", "delete") if m in methods]
    jobs = jobs[args.shard::args.num_shards]
    out = args.out / f"runs_{args.shard}.jsonl"
    done = set()
    if out.exists():
        done = {(r["task"], r["mutant"]) for r in map(json.loads, out.read_text().splitlines())}
    for tid, mutant in jobs:
        if (tid, mutant or "reference") in done:
            continue
        rec = run(tid, mutant)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    sys.exit(main())
