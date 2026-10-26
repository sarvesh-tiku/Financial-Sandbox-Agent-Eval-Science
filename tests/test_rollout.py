import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from finsandbox.rollout import Session
from finsandbox.tasks import TASKS

SRC = str(Path(__file__).resolve().parents[1] / "src")
RETRY = next(t for t in TASKS if t.id == "payment_retry")
PAY = {"src": "A", "dst": "V", "amount": 500, "invoice_ref": "INV-7", "idempotency_key": "k-inv7"}


class SessionTest(unittest.TestCase):
    def test_fault_masks_first_result_but_takes_effect(self):
        s = Session(RETRY, "reference")
        self.assertIn("error", s.call("submit_payment", PAY))
        self.assertTrue(s.call("submit_payment", PAY)["replayed"])
        rec = s.finish(model="test")
        self.assertTrue(rec["oracle_pass"] and rec["grader_pass"])

    def test_rebuild_matches_live_session(self):
        live = Session(RETRY, "no_idempotency")
        live.call("submit_payment", PAY)
        live.call("submit_payment", PAY)
        rebuilt = Session.rebuild("payment_retry", "no_idempotency", [[t, a] for t, a in live.actions])
        self.assertEqual(live.env.snapshot(), rebuilt.env.snapshot())
        live.env.db.close()
        self.assertEqual(rebuilt.finish(model="t")["grader_error"], "false_failure")

    def test_failed_first_call_keeps_seed_data(self):
        # Regression: a failing first call used to roll back the uncommitted seed accounts.
        s = Session(RETRY, "reference")
        self.assertIn("error", s.call("get_account", {"account_id": "Z"}))
        self.assertEqual(set(s.env.snapshot()["accounts"]), {"A", "B", "V"})
        s.env.db.close()

    def test_unknown_tool_is_not_logged(self):
        s = Session(RETRY, "reference")
        self.assertIn("error", s.call("delete_everything", {}))
        self.assertEqual(s.actions, [])
        s.env.db.close()


class CliTest(unittest.TestCase):
    def bank(self, state, *args):
        out = subprocess.run([sys.executable, "-m", "finsandbox.cli", "--state", state, *args],
                             capture_output=True, text=True, env={"PYTHONPATH": SRC}, check=True)
        return out.stdout

    def test_episode_round_trip_hides_variant(self):
        with tempfile.TemporaryDirectory() as state:
            reg = json.loads(self.bank(state, "register", "--model", "m", "--rep", "0",
                                       "--variants", "no_idempotency"))
            eid = reg["episodes"][[t.id for t in TASKS].index("payment_retry")]
            self.assertNotIn("no_idempotency", self.bank(state, "start", eid))
            self.bank(state, "call", eid, "submit_payment", json.dumps(PAY))
            self.bank(state, "call", eid, "submit_payment", json.dumps(PAY))
            self.bank(state, "done", eid, "paid")
            out = Path(state) / "out.jsonl"
            self.bank(state, "collect", "--out", str(out))
            rec = next(json.loads(l) for l in out.read_text().splitlines() if json.loads(l)["episode_id"] == eid)
            self.assertEqual((rec["oracle_pass"], rec["grader_pass"]), (True, False))


if __name__ == "__main__":
    unittest.main()
