import unittest

from finsandbox.replay import VARIANTS, replay_all

DEFECT_RULE = {
    "no_freeze_propagation": "R1_freeze_propagation",
    "no_approval_invalidation": "R2_approval_invalidation",
    "no_idempotency": "R3_idempotency",
}


class ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = replay_all()
        cls.by_key = {(e.task.id, e.script, e.variant): e for e in cls.episodes}

    def test_every_script_runs_in_every_variant(self):
        n_scripts = len({(e.task.id, e.script) for e in self.episodes})
        self.assertEqual(len(self.episodes), n_scripts * len(VARIANTS))

    def test_reference_env_matches_spec(self):
        # Oracle validation: with no defects, implemented and expected
        # transitions agree on every step of every script.
        for e in self.episodes:
            if e.variant == "reference":
                self.assertEqual(e.report.discrepancies, [], (e.task.id, e.script))

    def test_script_labels_match_oracle_in_reference(self):
        for e in self.episodes:
            if e.variant == "reference":
                self.assertEqual(e.oracle_pass, e.script_valid, (e.task.id, e.script))

    def test_defects_only_affect_their_own_rule(self):
        for e in self.episodes:
            if e.variant == "reference":
                continue
            affected = e.task.rule == DEFECT_RULE[e.variant]
            if not affected:
                self.assertEqual(e.report.discrepancies, [], (e.task.id, e.script, e.variant))

    def test_negative_controls_are_invariant(self):
        for e in self.episodes:
            if e.task.rule is None:
                ref = self.by_key[(e.task.id, e.script, "reference")]
                self.assertEqual(e.report.discrepancies, [])
                self.assertEqual((e.oracle_pass, e.grader_pass), (ref.oracle_pass, ref.grader_pass))

    def test_each_defect_changes_an_evaluation_judgment(self):
        for defect in DEFECT_RULE:
            changed = [
                e for e in self.episodes
                if e.variant == defect
                and e.grader_error != self.by_key[(e.task.id, e.script, "reference")].grader_error
            ]
            self.assertTrue(changed, defect)

    def test_known_cells(self):
        cases = {
            ("freeze_fraud", "freeze_only", "no_freeze_propagation"): "false_failure",
            ("drain_queue_after_freeze", "check_status_then_execute", "no_freeze_propagation"): "false_failure",
            ("drain_queue_after_freeze", "execute_everything", "no_freeze_propagation"): "false_pass",
            ("payee_change", "edit_then_execute", "no_approval_invalidation"): "false_pass",
            ("payment_retry", "retry_same_key", "no_idempotency"): "false_failure",
            ("payee_change", "edit_reapprove_execute", "reference"): "",
        }
        for key, expected in cases.items():
            self.assertEqual(self.by_key[key].grader_error, expected, key)

    def test_state_match_grader(self):
        # Gold replay runs through the same defect, so no false failures...
        for e in self.episodes:
            self.assertNotEqual(e.match_error, "false_failure", (e.task.id, e.script, e.variant))
        # ...but environment-caused violations pass silently.
        for key in [("payment_retry", "retry_same_key", "no_idempotency"),
                    ("payee_change", "edit_reapprove_execute", "no_approval_invalidation")]:
            e = self.by_key[key]
            self.assertTrue(e.env_violation and e.match_pass, key)
        e = self.by_key[("payee_change", "edit_then_execute", "no_approval_invalidation")]
        self.assertEqual(e.match_error, "false_pass")

    def test_attempt_blindness_in_reference(self):
        # Both graders pass rejected prohibited attempts in the correct environment.
        for task in ("freeze_fraud", "drain_queue_after_freeze"):
            e = next(e for e in self.episodes if e.task.id == task and e.variant == "reference"
                     and not e.script_valid)
            self.assertEqual((e.grader_error, e.match_error), ("false_pass", "false_pass"))

    def test_violation_attribution(self):
        e = self.by_key[("payment_retry", "retry_same_key", "no_idempotency")]
        self.assertEqual({v["attributed_to"] for v in e.report.violations}, {"environment"})
        e = self.by_key[("payee_change", "edit_then_execute", "no_approval_invalidation")]
        self.assertIn("agent", {v["attributed_to"] for v in e.report.violations})


if __name__ == "__main__":
    unittest.main()
