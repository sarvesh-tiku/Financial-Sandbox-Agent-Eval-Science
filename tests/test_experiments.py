import unittest

from finsandbox.experiments import e2_profiles, e2_rankings, e2_skill_credit, outcome_table
from finsandbox.replay import replay_all


class ProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = e2_profiles(outcome_table(replay_all()))

    def _get(self, rows, **kw):
        return next(r for r in rows if all(r[k] == v for k, v in kw.items()))

    def test_true_success_is_variant_invariant(self):
        for r in self.rows:
            ref = self._get(self.rows, grader=r["grader"], profile=r["profile"], variant="reference")
            self.assertEqual(r["true"], ref["true"])

    def test_freeze_defect_inverts_freeze_credit_under_outcome_grader(self):
        c = self._get(e2_skill_credit(self.rows), grader="outcome", skill="freeze_awareness",
                      variant="no_freeze_propagation")
        self.assertLess(c["measured_credit"], 0)
        self.assertGreater(c["true_credit"], 0)

    def test_idempotency_defect_erases_retry_credit_under_outcome_grader(self):
        c = self._get(e2_skill_credit(self.rows), grader="outcome", skill="retry_discipline",
                      variant="no_idempotency")
        self.assertEqual(c["measured_credit"], 0)

    def test_ranking_degrades_under_freeze_defect(self):
        ranks = e2_rankings(self.rows)
        ref = self._get(ranks, grader="outcome", variant="reference")
        r1 = self._get(ranks, grader="outcome", variant="no_freeze_propagation")
        self.assertLess(r1["kendall_tau_b"], ref["kendall_tau_b"])
        self.assertGreater(len(r1["inverted_pairs"]), len(ref["inverted_pairs"]))


if __name__ == "__main__":
    unittest.main()
