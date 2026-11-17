"""Tests for the data wiring and the Step 7 analyses (analysis.py)."""
import unittest

import analysis as A
import data_gen as dg
import horizons as h
import metrics as m
import optimizer as o

_HAS_RESULTS = (m.OUT / "block_plan_4w.csv").exists()


class TestDataWiring(unittest.TestCase):
    def test_generator_writes_where_the_planner_reads(self):
        self.assertEqual(dg.OUT.resolve(), o.DATA.resolve())

    def test_priority_reads_and_writes_planner_folders(self):
        try:
            import priority as pr
        except ImportError as e:                     # lightgbm / shap are optional for the planner itself
            self.skipTest(f"priority dependencies missing: {e}")
        self.assertEqual(pr.DATA.resolve(), o.DATA.resolve())
        self.assertEqual(pr.OUT.resolve(), o.OUT.resolve())

    def test_generator_reproduces_committed_tasks(self):
        tasks = dg.build_tasks(dg.make_sections())
        committed = o.load()[0]
        self.assertEqual(len(tasks), len(committed))
        self.assertEqual(sorted(tasks.task_id), sorted(committed.task_id))


@unittest.skipUnless(_HAS_RESULTS, "run horizons.py first")
class TestAnalyses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T, cls.W = m.load_inputs()
        res = m.load_results()
        cls.plan, cls.blocks = res["plan"], res["blocks"]

    def test_late_task_table_matches_the_kpi(self):
        lt = A.late_tasks(self.T, self.W, self.plan, self.blocks)
        k = m.kpis(m.with_week(self.plan), m.with_week(self.blocks), self.T, self.W)
        self.assertEqual(len(lt), k["newly_late"])
        if len(lt):
            self.assertTrue((lt.days_late > 0).all())
            self.assertTrue((lt.start_day == lt.due_day + lt.days_late).all())

    def test_safety_rule_is_restored_after_the_counterfactual(self):
        before = dict(h.CONFLICT_PAIRS)
        with A.no_safety_rule():
            self.assertEqual(h.conflict_pairs(self.T), [])
        self.assertEqual(h.CONFLICT_PAIRS, before)
        self.assertGreater(len(h.conflict_pairs(self.T)), 0)

    def test_saved_plan_never_puts_a_close_pair_in_one_block(self):
        win = self.plan.drop_duplicates("task_id").set_index("task_id").window_id
        for i, j in h.conflict_pairs(self.T):
            a, b = self.T.task_id[i], self.T.task_id[j]
            if a in win.index and b in win.index:
                self.assertNotEqual(win[a], win[b], (a, b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
