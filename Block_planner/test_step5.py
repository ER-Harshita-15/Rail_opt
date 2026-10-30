"""
Tests for Step 5 (baseline.py, metrics.py, whatif.py, bdms_export.py, app.py).
Run:  python -m unittest test_step5 -v
Small in-code fixtures come from test_horizons; the last classes need the CSVs from horizons.py.
"""
import json
import unittest

import pandas as pd

import baseline as bl
import bdms_export as bx
import horizons as h
import metrics as m
import optimizer as o
import whatif as wi
from test_horizons import D0, T_, W_, hand_plan, solve


def manual(T, W, order=bl.DEPTS, days=7):
    return bl.manual_baseline(T, W, start=D0, horizon_days=days, dept_order=order)


def errs(plan, W, T, days=7):
    return h.validate_parallel(plan, W, T, start=D0, horizon_days=days, merge=False, origin=D0)


class TestManualBaseline(unittest.TestCase):
    def test_one_department_per_window_and_no_parallel_crews(self):
        W = W_([("A", "S0", 0, 10, 13, True), ("B", "S0", 1, 10, 13, True)])
        T = T_([("E1", "ENGG", "S0", 1, 60, 5, .8), ("S1", "S&T", "S0", 9, 60, 5, .8)])
        plan, blocks = manual(T, W)
        self.assertEqual(len(plan), 2)
        self.assertEqual(len(blocks), 2)                                    # an optimiser would share one block
        self.assertFalse(blocks.merged.any())
        self.assertEqual(errs(plan, W, T), [])

    def test_a_department_batches_its_own_tasks_with_one_setup(self):
        W = W_([("A", "S0", 0, 10, 13, True)])
        T = T_([(f"E{i}", "ENGG", "S0", 1.0 + 5 * i, 50, 5, .8) for i in range(3)])   # 20 + 150 <= 180
        plan, blocks = manual(T, W)
        self.assertEqual((len(plan), len(blocks)), (3, 1))
        self.assertEqual(plan.start_min.min(), 10 * 60 + 20)

    def test_tasks_are_requested_most_overdue_first(self):
        W = W_([("A", "S0", 0, 10, 13, True)])                              # room for ONE 120-minute task
        T = T_([("NEW", "ENGG", "S0", 1, 120, 2, .9), ("OLD", "ENGG", "S0", 5, 120, 20, .1)])
        plan, _ = manual(T, W)
        self.assertEqual(plan.task_id.tolist(), ["OLD"])                    # overdue first, not by AI score

    def test_gang_pool_refuses_a_request_when_crews_are_scarce(self):
        # 3 sections hold a window at the same hour but only 2 tower wagons exist
        W = W_([(f"W{k}", f"S{k}", 0, 10, 13, True) for k in range(3)])
        T = T_([(f"P{k}", "TRD", f"S{k}", 3.0, 60, 5, .5) for k in range(3)])
        plan, blocks = manual(T, W)
        self.assertEqual(len(plan), 2)                                      # the third request is refused (gang pool)
        self.assertEqual(errs(plan, W, T), [])

    def test_blocks_per_day_and_earliest_start_are_respected(self):
        W = W_([("A", "S0", 0, 1, 4, True), ("B", "S0", 0, 10, 13, True), ("C", "S0", 0, 13, 16, True)])
        T = T_([(f"E{i}", "ENGG", "S0", 1.0 + 5 * i, 160, 5, .8) for i in range(3)])   # one task per window
        plan, blocks = manual(T, W)
        self.assertEqual(len(blocks), 2)                                    # at most 2 blocks per section-day
        T2 = T_([("L", "ENGG", "S0", 1, 60, -30, .9)])                      # earliest day 16: no window qualifies
        self.assertEqual(len(manual(T2, W)[0]), 0)

    def test_power_and_section_rules(self):
        W = W_([("A", "S0", 0, 10, 13, False), ("B", "S1", 0, 10, 13, True)])
        T = T_([("P", "TRD", "S0", 3, 60, 5, .8)])                          # power task, only a non-power window
        self.assertEqual(len(manual(T, W)[0]), 0)

    def test_every_order_is_constraint_compliant_on_random_data(self):
        import numpy as np
        rng = np.random.default_rng(2)
        W = W_([(f"W{s}_{d}_{k}", f"S{s}", d, a, b, bool(rng.random() < .7))
                for s in range(5) for d in range(4) for k, (a, b) in enumerate([(1, 4), (10, 13), (13, 16)])])
        rows = []
        for i in range(60):
            dept = str(rng.choice(["ENGG", "S&T", "TRD"]))
            rows.append((f"T{i:02d}", dept, f"S{int(rng.integers(5))}", float(rng.uniform(0, 22)),
                         int(rng.choice([60, 90, 120])), int(rng.integers(-10, 15)), float(rng.uniform(.2, .9)),
                         int(rng.integers(1, 6))))
        T = T_(rows)
        table, plans = bl.run_orders(T, W, start=D0, horizon_days=4)
        self.assertEqual(len(table), 6)
        self.assertEqual(table.violations.sum(), 0)

    def test_empty_plan_is_handled(self):
        W = W_([("A", "S0", 0, 10, 13, True)])
        T = T_([("X", "ENGG", "S9", 1, 60, 5, .8)])                         # no window on its section
        plan, blocks = manual(T, W)
        self.assertEqual((len(plan), len(blocks)), (0, 0))
        k = m.kpis(plan, blocks, T, W, start=D0, horizon_days=7)
        self.assertEqual(k["tasks_scheduled"], 0)


class TestKpis(unittest.TestCase):
    def setUp(self):
        self.W = W_([("A", "S0", 0, 10, 13, True), ("B", "S1", 8, 10, 13, True, 9)])
        self.T = T_([("T1", "ENGG", "S0", 1, 60, 5, .8, 3, "CRITICAL"),       # overdue 5, done day 0 -> 5 days past due
                     ("T2", "ENGG", "S1", 1, 90, -3, .6, 3, "LOW"),           # due day 3, done day 8 -> 5 days late
                     ("T3", "ENGG", "S1", 9, 170, 0, .6, 3, "LOW")])          # too long: never placeable
        p1, b1 = hand_plan([("T1", "A")], self.T, self.W)
        p2, b2 = hand_plan([("T2", "B")], self.T, self.W)
        self.plan, self.blocks = pd.concat([p1, p2]), pd.concat([b1, b2])

    def test_values(self):
        k = m.kpis(self.plan, self.blocks, self.T, self.W, start=D0, horizon_days=14)
        self.assertEqual((k["tasks_scheduled"], k["placeable_tasks"], k["blocks"]), (2, 2, 2))
        self.assertEqual(k["trains_disrupted"], 6 + 9)
        self.assertEqual(k["days_past_due_total"], 5 + 5)                     # T1: 0-(-5); T2: 8-3
        self.assertEqual(k["newly_late"], 1)                                  # T2
        self.assertEqual((k["overdue_scheduled"], k["overdue_total"]), (1, 1))
        self.assertEqual(k["crit_high_cleared"], "1/1")
        self.assertEqual(k["net_value"], o.net_value(m.with_week(self.plan, D0), m.with_week(self.blocks, D0), D0))
        self.assertEqual(k["merged_blocks"], 0)

    def test_unscheduled_tasks_count_until_the_end_of_the_horizon(self):
        p1, b1 = hand_plan([("T1", "A")], self.T, self.W)
        k = m.kpis(p1, b1, self.T, self.W, start=D0, horizon_days=14)
        self.assertEqual(k["days_past_due_total"], 5 + (14 - 3))              # T2 never scheduled: 14 - due day 3

    def test_productive_share_excludes_setup(self):
        k = m.kpis(self.plan, self.blocks, self.T, self.W, start=D0, horizon_days=14)
        self.assertAlmostEqual(k["productive_pct"], round(100 * (60 + 90) / (60 + 20 + 90 + 20), 1))


class TestCompare(unittest.TestCase):
    def test_verdicts_follow_direction(self):
        base = pd.DataFrame([dict(blocks=10, tasks_scheduled=5, merged_blocks=0),
                             dict(blocks=12, tasks_scheduled=7, merged_blocks=0)])
        orig = bl.NUMERIC_KPIS
        try:
            bl.NUMERIC_KPIS = ["blocks", "tasks_scheduled", "merged_blocks"]
            c = bl.compare(dict(blocks=6, tasks_scheduled=9, merged_blocks=3), base).set_index("key")
            self.assertEqual(c.verdict["blocks"], "better than all orders")      # lower is better
            self.assertEqual(c.verdict["tasks_scheduled"], "better than all orders")   # higher is better
            self.assertEqual(c.verdict["merged_blocks"], "info")
            c = bl.compare(dict(blocks=15, tasks_scheduled=3, merged_blocks=0), base).set_index("key")
            self.assertEqual(c.verdict["blocks"], "worse than all orders")
            self.assertEqual(c.verdict["tasks_scheduled"], "worse than all orders")
            c = bl.compare(dict(blocks=11, tasks_scheduled=6, merged_blocks=0), base).set_index("key")
            self.assertEqual(c.verdict["blocks"], "within the manual range")
        finally:
            bl.NUMERIC_KPIS = orig

    def test_best_order_is_the_highest_net_value(self):
        base = pd.DataFrame([dict(order="A > B", net_value=5), dict(order="B > A", net_value=9)])
        self.assertEqual(bl.best_order(base), "B > A")


class TestWhatIf(unittest.TestCase):
    def setUp(self):
        self.W = W_([("A", "S0", 0, 10, 13, True), ("B", "S1", 0, 10, 13, True), ("C", "S1", 1, 10, 13, True)])
        self.T = T_([("E1", "ENGG", "S0", 1, 60, 5, .8), ("S1", "S&T", "S1", 9, 60, 3, .7, 4)])
        plan, blocks, _, _, _ = solve(self.T, self.W)
        self.plan, self.blocks = m.with_week(plan, D0), m.with_week(blocks, D0)

    def test_emergency_task_has_department_defaults(self):
        e = wi.make_emergency_task(self.T, "S&T", "S1", 7.5, 90)
        r = e.iloc[0]
        self.assertEqual((r.task_id, r.dept, r.gang_type, r.tier), ("EMG-001", "S&T", "signal_team", "CRITICAL"))
        self.assertEqual(r.need_min, 90 + int(self.T[self.T.dept == "S&T"].crew_setup_min.max()))
        self.assertEqual(r.earliest_idx, -1 - h.EARLY_DAYS)
        self.assertEqual(list(e.columns), list(self.T.columns))
        with self.assertRaises(ValueError):
            wi.make_emergency_task(self.T, "TRD", "S1", 1, 60)                # no Traction tasks in this fixture

    def test_impact_places_the_task_and_splits_the_value(self):
        e = wi.make_emergency_task(self.T, "ENGG", "S0", 8.0, 60)
        r = wi.impact_in_week(self.T, self.W, self.plan, self.blocks, 1, e, start=D0, time_limit=10, workers=2)
        self.assertTrue(r["feasible"])
        s = r["summary"]
        self.assertEqual(len(r["emergency"]), 1)
        self.assertEqual(s["net_value_change"], s["emergency_value"] + s["cost_to_rest_of_plan"])
        self.assertEqual(s["tasks_before"], s["tasks_after"])                  # nothing displaced here
        self.assertEqual(h.validate_parallel(r["new_plan"], self.W, pd.concat([self.T, e], ignore_index=True),
                                             start=D0, origin=D0), [])

    def test_impact_reports_infeasible_emergency(self):
        e = wi.make_emergency_task(self.T, "ENGG", "S0", 8.0, 175)              # 20 + 175 > 180
        r = wi.impact_in_week(self.T, self.W, self.plan, self.blocks, 1, e, start=D0, time_limit=10, workers=2)
        self.assertFalse(r["feasible"])
        self.assertIn("no window", r["reason"])

    def test_scan_weeks_lists_each_week(self):
        e = wi.make_emergency_task(self.T, "ENGG", "S0", 8.0, 60)
        scan = wi.scan_weeks(self.T, self.W, self.plan, self.blocks, e, weeks=(1, 2), start=D0, time_limit=10, workers=2)
        self.assertEqual(scan.week.tolist(), [1, 2])
        self.assertTrue(scan.feasible[0])
        self.assertFalse(scan.feasible[1])                                      # week 2 has no window at all


class TestBdmsExport(unittest.TestCase):
    def setUp(self):
        self.W = W_([("A", "S0", 0, 10, 13, True), ("B", "S1", 1, 10, 13, True)])
        self.T = T_([("E1", "ENGG", "S0", 1, 60, 5, .8), ("P1", "TRD", "S0", 9, 60, 5, .8), ("S1", "S&T", "S1", 3, 60, 5, .8)])
        self.plan, self.blocks = hand_plan([("E1", "A"), ("P1", "A"), ("S1", "B")], self.T, self.W)

    def test_one_row_per_block_with_unique_ids(self):
        req = bx.to_bdms(self.plan, self.blocks)
        self.assertEqual(len(req), 2)
        self.assertEqual(req.block_id.nunique(), 2)
        a = req[req.section == "S0"].iloc[0]
        self.assertEqual((a.start, a.end, a.duration_min), ("10:00", "13:00", 180))
        self.assertTrue(a.power_block_required)                                 # the Traction task needs a power block
        self.assertEqual(a.n_work_items, 2)
        self.assertIn("parallel", a.remarks)
        self.assertIn("approval", a.status)
        self.assertFalse(req[req.section == "S1"].iloc[0].power_block_required)

    def test_work_items_match_blocks(self):
        items = bx.work_items(self.plan)
        req = bx.to_bdms(self.plan, self.blocks)
        self.assertEqual(len(items), 3)
        self.assertEqual(set(items.block_id), set(req.block_id))
        self.assertEqual(items[items.task_id == "E1"].start.iloc[0], "10:20")    # after the crew's 20 min setup

    def test_json_round_trip(self):
        data = json.loads(bx.to_json(self.plan, self.blocks))
        self.assertEqual(len(data), 2)
        self.assertEqual(sum(len(b["work_item_list"]) for b in data), 3)


_HAS_RESULTS = (m.OUT / "block_plan_4w.csv").exists() and (o.DATA / "tasks_common_schema.csv").exists()


@unittest.skipUnless(_HAS_RESULTS, "run horizons.py first")
class TestRealResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T, cls.W = m.load_inputs()
        cls.res = m.load_results()
        cls.base, cls.plans = bl.run_orders(cls.T, cls.W)

    def test_all_manual_orders_satisfy_the_hard_constraints(self):
        self.assertEqual(self.base.violations.sum(), 0)
        self.assertEqual(len(self.base), 6)

    def test_optimised_kpis_match_the_saved_summary(self):
        k = m.kpis(self.res["plan"], self.res["blocks"], self.T, self.W)
        self.assertEqual(k["net_value"], int(self.res["summary"].net_value.sum()))
        self.assertEqual(k["tasks_scheduled"], self.res["plan"].task_id.nunique())

    def test_comparison_has_every_kpi(self):
        k = m.kpis(self.res["plan"], self.res["blocks"], self.T, self.W)
        c = bl.compare(k, self.base)
        self.assertEqual(set(c.key), set(m.KPI_LABELS))

    def test_export_covers_every_block_and_task(self):
        plan, blocks = m.with_week(self.res["plan"]), m.with_week(self.res["blocks"])
        req = bx.to_bdms(plan, blocks)
        self.assertEqual(len(req), len(blocks))
        self.assertEqual(len(bx.work_items(plan)), len(plan))
        self.assertEqual(req.n_work_items.sum(), len(plan))


@unittest.skipUnless(_HAS_RESULTS, "run horizons.py first")
class TestDashboardSmoke(unittest.TestCase):
    def test_app_renders_and_whatif_runs(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(m.ROOT / "app.py"), default_timeout=240)
        at.run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertEqual(len(at.tabs), 6)
        labels = [x.label for x in at.metric]
        self.assertIn("Blocks opened", labels)
        at.selectbox(key="wi_mode").set_value("Week 1").run()
        at.button(key="wi_btn").click().run(timeout=240)
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        self.assertIn("Cost to the rest of the plan", [x.label for x in at.metric])


if __name__ == "__main__":
    unittest.main(verbosity=2)
