"""
Tests for optimizer.py.   Run:  python -m unittest test_optimizer -v
All tests except the last use small in-code fixtures (no CSV files needed).
"""
import unittest

import numpy as np
import pandas as pd

import optimizer as o

D0 = pd.Timestamp("2026-10-26")


def make_windows(rows):
    """rows: (window_id, section, day_offset, start_h, end_h, power_ok)"""
    return pd.DataFrame([dict(window_id=w, section_id=s, date=D0 + pd.Timedelta(days=d),
                              start_hour=a, end_hour=b, length_min=(b - a) * 60,
                              trains_affected=6, goods_trains_affected=2,
                              power_block_possible=p) for w, s, d, a, b, p in rows])


def make_tasks(rows):
    """rows: (task_id, section, duration, setup, block_type, gang, severity, overdue, score)"""
    return pd.DataFrame([dict(task_id=t, dept="X", section_id=s, task="job", tier="HIGH",
                              duration_min=du, setup_min=su, need_min=du + su, block_type=bt,
                              gang_type=g, severity=sv, overdue_days=ov, priority_score=sc)
                         for t, s, du, su, bt, g, sv, ov, sc in rows])


def plan_from(pairs, windows, tasks):
    """Hand-built plan: list of (task_id, window_id) -> plan DataFrame like finish_plan."""
    W = windows.set_index("window_id")
    rows = [dict(window_id=w, section_id=W.section_id[w], date=W.date[w],
                 win_start=int(W.start_hour[w]) * 60, win_end=int(W.end_hour[w]) * 60, task_id=t)
            for t, w in pairs]
    return o.finish_plan(pd.DataFrame(rows), tasks, windows)


def random_instance(seed=3, n_sections=5, days=3, n_tasks=40):
    rng = np.random.default_rng(seed)
    wrows = []
    for s in range(n_sections):
        for d in range(days):
            for k, (a, b) in enumerate([(1, 4), (10, 13), (13, 16)]):
                wrows.append((f"W{s}_{d}_{k}", f"S{s}", d, a, b, bool(rng.random() < 0.7)))
    trows = []
    for i in range(n_tasks):
        bt = str(rng.choice(["LINE", "POWER"]))
        g = {"LINE": str(rng.choice(["track_gang", "signal_team"])), "POWER": "ohe_tower_wagon"}[bt]
        trows.append((f"T{i:03d}", f"S{int(rng.integers(n_sections))}", int(rng.choice([60, 90, 120, 150])),
                      int(rng.choice([15, 20, 25])), bt, g, int(rng.integers(1, 6)),
                      int(rng.integers(-10, 20)), float(rng.uniform(0.2, 0.9))))
    return make_tasks(trows), make_windows(wrows)


class TestObjectiveAndGap(unittest.TestCase):
    def test_objective_equals_net_value_and_independent_recompute(self):
        tasks, windows = random_instance()
        plan, blocks, info, W, T = o.solve(tasks, windows, horizon_days=3, start=D0,
                                           time_limit=20, workers=4)
        self.assertIn(info["status"], ("OPTIMAL", "FEASIBLE"))
        # independent recompute straight from the input tables, not from plan columns
        t, w = tasks.set_index("task_id"), windows.set_index("window_id")
        gain = sum(o.assignment_value(t.priority_score[r.task_id], t.severity[r.task_id],
                                      t.overdue_days[r.task_id], (w.date[r.window_id] - D0).days)
                   for r in plan.itertuples())
        cost = sum(o.block_cost(w.trains_affected[b], w.goods_trains_affected[b])
                   for b in set(plan.window_id))
        self.assertEqual(info["objective"], gain - cost)
        self.assertEqual(info["objective"], o.net_value(plan, blocks, D0))

    def test_gap_math(self):
        self.assertEqual(o.compute_gap(100, 100), (0, 0.0))
        abs_gap, rel = o.compute_gap(900, 927)
        self.assertEqual(abs_gap, 27)
        self.assertAlmostEqual(rel, 27 / 900)

    def test_gap_reported_from_solver_values(self):
        tasks, windows = random_instance(seed=5)
        _, _, info, _, _ = o.solve(tasks, windows, horizon_days=3, start=D0, time_limit=20)
        self.assertGreaterEqual(info["bound"] + 1e-6, info["objective"])
        self.assertAlmostEqual(info["abs_gap"], info["bound"] - info["objective"])
        if info["status"] == "OPTIMAL":
            self.assertTrue(info["proven_optimal"])
        else:
            self.assertFalse(info["proven_optimal"])


class TestValidator(unittest.TestCase):
    def setUp(self):
        # three sections whose windows all cover 10:00-13:00 on day 0 -> they compete for gangs
        self.windows = make_windows([(f"W{k}", f"S{k}", 0, 10, 13, True) for k in range(3)]
                                    + [("X1", "S0", 0, 1, 4, True), ("X2", "S0", 0, 13, 16, True)])
        self.tasks = make_tasks(
            [(f"P{k}", f"S{k}", 60, 20, "POWER", "ohe_tower_wagon", 3, 0, .5) for k in range(3)]
            + [("L0", "S0", 60, 20, "LINE", "track_gang", 3, 0, .5),
               ("L1", "S0", 60, 20, "LINE", "track_gang", 3, 0, .5),
               ("L2", "S0", 60, 20, "LINE", "track_gang", 3, 0, .5),
               ("BIG", "S0", 170, 20, "LINE", "track_gang", 3, 0, .5)])

    def _errs(self, pairs, horizon=7):
        plan, blocks = plan_from(pairs, self.windows, self.tasks)
        return o.validate(plan, blocks, self.windows, self.tasks, D0, horizon)

    def test_valid_plan_has_no_errors(self):
        self.assertEqual(self._errs([("P0", "W0"), ("P1", "W1"), ("L0", "X1")]), [])

    def test_gang_overuse_detected(self):          # 3 tower wagons wanted, only 2 exist
        errs = self._errs([("P0", "W0"), ("P1", "W1"), ("P2", "W2")])
        self.assertTrue(any(e.startswith("[GANG] ohe_tower_wagon") for e in errs), errs)
        self.assertEqual(sum(e.startswith("[GANG]") for e in errs), 3)   # hours 10, 11, 12

    def test_gang_adjacent_windows_do_not_conflict(self):
        errs = self._errs([("L0", "W0"), ("L1", "X2")])      # 10-13 and 13-16, same gang type
        self.assertEqual([e for e in errs if e.startswith("[GANG]")], [])

    def test_capacity_detected(self):
        errs = self._errs([("BIG", "W0"), ("L0", "W0")])
        self.assertTrue(any(e.startswith("[CAPACITY]") for e in errs), errs)

    def test_power_detected(self):
        windows = self.windows.copy()
        windows.loc[windows.window_id == "W0", "power_block_possible"] = False
        plan, blocks = plan_from([("P0", "W0")], windows, self.tasks)
        errs = o.validate(plan, blocks, windows, self.tasks, D0, 7)
        self.assertTrue(any(e.startswith("[POWER]") for e in errs), errs)

    def test_section_detected(self):
        self.assertTrue(any(e.startswith("[SECTION]") for e in self._errs([("P1", "W0")])))

    def test_daily_block_limit_detected(self):     # S0 has 3 windows on day 0
        errs = self._errs([("L0", "W0"), ("L1", "X1"), ("L2", "X2")])
        self.assertTrue(any(e.startswith("[BLOCKS_PER_DAY]") for e in errs), errs)

    def test_duplicate_and_horizon_detected(self):
        plan, blocks = plan_from([("L0", "W0"), ("L0", "X1")], self.windows, self.tasks)
        errs = o.validate(plan, blocks, self.windows, self.tasks, D0 + pd.Timedelta(days=2), 7)
        self.assertTrue(any(e.startswith("[DUPLICATE]") for e in errs))
        self.assertTrue(any(e.startswith("[HORIZON]") for e in errs))

    def test_validator_ignores_tampered_plan_columns(self):
        plan, blocks = plan_from([("P0", "W0"), ("P1", "W1"), ("P2", "W2")], self.windows, self.tasks)
        plan["gang_type"] = "track_gang"            # tamper: validator must use the tasks table
        errs = o.validate(plan, blocks, self.windows, self.tasks, D0, 7)
        self.assertTrue(any(e.startswith("[GANG]") for e in errs))

    def test_sequence_overlap_detected(self):
        plan, blocks = plan_from([("L0", "W0"), ("L1", "W0")], self.windows, self.tasks)
        plan.loc[plan.task_id == "L1", "start_min"] = plan.start_min.min()   # force overlap
        errs = o.validate(plan, blocks, self.windows, self.tasks, D0, 7)
        self.assertTrue(any(e.startswith("[SEQUENCE]") for e in errs), errs)


class TestUnschedulable(unittest.TestCase):
    def test_reasons_and_shortfall(self):
        windows = make_windows([("A", "S0", 0, 10, 13, False), ("B", "S1", 0, 10, 13, True)])
        tasks = make_tasks([("LONG", "S1", 180, 20, "LINE", "track_gang", 3, 0, .5),
                            ("NOPOW", "S0", 60, 20, "POWER", "ohe_tower_wagon", 3, 0, .5),
                            ("NOWIN", "S9", 60, 20, "LINE", "track_gang", 3, 0, .5),
                            ("OK", "S0", 60, 20, "LINE", "track_gang", 3, 0, .5)])
        u = o.find_unschedulable(tasks, windows).set_index("task_id")
        self.assertEqual(u.reason["LONG"], "TOO_LONG")
        self.assertEqual(u.shortfall_min["LONG"], 20)
        self.assertEqual(u.reason["NOPOW"], "NO_POWER_WINDOW")
        self.assertEqual(u.reason["NOWIN"], "NO_WINDOW_IN_HORIZON")
        self.assertNotIn("OK", u.index)
        plan, _, info, _, _ = o.solve(tasks, windows, horizon_days=1, start=D0, time_limit=10)
        self.assertEqual(set(plan.task_id), {"OK"})
        self.assertEqual(info["too_long"], ["LONG"])

    def test_power_task_too_long_for_power_windows_only(self):
        # a long window exists but is not power-capable; the power-capable one is too short
        windows = make_windows([("A", "S0", 0, 1, 5, False), ("B", "S0", 0, 10, 12, True)])
        tasks = make_tasks([("P", "S0", 150, 20, "POWER", "ohe_tower_wagon", 3, 0, .5)])
        u = o.find_unschedulable(tasks, windows)
        self.assertEqual(u.reason.tolist(), ["TOO_LONG"])
        self.assertEqual(u.longest_eligible_window_min.tolist(), [120])


class TestBaselines(unittest.TestCase):
    def test_greedy_plans_respect_all_constraints(self):
        for seed in range(5):
            tasks, windows = random_instance(seed=seed, n_tasks=70)
            for rule in ("first_fit", "value"):
                plan, blocks = o.greedy_plan(tasks, windows, horizon_days=3, start=D0, rule=rule)
                self.assertEqual(o.validate(plan, blocks, windows, tasks, D0, 3), [],
                                 f"seed {seed} rule {rule}")

    def test_all_methods_bounded_by_cpsat_bound_and_optimal_dominates(self):
        tasks, windows = random_instance(seed=11, n_sections=3, days=2, n_tasks=25)
        plan, blocks, info, _, _ = o.solve(tasks, windows, horizon_days=2, start=D0, time_limit=30)
        self.assertEqual(o.validate(plan, blocks, windows, tasks, D0, 2), [])
        for rule in ("first_fit", "value"):
            gp, gb = o.greedy_plan(tasks, windows, horizon_days=2, start=D0, rule=rule)
            v = o.net_value(gp, gb, D0)
            self.assertLessEqual(v, info["bound"] + 1e-6)            # always true
            if info["proven_optimal"]:
                self.assertLessEqual(v, info["objective"])           # true only when proven

    def test_cpsat_plan_respects_all_constraints_on_gang_heavy_instance(self):
        tasks, windows = random_instance(seed=21, n_sections=6, days=2, n_tasks=60)
        plan, blocks, info, _, _ = o.solve(tasks, windows, horizon_days=2, start=D0,
                                           time_limit=20, workers=4)
        self.assertEqual(o.validate(plan, blocks, windows, tasks, D0, 2), [])


@unittest.skipUnless((o.DATA / "tasks_common_schema.csv").exists()
                     and (o.OUT / "prioritized_tasks.csv").exists(),
                     "run data_gen.py and priority.py first")
class TestRealData(unittest.TestCase):
    def test_week1_network_is_feasible_and_consistent(self):
        tasks, windows = o.load()
        plan, blocks, info, _, T = o.solve(tasks, windows, horizon_days=7, time_limit=30)
        self.assertEqual(o.validate(plan, blocks, windows, tasks), [])
        self.assertEqual(info["objective"], o.net_value(plan, blocks, info["start"]))
        self.assertEqual(len(info["unschedulable"]), len(info["too_long"]) +
                         (info["unschedulable"].reason != "TOO_LONG").sum())
        for rule in ("first_fit", "value"):
            gp, gb = o.greedy_plan(tasks, windows, rule=rule)
            self.assertEqual(o.validate(gp, gb, windows, tasks), [])
            self.assertLessEqual(o.net_value(gp, gb), info["bound"] + 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
