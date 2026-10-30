"""
Tests for horizons.py.   Run:  python -m unittest test_horizons -v
Small in-code fixtures; the last test needs the CSVs from data_gen.py / priority.py.
"""
import unittest

import numpy as np
import pandas as pd

import horizons as h
import optimizer as o

D0 = pd.Timestamp("2026-10-26")
GANG = {"ENGG": "track_gang", "S&T": "signal_team", "TRD": "ohe_tower_wagon"}
SETUP = {"ENGG": 20, "S&T": 15, "TRD": 25}


def W_(rows):
    """rows: (window_id, section, day, start_h, end_h, power_ok[, trains])"""
    out = []
    for r in rows:
        wid, sec, d, a, b, p = r[:6]
        out.append(dict(window_id=wid, section_id=sec, date=D0 + pd.Timedelta(days=d), start_hour=a,
                        end_hour=b, length_min=(b - a) * 60, trains_affected=r[6] if len(r) > 6 else 6,
                        goods_trains_affected=2, power_block_possible=p))
    return pd.DataFrame(out)


def T_(rows, early_days=14):
    """rows: (task_id, dept, section, km, duration, overdue_days, score[, severity])"""
    out = []
    for r in rows:
        tid, dept, sec, km, dur, ov, sc = r[:7]
        out.append(dict(task_id=tid, dept=dept, section_id=sec, km=km, task="job", tier="HIGH",
                        duration_min=dur, setup_min=SETUP[dept], block_type="POWER" if dept == "TRD" else "LINE",
                        gang_type=GANG[dept], severity=r[7] if len(r) > 7 else 3,
                        overdue_days=ov, priority_score=sc))
    return h.prepare_tasks(pd.DataFrame(out), early_days)


def solve(T, W, **kw):
    kw.setdefault("start", D0)
    kw.setdefault("time_limit", 20)
    kw.setdefault("workers", 4)
    return h.solve_weekly(T, W, **kw)


def hand_plan(pairs, T, W):
    Wi = W.set_index("window_id")
    rows = pd.DataFrame([dict(window_id=w, section_id=Wi.section_id[w], date=Wi.date[w],
                              win_start=int(Wi.start_hour[w]) * 60, win_end=int(Wi.end_hour[w]) * 60,
                              task_id=t) for t, w in pairs])
    return h.finish_plan_parallel(rows, T, W)


class TestParallelAndSharedSetup(unittest.TestCase):
    def test_shared_setup_fits_what_sequential_setup_would_not(self):
        W = W_([("A", "S0", 0, 10, 13, True)])                       # 180 min
        T = T_([(f"E{i}", "ENGG", "S0", 1.0 + 5 * i, 50, 5, .8) for i in range(3)])  # 3 x (50+20) = 210 > 180
        plan, blocks, info, _, _ = solve(T, W)
        self.assertEqual(len(plan), 3)                                # 20 setup + 150 work = 170 <= 180
        self.assertEqual(len(blocks), 1)
        self.assertEqual(h.validate_parallel(plan, W, T), [])
        self.assertEqual(plan.start_min.min(), 10 * 60 + 20)          # work starts after the single setup
        self.assertLessEqual(plan.end_min.max(), 13 * 60)

    def test_departments_work_in_parallel(self):
        W = W_([("A", "S0", 0, 10, 13, True)])
        T = T_([("E1", "ENGG", "S0", 1, 70, 5, .8), ("E2", "ENGG", "S0", 5, 70, 5, .8),
                ("S1", "S&T", "S0", 10, 70, 5, .8), ("S2", "S&T", "S0", 15, 70, 5, .8)])
        plan, blocks, info, _, _ = solve(T, W)
        self.assertEqual(len(plan), 4)                                # 280 min of work in a 180 min block
        self.assertTrue(blocks.merged.iloc[0])
        self.assertEqual(h.validate_parallel(plan, W, T), [])

    def test_crew_capacity_still_binds(self):
        W = W_([("A", "S0", 0, 10, 13, True)])
        T = T_([(f"E{i}", "ENGG", "S0", 1.0 + 5 * i, 60, 5, .8) for i in range(3)])  # 20 + 180 = 200 > 180
        plan, _, _, _, _ = solve(T, W)
        self.assertEqual(len(plan), 2)
        self.assertEqual(h.validate_parallel(plan, W, T), [])

    def test_no_merge_ablation_forces_separate_blocks(self):
        W = W_([("A", "S0", 0, 10, 13, True), ("B", "S0", 1, 10, 13, True)])
        T = T_([("E1", "ENGG", "S0", 1, 70, 5, .8), ("S1", "S&T", "S0", 10, 70, 5, .8)])
        _, b1, _, _, _ = solve(T, W, merge=True)
        p2, b2, _, _, _ = solve(T, W, merge=False)
        self.assertEqual(len(b1), 1)
        self.assertEqual(len(b2), 2)
        self.assertEqual(h.validate_parallel(p2, W, T, merge=False), [])


class TestSafetyRule(unittest.TestCase):
    def setUp(self):
        self.W = W_([("A", "S0", 0, 10, 13, True)])

    def test_close_st_and_traction_cannot_share_a_block(self):
        T = T_([("S1", "S&T", "S0", 5.0, 60, 5, .8), ("P1", "TRD", "S0", 5.5, 60, 5, .8)])
        plan, _, _, _, _ = solve(T, self.W)
        self.assertEqual(len(plan), 1)
        self.assertEqual(h.validate_parallel(plan, self.W, T), [])

    def test_distant_st_and_traction_can_share_a_block(self):
        T = T_([("S1", "S&T", "S0", 5.0, 60, 5, .8), ("P1", "TRD", "S0", 9.0, 60, 5, .8)])
        plan, blocks, _, _, _ = solve(T, self.W)
        self.assertEqual(len(plan), 2)
        self.assertTrue(blocks.merged.iloc[0])

    def test_engineering_is_not_restricted(self):
        T = T_([("E1", "ENGG", "S0", 5.0, 60, 5, .8), ("P1", "TRD", "S0", 5.2, 60, 5, .8)])
        plan, _, _, _, _ = solve(T, self.W)
        self.assertEqual(len(plan), 2)


class TestValidatorParallel(unittest.TestCase):
    def setUp(self):
        self.W = W_([(f"W{k}", f"S{k}", 0, 10, 13, True) for k in range(3)]
                    + [("X1", "S0", 0, 1, 4, True), ("X2", "S0", 0, 13, 16, True), ("Y", "S0", 3, 10, 13, True)])
        self.T = T_([(f"P{k}", "TRD", f"S{k}", 3.0, 60, 5, .5) for k in range(3)]
                    + [("E1", "ENGG", "S0", 1, 60, 5, .5), ("E2", "ENGG", "S0", 8, 60, 5, .5),
                       ("E3", "ENGG", "S0", 12, 60, 5, .5), ("S1", "S&T", "S0", 3.2, 60, 5, .5),
                       ("EARLY", "ENGG", "S0", 4, 60, -30, .5)])           # earliest day 16

    def errs(self, pairs, **kw):
        plan, _ = hand_plan(pairs, self.T, self.W)
        kw.setdefault("start", D0)
        return h.validate_parallel(plan, self.W, self.T, **kw)

    def has(self, errs, code):
        self.assertTrue(any(e.startswith(f"[{code}]") for e in errs), errs)

    def test_valid_plan(self):
        self.assertEqual(self.errs([("E1", "W0"), ("E2", "W0"), ("P1", "W1")]), [])

    def test_crew_capacity(self):
        self.has(self.errs([("E1", "W0"), ("E2", "W0"), ("E3", "W0")]), "CREW_CAPACITY")   # 20 + 180 > 180

    def test_gang_pool(self):
        errs = self.errs([("P0", "W0"), ("P1", "W1"), ("P2", "W2")])
        self.has(errs, "GANG")
        self.assertEqual(sum(e.startswith("[GANG]") for e in errs), 3)

    def test_gang_adjacent_windows_ok(self):
        self.assertEqual(self.errs([("E1", "W0"), ("E2", "X2")]), [])

    def test_safety(self):
        self.has(self.errs([("P0", "W0"), ("S1", "W0")]), "SAFETY")

    def test_blocks_per_day(self):
        self.has(self.errs([("E1", "W0"), ("E2", "X1"), ("E3", "X2")]), "BLOCKS_PER_DAY")

    def test_no_merge(self):
        self.has(self.errs([("E1", "W0"), ("S1", "W0")], merge=False), "NO_MERGE")

    def test_early_start(self):
        self.has(self.errs([("EARLY", "W0")]), "EARLY")

    def test_duplicate_and_horizon(self):
        errs = self.errs([("E1", "W0"), ("E1", "Y")], horizon_days=3)
        self.has(errs, "DUPLICATE")
        self.has(errs, "HORIZON")

    def test_sequence_overlap(self):
        plan, _ = hand_plan([("E1", "W0"), ("E2", "W0")], self.T, self.W)
        plan.loc[plan.task_id == "E2", "start_min"] = plan.start_min.min()
        self.has(h.validate_parallel(plan, self.W, self.T, start=D0), "SEQUENCE")

    def test_validator_uses_tasks_table_not_plan_columns(self):
        plan, _ = hand_plan([("E1", "W0"), ("E2", "W0"), ("E3", "W0")], self.T, self.W)
        plan["duration_min"] = 1                                    # tamper
        self.has(h.validate_parallel(plan, self.W, self.T, start=D0), "CREW_CAPACITY")


class TestDueWindowAndObjective(unittest.TestCase):
    def test_task_never_scheduled_before_earliest_start(self):
        W = W_([(f"W{d}", "S0", d, 10, 13, True) for d in range(7)])
        T = T_([("A", "ENGG", "S0", 2, 60, -17, .9)])                # due day 17 -> earliest day 3
        plan, _, info, _, _ = solve(T, W, horizon_days=7)
        self.assertEqual(len(plan), 1)
        self.assertGreaterEqual((plan.date.iloc[0] - D0).days, 3)
        self.assertEqual(h.validate_parallel(plan, W, T, horizon_days=7), [])

    def test_objective_equals_net_value_and_bound(self):
        rng = np.random.default_rng(4)
        W = W_([(f"W{s}_{d}_{k}", f"S{s}", d, a, b, bool(rng.random() < .7))
                for s in range(5) for d in range(3) for k, (a, b) in enumerate([(1, 4), (10, 13), (13, 16)])])
        rows = []
        for i in range(40):
            dept = str(rng.choice(["ENGG", "S&T", "TRD"]))
            rows.append((f"T{i:02d}", dept, f"S{int(rng.integers(5))}", float(rng.uniform(0, 22)),
                         int(rng.choice([60, 90, 120])), int(rng.integers(-10, 15)), float(rng.uniform(.2, .9)),
                         int(rng.integers(1, 6))))
        T = T_(rows)
        plan, blocks, info, _, _ = solve(T, W, horizon_days=3)
        self.assertEqual(info["objective"], o.net_value(plan, blocks, D0))
        self.assertGreaterEqual(info["bound"] + 1e-6, info["objective"])
        self.assertEqual(h.validate_parallel(plan, W, T, horizon_days=3), [])


class TestMonthlyAndRolling(unittest.TestCase):
    def setUp(self):
        self.W = W_([("W0", "S0", 0, 10, 13, True), ("W1", "S0", 7, 10, 13, True), ("W2", "S0", 14, 10, 13, True)])
        self.T = T_([(f"E{i}", "ENGG", "S0", 2.0 * i, 60, 30, .8) for i in range(6)])   # need 80 each

    def per_week(self, reserve):
        a, info = h.solve_monthly(self.T, self.W, start=D0, n_weeks=3, reserve=reserve, time_limit=10, workers=2)
        self.assertIn(info["status"], ("OPTIMAL", "FEASIBLE"))
        self.assertFalse(a.task_id.duplicated().any())
        return a.groupby("week").size().to_dict()

    def test_reserve_reduces_weekly_capacity(self):
        self.assertTrue(all(n <= 2 for n in self.per_week(0.0).values()))   # floor(180/80) = 2
        self.assertTrue(all(n <= 1 for n in self.per_week(0.2).values()))   # floor(144/80) = 1

    def test_monthly_respects_earliest_start(self):
        T = T_([("L", "ENGG", "S0", 1, 60, -20, .9)])                       # due day 20 -> earliest day 6
        a, _ = h.solve_monthly(T, self.W, start=D0, n_weeks=3, time_limit=10, workers=2)
        self.assertGreaterEqual(int(a.week.iloc[0]), 0)
        T2 = T_([("L", "ENGG", "S0", 1, 60, -30, .9)])                      # earliest day 16 -> only week 3
        a2, _ = h.solve_monthly(T2, self.W, start=D0, n_weeks=3, time_limit=10, workers=2)
        self.assertEqual(a2.week.tolist(), [2])

    def test_carry_over_loses_nothing(self):
        # monthly plan puts everything in week 0, which has room for only 2 tasks in one block
        assign = pd.DataFrame(dict(task_id=self.T.task_id, week=0))
        plan, blocks, summ, done = h.run_horizon(self.T, self.W, assign, start=D0, n_weeks=3,
                                                 time_limit=10, workers=2)
        self.assertEqual(summ.loc[0, "pool"], 6)
        self.assertGreater(summ.loc[0, "carried_over"], 0)
        self.assertEqual(summ.scheduled.sum(), len(done))
        self.assertEqual(len(done), 6)                                      # carried work is done in weeks 2-3
        self.assertEqual(summ.violations.sum(), 0)
        real = h.realization(assign, plan)
        self.assertEqual(set(real.status), {"ON_TIME", "SLIPPED"})


@unittest.skipUnless((o.DATA / "tasks_common_schema.csv").exists()
                     and (o.OUT / "prioritized_tasks.csv").exists(), "run data_gen.py and priority.py first")
class TestRealData(unittest.TestCase):
    def test_week1_parallel_plan(self):
        raw, windows = o.load()
        T = h.prepare_tasks(raw)
        plan, blocks, info, _, _ = h.solve_weekly(T, windows, time_limit=20, workers=4)
        self.assertEqual(h.validate_parallel(plan, windows, T), [])
        self.assertEqual(info["objective"], o.net_value(plan, blocks, o.START))
        self.assertTrue((plan.end_min <= plan.win_end).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
