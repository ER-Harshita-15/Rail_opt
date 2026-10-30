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
    """rows: (task_id, dept, section, km, duration, overdue_days, score[, severity[, tier]])"""
    out = []
    for r in rows:
        tid, dept, sec, km, dur, ov, sc = r[:7]
        out.append(dict(task_id=tid, dept=dept, section_id=sec, km=km, task="job",
                        tier=r[8] if len(r) > 8 else "HIGH",
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
        T2 = T_([("L", "ENGG", "S0", 1, 60, -30, .9)])                      # earliest day 16
        a2, _ = h.solve_monthly(T2, self.W, start=D0, n_weeks=3, time_limit=10, workers=2, model="simple")
        self.assertEqual(a2.week.tolist(), [2])                             # simple: week 3 ends after day 16
        a3, _ = h.solve_monthly(T2, self.W, start=D0, n_weeks=3, time_limit=10, workers=2)
        self.assertEqual(a3.week.tolist(), [])                              # blocks: week 3's only window is day 14
        W2 = pd.concat([self.W, W_([("W3", "S0", 17, 10, 13, True)])], ignore_index=True)
        a4, _ = h.solve_monthly(T2, W2, start=D0, n_weeks=3, time_limit=10, workers=2)
        self.assertEqual(a4.week.tolist(), [2])                             # a usable window on day 17 exists

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


class TestForcedDiagnosis(unittest.TestCase):
    def setUp(self):
        self.W = W_([("A", "S0", 0, 10, 13, True, 15)])                    # block cost 60 + 150 + 10 = 220
        self.T = T_([("L", "ENGG", "S0", 2, 60, 5, .2)])                    # value 200 < 220

    def test_forcing_places_the_task_and_shows_its_cost(self):
        p0, _, i0, _, _ = solve(self.T, self.W)
        p1, _, i1, _, _ = solve(self.T, self.W, force_ids=["L"])
        self.assertEqual(len(p0), 0)
        self.assertEqual(len(p1), 1)
        self.assertEqual(i1["objective"] - i0["objective"], -20)

    def test_forcing_without_any_eligible_window_is_flagged(self):
        T = T_([("L", "ENGG", "S0", 2, 170, 5, .9)])                        # 20 + 170 > 180
        plan, _, info, _, _ = solve(T, self.W, force_ids=["L"])
        self.assertEqual(info["status"], "INFEASIBLE_FORCED")
        self.assertEqual(info["forced_missing"], ["L"])

    def test_explain_reasons(self):
        ex = h.explain_unscheduled(self.T, self.W, pd.DataFrame(), ["L"], start=D0, n_weeks=1,
                                   time_limit=10, workers=2).iloc[0]
        self.assertEqual(ex.reason, "DECLINED_NET_NEGATIVE")
        self.assertEqual(ex.delta_if_forced, -20)
        self.assertTrue(ex.proven)
        good = T_([("G", "ENGG", "S0", 2, 60, 5, .9)])                      # worth a block, but excluded by the pool
        ex2 = h.explain_unscheduled(good, self.W, pd.DataFrame(), ["G"], start=D0, n_weeks=1,
                                    time_limit=10, workers=2).iloc[0]
        self.assertEqual(ex2.reason, "WEEKLY_POOL_EFFECT")
        self.assertGreater(ex2.delta_if_forced, 0)
        long_ = T_([("X", "ENGG", "S0", 2, 170, 5, .9)])
        ex3 = h.explain_unscheduled(long_, self.W, pd.DataFrame(), ["X"], start=D0, n_weeks=1,
                                    time_limit=10, workers=2).iloc[0]
        self.assertEqual(ex3.reason, "NO_ELIGIBLE_WINDOW")


class TestFinalUnscheduledReport(unittest.TestCase):
    def test_every_task_is_scheduled_or_listed_with_a_reason(self):
        W = W_([("A", "S0", 0, 10, 13, True, 15), ("B", "S1", 0, 10, 13, True), ("C", "S2", 0, 10, 13, True, 15)])
        T = T_([("LOW", "ENGG", "S0", 2, 60, 5, .2),                        # worth less than its block
                ("GOOD", "ENGG", "S1", 2, 60, 5, .9),
                ("LONG", "ENGG", "S1", 9, 170, 5, .9),                      # 190 > 180 -> never placeable
                ("LATE", "ENGG", "S2", 9, 60, -10, .2),                     # due day 10, not worth a block
                ("FAR", "ENGG", "S1", 15, 60, -40, .9)])                    # earliest day 26 > horizon
        plan, _, _, _, _ = solve(T, W)
        plan = plan.assign(week=1)
        un = h.final_unscheduled(T, W, plan, start=D0, n_weeks=1, time_limit=10, workers=2)
        self.assertEqual(set(plan.task_id) | set(un.task_id), set(T.task_id))
        self.assertFalse(set(plan.task_id) & set(un.task_id))
        self.assertEqual(len(plan.task_id.unique()) + len(un), len(T))
        r = un.set_index("task_id")
        self.assertEqual(r.reason["LONG"], "TOO_LONG")
        self.assertEqual(r.group["LONG"], "NEVER_PLACEABLE")
        self.assertIn("extended block", r.recommended_action["LONG"])
        self.assertEqual(r.reason["LOW"], "DECLINED_NET_NEGATIVE")
        self.assertIn("REVIEW", r.recommended_action["LOW"])                # due inside the horizon -> flagged
        self.assertEqual(r.reason["LATE"], "DECLINED_NET_NEGATIVE")
        self.assertTrue(r.due_after_horizon["LATE"])
        self.assertIn("next planning cycle", r.recommended_action["LATE"])
        self.assertEqual(r.reason["FAR"], "NOT_YET_ELIGIBLE")
        self.assertIn("earliest start day 26", r.recommended_action["FAR"])

    def test_no_explain_still_lists_tasks(self):
        W = W_([("A", "S0", 0, 10, 13, True, 15)])
        T = T_([("LOW", "ENGG", "S0", 2, 60, 5, .2)])
        un = h.final_unscheduled(T, W, pd.DataFrame(), start=D0, n_weeks=1, explain=False)
        self.assertEqual(un.reason.tolist(), ["NOT_EXPLAINED"])


class TestMonthlyMatchesWeeklyFeasibility(unittest.TestCase):
    def test_blocks_model_does_not_share_one_visit_between_tasks_that_each_need_a_block(self):
        # 4 days x 1 window, each block costs 220; 4 tasks of 120 min: one task fills a block's crew time
        W = W_([(f"W{d}", "S0", d, 10, 13, True, 15) for d in range(4)])
        T = T_([(f"E{i}", "ENGG", "S0", 2.0 * i, 120, 5, .2) for i in range(4)])
        kw = dict(start=D0, n_weeks=1, reserve=0.0, time_limit=10, workers=2)
        simple, _ = h.solve_monthly(T, W, model="simple", **kw)
        blocks, _ = h.solve_monthly(T, W, model="blocks", **kw)
        self.assertEqual(len(simple), 4)                                    # the old model: one shared visit
        self.assertEqual(len(blocks), 0)                                    # needs 4 blocks, worth less than they cost
        plan, _, _, _, _ = solve(T, W, task_ids=list(simple.task_id))
        self.assertEqual(len(plan), 0)                                      # the weekly model agrees with 'blocks'

    def test_blocks_model_only_offers_weeks_with_a_usable_window(self):
        W = W_([(f"W{d}", "S0", d, 10, 13, True) for d in range(4)])        # windows on days 0-3 only
        T = T_([("L", "ENGG", "S0", 2, 60, -19, .9)])                       # due day 19 -> earliest day 5
        kw = dict(start=D0, n_weeks=1, reserve=0.0, time_limit=10, workers=2)
        simple, _ = h.solve_monthly(T, W, model="simple", **kw)
        blocks, _ = h.solve_monthly(T, W, model="blocks", **kw)
        self.assertEqual(len(simple), 1)
        self.assertEqual(len(blocks), 0)
        plan, _, _, _, _ = solve(T, W, task_ids=["L"], horizon_days=7)
        self.assertEqual(len(plan), 0)

    def test_month_value_formulas(self):
        t = T_([("N", "ENGG", "S0", 1, 60, -10, .5, 3)]).iloc[0]            # due day 10, severity 3 -> coef 15
        self.assertEqual(h.month_value(t, 6, True), 500 - (15 * 6) // 2)
        self.assertEqual(h.month_value(t, 12, True), 500 - (15 * (10 + 3 * 2)) // 2)      # 2 days late: x3
        self.assertEqual(h.month_value(t, 12, False), o.assignment_value(t.priority_score, t.severity, t.overdue_days, 12))
        ov = T_([("O", "ENGG", "S0", 1, 60, 4, .5, 3)]).iloc[0]
        self.assertEqual(h.month_value(ov, 7, True), o.assignment_value(ov.priority_score, ov.severity, ov.overdue_days, 7))


class TestDeferralAndBalancing(unittest.TestCase):
    def setUp(self):
        # sections use different hours so the network crew pool (3 track gangs) does not bind
        self.W = W_([(f"W{s}_{d}", f"S{s}", d, 1 + s, 4 + s, True) for s in range(8) for d in (0, 7, 14)])
        rows = [(f"D{s}{j}", "ENGG", f"S{s}", 2.0 + 6 * j, 60, -25, .5, 3, "LOW") for s in range(6) for j in range(2)]
        rows += [("O0", "ENGG", "S6", 3, 60, 10, .5, 3, "MEDIUM"), ("O1", "ENGG", "S7", 3, 60, 10, .5, 3, "MEDIUM")]
        self.T = T_(rows, early_days=30)                                    # deferrable tasks: earliest day -5
        self.kw = dict(start=D0, n_weeks=3, reserve=0.0, time_limit=15, workers=2)

    def test_balancing_reduces_week1_load_without_delaying_overdue_or_making_tasks_late(self):
        base, _ = h.solve_monthly(self.T, self.W, defer=False, balance=0, **self.kw)
        bal, info = h.solve_monthly(self.T, self.W, defer=True, balance=50, **self.kw)
        self.assertEqual(len(base), 14)
        self.assertTrue((base.week == 0).all())                             # without it everything piles into week 1
        self.assertEqual(len(bal), 14)
        self.assertLess((bal.week == 0).sum(), (base.week == 0).sum())      # reduced concentration
        self.assertTrue((bal[bal.task_id.isin(["O0", "O1"])].week == 0).all())            # overdue stay in week 1
        deferred = bal[bal.task_id.str.startswith("D")]
        self.assertTrue(((7 * deferred.week + 6) <= 25).all())              # never past the due date

    def test_balance_cap_controls_how_much_load_is_tolerated(self):
        tight, _ = h.solve_monthly(self.T, self.W, defer=True, balance=50, balance_cap=1.25, **self.kw)
        loose, _ = h.solve_monthly(self.T, self.W, defer=True, balance=50, balance_cap=3.0, **self.kw)
        self.assertEqual((loose.week == 0).sum(), 14)                       # cap 3.0 x average: nothing to smooth
        self.assertLess((tight.week == 0).sum(), (loose.week == 0).sum())

    def test_zero_balance_with_defer_never_moves_critical_work(self):
        T = self.T.copy()
        T.loc[T.task_id.isin(["D00", "D01"]), "tier"] = "CRITICAL"
        a, _ = h.solve_monthly(T, self.W, defer=True, balance=50, **self.kw)
        self.assertTrue((a[a.task_id.isin(["D00", "D01"])].week == 0).all())

    def test_first_week_restricts_assignment(self):
        a, _ = h.solve_monthly(self.T, self.W, first_week=2, **{**self.kw})
        self.assertTrue((a.week >= 2).all())
        sub, _ = h.solve_monthly(self.T, self.W, task_ids=["D00", "D01"], **self.kw)
        self.assertEqual(set(sub.task_id), {"D00", "D01"})


class TestReplanAndRealization(unittest.TestCase):
    def setUp(self):
        self.W = W_([("W0", "S0", 0, 10, 13, True), ("W1", "S0", 7, 10, 13, True), ("W2", "S0", 14, 10, 13, True)])
        self.T = T_([(f"E{i}", "ENGG", "S0", 2.0 * i, 60, 30, .8) for i in range(6)])

    def test_replan_reassigns_unplaced_work_and_loses_nothing(self):
        assign = pd.DataFrame(dict(task_id=self.T.task_id, week=0))          # deliberately over-full week 1
        kw = dict(model="blocks", reserve=0.0)
        plan, blocks, summ, done = h.run_horizon(self.T, self.W, assign, start=D0, n_weeks=3, time_limit=10,
                                                 workers=2, replan=kw)
        self.assertEqual(len(summ.attrs["monthly_log"]), 3)
        self.assertEqual(len(done), 6)
        self.assertEqual(plan.task_id.nunique(), len(plan))                  # no task scheduled twice
        self.assertEqual(summ.violations.sum(), 0)
        self.assertGreater(summ.loc[0, "carried_over"], 0)                   # week 1 could not take all six
        later = summ.attrs["monthly_log"][1]
        self.assertTrue((later.week >= 1).all())
        self.assertFalse(set(later.task_id) & set(plan[plan.week == 1].task_id))   # week-1 work is not re-planned

    def test_realization_statuses_and_summary(self):
        assign = pd.DataFrame(dict(task_id=["A", "B", "C", "D"], week=[0, 1, 1, 2]))
        plan = pd.DataFrame(dict(task_id=["A", "B", "D", "Z"], week=[1, 1, 4, 2]))
        r = h.realization(assign, plan).set_index("task_id").status
        self.assertEqual(r.to_dict(), {"A": "ON_TIME", "B": "PULLED_FORWARD", "C": "NOT_SCHEDULED", "D": "SLIPPED"})
        s = h.realization_summary(assign, plan)
        self.assertEqual((s["on_time"], s["pulled_forward"], s["not_scheduled"], s["slipped"]), (1, 1, 1, 1))
        self.assertEqual(s["added_after_initial_plan"], 1)                   # Z was never in the monthly plan
        self.assertEqual(s["realization_pct"], 25.0)


class TestDueBackfill(unittest.TestCase):
    def setUp(self):
        self.W = W_([("A", "S0", 0, 10, 13, True), ("B", "S1", 8, 10, 13, True)])
        self.T = T_([("DUE", "ENGG", "S0", 2, 60, -3, .8),                  # due day 3: due inside week 1
                     ("LATER", "ENGG", "S1", 2, 60, -12, .8)], early_days=30)   # due day 12: due in week 2
        self.empty = pd.DataFrame(dict(task_id=[], week=[]))                # a monthly plan that skipped both

    def run_it(self, backfill):
        return h.run_horizon(self.T, self.W, self.empty, start=D0, n_weeks=2, time_limit=10, workers=2,
                             backfill=backfill)

    def test_without_backfill_skipped_work_is_never_considered(self):
        plan, _, summ, done = self.run_it(False)
        self.assertEqual(done, set())
        self.assertEqual(summ.backfilled.sum(), 0)

    def test_backfill_offers_only_tasks_due_by_the_end_of_the_week(self):
        plan, _, summ, done = self.run_it(True)
        self.assertEqual(done, {"DUE", "LATER"})
        self.assertEqual(summ.backfilled.tolist(), [1, 1])                   # DUE in week 1, LATER only in week 2
        self.assertEqual(plan.set_index("task_id").week.to_dict(), {"DUE": 1, "LATER": 2})
        self.assertEqual(summ.violations.sum(), 0)

    def test_backfill_still_lets_the_objective_decline_unprofitable_work(self):
        W = W_([("A", "S0", 0, 10, 13, True, 15)])                          # block cost 220
        T = T_([("LOW", "ENGG", "S0", 2, 60, -3, .2)], early_days=30)       # value 200 < 220
        plan, _, summ, done = h.run_horizon(T, W, self.empty, start=D0, n_weeks=1, time_limit=10, workers=2,
                                            backfill=True)
        self.assertEqual(done, set())
        self.assertEqual(summ.backfilled.tolist(), [1])


class TestDiagnosticTables(unittest.TestCase):
    def test_capacity_diagnosis_counts_supply_and_leftovers(self):
        W = W_([("A", "S0", 0, 10, 13, True), ("B", "S1", 1, 10, 13, True), ("C", "S0", 15, 10, 13, True),
                ("D", "S0", 17, 10, 13, True)])
        T = T_([("T1", "ENGG", "S0", 1, 60, 5, .8), ("T2", "ENGG", "S1", 1, 60, 5, .8),
                ("T3", "ENGG", "S0", 9, 60, -30, .8)])                      # due day 30 -> earliest day 16 (week 3)
        plan, blocks = hand_plan([("T1", "A"), ("T2", "B")], T, W)
        plan, blocks = plan.assign(week=1), blocks.assign(week=1)
        cd = h.capacity_diagnosis(T, W, plan, blocks, start=D0, n_weeks=3).set_index("week")
        self.assertEqual(cd.new_supply.tolist(), [2, 0, 1])
        self.assertEqual(cd.eligible_at_week_end.tolist(), [2, 0, 1])
        self.assertEqual(cd.scheduled.tolist(), [2, 0, 0])
        self.assertEqual(cd.left_for_later.tolist(), [0, 0, 1])
        self.assertEqual(cd.blocks.tolist(), [2, 0, 0])
        self.assertEqual(cd.window_count.tolist(), [2, 0, 2])
        self.assertEqual(cd.eligible_work_min[3], 80)                        # T3: 60 min + 20 setup
        self.assertEqual((cd.crew_work_min[1], cd.crew_setup_min[1]), (120, 40))   # 2 blocks, 1 crew each
        self.assertAlmostEqual(cd.free_pct[3], 100.0)
        self.assertEqual(cd.window_min[2], 0)                                # a week without windows is reported, not an error

    def test_workload_profile_and_priority_check(self):
        W = W_([("A", "S0", 0, 10, 13, True), ("B", "S1", 8, 10, 13, True)])
        T = T_([("T1", "ENGG", "S0", 1, 60, 5, .8, 3, "CRITICAL"), ("T2", "ENGG", "S1", 1, 90, 2, .8, 3, "MEDIUM"),
                ("T3", "ENGG", "S0", 9, 60, -3, .8, 3, "LOW")])             # T3 due day 3, done day 8 -> late
        p1, b1 = hand_plan([("T1", "A"), ("T3", "A")], T, W)
        p2, b2 = hand_plan([("T2", "B")], T, W)
        plan, blocks = pd.concat([p1.assign(week=1), p2.assign(week=2)]), pd.concat([b1.assign(week=1), b2.assign(week=2)])
        wp, prof = h.workload_profile(plan, blocks, n_weeks=2)
        self.assertEqual(wp.tasks.tolist(), [2, 1])
        self.assertEqual(prof["week1_tasks"], 2)
        self.assertAlmostEqual(float(prof["week1_share_pct"]), 66.7, places=1)
        pc = h.priority_check(plan, T, W, start=D0, horizon_days=14)
        self.assertEqual(pc["crit_high"], "1/1")
        self.assertEqual(pc["crit_high_mean_day"], 0.0)
        self.assertEqual(pc["overdue"], "2/2")
        self.assertEqual(pc["overdue_mean_day"], 4.0)                        # T1 day 0, T2 day 8
        self.assertEqual(pc["newly_late"], 0)


    def test_capacity_diagnosis_ignores_windows_before_the_earliest_start(self):
        W = W_([("C", "S0", 15, 10, 13, True)])                              # only window is day 15
        T = T_([("T3", "ENGG", "S0", 9, 60, -30, .8)])                       # earliest day 16
        cd = h.capacity_diagnosis(T, W, pd.DataFrame(), pd.DataFrame(), start=D0, n_weeks=3).set_index("week")
        self.assertEqual(cd.eligible_at_week_end.tolist(), [0, 0, 0])        # not usable in week 3
        self.assertEqual(cd.new_supply.tolist(), [0, 0, 1])                  # ...although it is 'released' in week 3

    def test_capacity_diagnosis_works_on_real_solver_output(self):
        # regression: the block table has work_min / crew_total_min (no crew_work / setup columns)
        W = W_([("A", "S0", 0, 10, 13, True), ("B", "S1", 0, 10, 13, True)])
        T = T_([("E1", "ENGG", "S0", 1, 60, 5, .8), ("S1", "S&T", "S0", 9, 60, 5, .8),
                ("E2", "ENGG", "S1", 1, 60, 5, .8)])
        plan, blocks, _, _, _ = solve(T, W, horizon_days=7)
        plan, blocks = plan.assign(week=1), blocks.assign(week=1)
        cd = h.capacity_diagnosis(T, W, plan, blocks, start=D0, n_weeks=1).iloc[0]
        self.assertEqual(cd.crew_work_min, int(plan.duration_min.sum()))
        self.assertEqual(cd.crew_work_min + cd.crew_setup_min, int(blocks.crew_total_min.sum()))
        self.assertEqual(cd.scheduled, 3)


class TestObjectiveAudit(unittest.TestCase):
    def setUp(self):
        self.W = W_([("A", "S0", 0, 10, 13, True), ("B", "S1", 1, 10, 13, True, 9)])
        self.T = T_([("E1", "ENGG", "S0", 1, 60, 5, .8), ("S1", "S&T", "S0", 9, 60, 3, .7, 4),
                     ("E2", "ENGG", "S1", 1, 90, -2, .6)])
        plan, blocks, self.info, _, _ = solve(self.T, self.W)
        self.plan, self.blocks = plan.assign(week=1), blocks.assign(week=1)
        self.summ = pd.DataFrame(dict(week=[1], solver_objective=[self.info["objective"]]))

    def test_recomputed_value_matches_the_solver_objective(self):
        a = h.objective_audit(self.plan, self.blocks, self.T, start=D0, summ=self.summ).iloc[0]
        self.assertEqual(a.recomputed_net_value, self.info["objective"])
        self.assertEqual(a.assignment_reward - a.block_cost, a.recomputed_net_value)
        self.assertTrue(a.matches)
        self.assertEqual(a.reference, "solver objective")

    def test_audit_flags_a_mismatch(self):
        bad = self.summ.assign(solver_objective=self.info["objective"] + 1)
        self.assertFalse(h.objective_audit(self.plan, self.blocks, self.T, start=D0, summ=bad).iloc[0].matches)

    def test_audit_without_summary_uses_net_value_function(self):
        a = h.objective_audit(self.plan, self.blocks, self.T, start=D0).iloc[0]
        self.assertEqual(a.reference, "net_value()")
        self.assertTrue(a.matches)

    def test_run_horizon_reports_solver_objective(self):
        assign = pd.DataFrame(dict(task_id=self.T.task_id, week=0))
        plan, blocks, summ, _ = h.run_horizon(self.T, self.W, assign, start=D0, n_weeks=1, time_limit=10, workers=2)
        self.assertIn("solver_objective", summ)
        a = h.objective_audit(plan, blocks, self.T, start=D0, summ=summ)
        self.assertTrue(a.matches.all())


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
