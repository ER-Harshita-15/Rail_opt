"""
Tests for Step 6 (horizons.solve_weekly stability options, replan.py, arrivals.py).
Run:  python -m unittest test_replan -v
"""
import unittest

import pandas as pd

import arrivals as ar
import horizons as h
import metrics as m
import optimizer as o
import replan as rp
import whatif as wi
from test_horizons import D0, T_, W_, hand_plan, solve


def with_week(plan):
    return m.with_week(plan, D0)


class TestSolverStability(unittest.TestCase):
    def setUp(self):
        self.W = W_([("A", "S0", 0, 10, 13, True, 6), ("B", "S0", 1, 10, 13, True, 2)])   # B is much cheaper
        self.T = T_([("E1", "ENGG", "S0", 1, 60, 5, .8)])

    def test_without_stability_arguments_it_is_the_step4_model(self):
        plan, blocks, info, _, _ = solve(self.T, self.W)
        self.assertEqual(info["objective"], o.net_value(plan, blocks, D0))
        self.assertNotIn("stability_penalty", info)

    def test_move_penalty_keeps_a_task_in_its_prior_block(self):
        free, _, _, _, _ = solve(self.T, self.W, prior={"E1": "A"}, move_penalty=10, drop_penalty=100)
        self.assertEqual(free.window_id.tolist(), ["B"])                    # cheaper block is worth a 10-point move
        held, _, info, _, _ = solve(self.T, self.W, prior={"E1": "A"}, move_penalty=500, drop_penalty=100)
        self.assertEqual(held.window_id.tolist(), ["A"])
        self.assertEqual((info["moved"], info["dropped"]), (0, 0))
        _, _, i2, _, _ = solve(self.T, self.W, prior={"E1": "A"}, move_penalty=10, drop_penalty=100)
        self.assertEqual(i2["objective"], i2["net_value"] - i2["stability_penalty"])
        self.assertEqual((i2["moved"], i2["stability_penalty"]), (1, 10))

    def test_drop_penalty_keeps_marginal_work_scheduled(self):
        W = W_([("A", "S0", 0, 10, 13, True, 15)])                          # block cost 220
        T = T_([("L", "ENGG", "S0", 1, 60, 5, .2)])                         # value 200: normally declined
        none, _, _, _, _ = solve(T, W)
        self.assertEqual(len(none), 0)
        kept, _, info, _, _ = solve(T, W, prior={"L": "A"}, move_penalty=0, drop_penalty=100)
        self.assertEqual(len(kept), 1)                                      # -20 net beats a 100-point drop
        self.assertEqual(info["objective"], info["net_value"] - info["stability_penalty"])

    def test_dropped_work_is_counted(self):
        W = W_([("A", "S0", 0, 10, 13, True, 15)])
        T = T_([("L", "ENGG", "S0", 1, 60, 5, .2)])
        plan, _, info, _, _ = solve(T, W, prior={"L": "A"}, move_penalty=5, drop_penalty=10)
        self.assertEqual(len(plan), 0)
        self.assertEqual((info["dropped"], info["stability_penalty"]), (1, 10))
        self.assertEqual(info["objective"], info["net_value"] - 10)

    def test_freeze_forces_committed_work_to_stay(self):
        W = W_([("A", "S0", 0, 10, 13, True, 15), ("B", "S0", 1, 10, 13, True, 2)])
        T = T_([("E1", "ENGG", "S0", 1, 60, 5, .3)])
        free, _, _, _, _ = solve(T, W)
        self.assertEqual(free.window_id.tolist(), ["B"])
        frozen, _, info, _, _ = solve(T, W, freeze={"E1": "A"})
        self.assertEqual(frozen.window_id.tolist(), ["A"])
        self.assertEqual(info["freeze_missing"], [])
        _, _, i2, _, _ = solve(T, W, freeze={"E1": "ZZ"})
        self.assertEqual(i2["freeze_missing"], ["E1"])                      # unknown window is reported

    def test_latest_deadline_for_urgent_tasks(self):
        W = W_([("A", "S0", 0, 10, 13, True, 15), ("B", "S0", 3, 10, 13, True, 2)])
        T = T_([("EMG", "ENGG", "S0", 1, 60, 1, .95, 5)])
        late, _, _, _, _ = solve(T, W, force_ids=["EMG"])
        self.assertEqual(late.window_id.tolist(), ["B"])                    # cheaper block three days later
        soon, _, _, _, _ = solve(T, W, force_ids=["EMG"], latest={"EMG": 1})
        self.assertEqual(soon.window_id.tolist(), ["A"])
        _, _, info, _, _ = solve(T, W, force_ids=["EMG"], latest={"EMG": -1})   # nothing meets the deadline
        self.assertEqual(info["status"], "INFEASIBLE_FORCED")


class SmallWorld(unittest.TestCase):
    """6 days x 2 sections, one 10-13 window per day; E0..E5 planned one per day."""

    def setUp(self):
        self.W = W_([(f"W{s}_{d}", f"S{s}", d, 10, 13, True) for s in range(2) for d in range(6)])
        self.T = T_([(f"E{i}", "ENGG", f"S{i % 2}", 1.0 + 3 * i, 60, 8, .7) for i in range(6)])
        layout = [("E0", "W0_0"), ("E1", "W1_1"), ("E2", "W0_2"), ("E3", "W1_3"), ("E4", "W0_4"), ("E5", "W1_5")]
        plan, _ = hand_plan(layout, self.T, self.W)
        self.plan = with_week(plan)
        self.kw = dict(start=D0, lock_days=1, lookahead_days=3, time_limit=10, workers=2)

    def emergency(self, section="S0", dur=60, task_id="EMG-001"):
        return wi.make_emergency_task(self.T, "ENGG", section, 4.0, dur, task_id=task_id)

    def rows(self, plan):
        return set(zip(plan.task_id, plan.window_id))


class TestSplitAndRebuild(SmallWorld):
    def test_split_partitions_the_plan(self):
        parts = rp.split_plan(self.plan, 2, lock_days=1, lookahead_days=3, start=D0)
        self.assertEqual(set(parts["executed"].task_id), {"E0", "E1"})
        self.assertEqual(set(parts["frozen"].task_id), {"E2"})
        self.assertEqual(set(parts["open"].task_id), {"E3", "E4"})
        self.assertEqual(set(parts["future"].task_id), {"E5"})
        self.assertEqual(sum(len(p) for p in parts.values()), len(self.plan))

    def test_rebuild_is_consistent(self):
        plan, blocks = rp.rebuild(self.plan[rp.ROW_COLS], self.T, self.W, D0)
        self.assertEqual(self.rows(plan), self.rows(self.plan))
        self.assertEqual(len(blocks), 6)
        self.assertIn("week", plan.columns)
        self.assertEqual(h.validate_parallel(plan, self.W, self.T, start=D0, horizon_days=6, origin=D0), [])


class TestStableReplan(SmallWorld):
    def test_history_commitments_and_future_stay_put(self):
        e = self.emergency()
        T2 = pd.concat([self.T, e], ignore_index=True)
        r = rp.replan(T2, self.W, self.plan, 2, new_ids=["EMG-001"], force_ids=["EMG-001"], mode="stable", **self.kw)
        self.assertTrue(r["feasible"])
        new = self.rows(r["plan"])
        for kept in ("E0", "E1", "E2", "E5"):                               # executed, frozen, future
            self.assertIn((kept, dict(zip(self.plan.task_id, self.plan.window_id))[kept]), new)
        self.assertEqual(r["diff"]["broken_commitments"], [])
        self.assertIn("EMG-001", r["diff"]["new_scheduled"])
        self.assertEqual(r["validation_errors"], [])
        self.assertEqual(h.validate_parallel(r["plan"], self.W, T2, start=D0, horizon_days=6, origin=D0), [])
        emg_day = int((r["plan"][r["plan"].task_id == "EMG-001"].date.min() - D0).days)
        self.assertLessEqual(emg_day, 2 + rp.EMERGENCY_MAX_DELAY_DAYS)      # R7 deadline

    def test_diff_counts_match_a_manual_comparison(self):
        e = self.emergency()
        T2 = pd.concat([self.T, e], ignore_index=True)
        r = rp.replan(T2, self.W, self.plan, 2, new_ids=["EMG-001"], force_ids=["EMG-001"], mode="stable", **self.kw)
        old, new = dict(zip(self.plan.task_id, self.plan.window_id)), dict(zip(r["plan"].task_id, r["plan"].window_id))
        moved = sorted(t for t in old if t in new and old[t] != new[t])
        self.assertEqual(r["diff"]["moved"], moved)
        self.assertEqual(r["diff"]["dropped"], sorted(t for t in old if t not in new))

    def test_unplaceable_emergency_is_reported_not_dropped_silently(self):
        e = self.emergency(dur=170)                                         # 20 + 170 > 180
        T2 = pd.concat([self.T, e], ignore_index=True)
        r = rp.replan(T2, self.W, self.plan, 2, new_ids=["EMG-001"], force_ids=["EMG-001"], mode="stable", **self.kw)
        self.assertEqual(r["emergency_unplaced"], ["EMG-001"])
        self.assertNotIn("EMG-001", set(r["plan"].task_id))
        self.assertEqual(r["diff"]["broken_commitments"], [])
        self.assertEqual(r["diff"]["dropped"], [])                          # nothing falls out of the plan
        kept = self.rows(self.plan[self.plan.task_id.isin(["E0", "E1", "E2", "E5"])])
        self.assertTrue(kept <= self.rows(r["plan"]))                       # history, frozen and future untouched

    def test_deadline_is_relaxed_when_no_window_meets_it(self):
        W = self.W[~self.W.window_id.isin(["W0_2", "W0_3"])]                # S0 has no window on days 2-3
        T2 = pd.concat([self.T, self.emergency()], ignore_index=True)
        plan = with_week(hand_plan([("E0", "W0_0"), ("E1", "W1_1"), ("E3", "W1_3"), ("E4", "W0_4")], self.T, W)[0])
        r = rp.replan(T2, W, plan, 2, new_ids=["EMG-001"], force_ids=["EMG-001"], mode="stable", **self.kw)
        self.assertTrue(r["deadline_relaxed"])
        self.assertIn("EMG-001", set(r["plan"].task_id))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            rp.replan(self.T, self.W, self.plan, 2, mode="bogus", **self.kw)


class TestFullVsStable(unittest.TestCase):
    def setUp(self):
        # X is committed to an expensive block on day 2; a cheap block exists on day 3
        self.W = W_([("F", "S0", 2, 10, 13, True, 15), ("C", "S0", 3, 10, 13, True, 2), ("Y", "S1", 3, 10, 13, True)])
        self.T = T_([("X", "ENGG", "S0", 1, 60, 5, .5), ("NEW", "ENGG", "S1", 2, 60, 5, .5)])
        self.plan = with_week(hand_plan([("X", "F")], self.T.iloc[:1], self.W)[0])
        self.kw = dict(start=D0, lock_days=1, lookahead_days=3, time_limit=10, workers=2)

    def test_full_reoptimisation_breaks_a_commitment_stable_does_not(self):
        full = rp.replan(self.T, self.W, self.plan, 2, new_ids=["NEW"], mode="full", **self.kw)
        self.assertEqual(full["diff"]["moved"], ["X"])                      # cheaper block is worth the move
        self.assertEqual(full["diff"]["broken_commitments"], ["X"])
        stable = rp.replan(self.T, self.W, self.plan, 2, new_ids=["NEW"], mode="stable", **self.kw)
        self.assertEqual(stable["diff"]["moved"], [])
        self.assertEqual(stable["diff"]["broken_commitments"], [])
        self.assertIn("NEW", set(stable["plan"].task_id))

    def test_prices_alone_also_protect_work_outside_the_lock(self):
        kw = {**self.kw, "lock_days": 0}                                    # nothing frozen
        cheap = rp.replan(self.T, self.W, self.plan, 2, new_ids=["NEW"], mode="stable", move_penalty=10, **kw)
        dear = rp.replan(self.T, self.W, self.plan, 2, new_ids=["NEW"], mode="stable", move_penalty=500, **kw)
        self.assertEqual(cheap["diff"]["moved"], ["X"])
        self.assertEqual(dear["diff"]["moved"], [])


class TestPatchInsert(SmallWorld):
    def test_new_task_joins_an_existing_block_without_moving_anything(self):
        n = self.emergency(task_id="N1")
        T2 = pd.concat([self.T, n], ignore_index=True)
        r = rp.patch_insert(T2, self.W, self.plan, 2, ["N1"], start=D0)
        self.assertEqual(r["emergency_unplaced"], [])
        self.assertEqual(r["diff"]["moved"], [])
        self.assertEqual(r["diff"]["dropped"], [])
        self.assertTrue(self.rows(self.plan) <= self.rows(r["plan"]))
        self.assertEqual(len(r["blocks"]), len(self.plan.window_id.unique()))   # joined an existing S0 block
        self.assertEqual(r["validation_errors"], [])

    def test_unplaceable_task_is_reported(self):
        n = self.emergency(dur=170, task_id="N1")
        T2 = pd.concat([self.T, n], ignore_index=True)
        r = rp.patch_insert(T2, self.W, self.plan, 2, ["N1"], start=D0)
        self.assertEqual(r["emergency_unplaced"], ["N1"])
        self.assertEqual(self.rows(r["plan"]), self.rows(self.plan))

    def test_safety_rule_sends_a_close_traction_task_to_another_block(self):
        W = W_([("D0", "S0", 0, 10, 13, True), ("D1", "S0", 1, 10, 13, True)])
        T = T_([("ST", "S&T", "S0", 5.0, 60, 5, .8)])
        plan = with_week(hand_plan([("ST", "D0")], T, W)[0])
        like = T_([("ST", "S&T", "S0", 5.0, 60, 5, .8), ("TR", "TRD", "S0", 5.4, 60, 5, .9)])
        T2 = like
        r = rp.patch_insert(T2, W, plan, 0, ["TR"], start=D0)
        self.assertEqual(r["plan"][r["plan"].task_id == "TR"].window_id.tolist(), ["D1"])
        self.assertEqual(r["validation_errors"], [])

    def test_gang_pool_refuses_a_third_tower_wagon(self):
        W = W_([(f"W{k}", f"S{k}", 0, 10, 13, True) for k in range(3)])
        T = T_([(f"P{k}", "TRD", f"S{k}", 3.0, 60, 5, .5) for k in range(3)])
        plan = with_week(hand_plan([("P0", "W0"), ("P1", "W1")], T.iloc[:2], W)[0])
        r = rp.patch_insert(T, W, plan, 0, ["P2"], start=D0)
        self.assertEqual(r["emergency_unplaced"], ["P2"])


class TestSimulation(SmallWorld):
    def events(self):
        e2 = self.emergency(task_id="EMG-001")
        e3 = pd.concat([wi.make_emergency_task(self.T, "ENGG", "S1", 5.0, 60, task_id=f"NEW-00{i}", priority_score=.6,
                                               tier="HIGH", severity=3, overdue_days=-4) for i in (1, 2)],
                       ignore_index=True)
        return [dict(day=2, kind="emergency", tasks=e2), dict(day=3, kind="routine", tasks=e3)]

    def run_strategy(self, strategy, lock_days=rp.LOCK_DAYS):
        if strategy == "patch":
            return rp.simulate(self.T, self.W, self.plan, self.events(), strategy, start=D0, workers=1)
        return rp.simulate(self.T, self.W, self.plan, self.events(), strategy, start=D0, time_limit=10, workers=1,
                           lock_days=lock_days, lookahead_days=4)

    def test_totals_equal_the_sum_of_events_and_final_plan_is_valid(self):
        for strategy in ("stable", "full", "patch"):
            ev, s, plan, blocks, T_all = self.run_strategy(strategy)
            self.assertEqual(len(ev), 2)
            self.assertEqual(s["total_moved"], ev.moved.sum())
            self.assertEqual(s["total_blocks_changed"], ev.blocks_changed.sum())
            self.assertEqual(s["final_violations"], 0, strategy)
            self.assertTrue({"EMG-001", "NEW-001", "NEW-002"} <= set(plan.task_id), strategy)
            self.assertEqual(len(T_all), len(self.T) + 3)

    def test_patch_never_moves_anything(self):
        _, s, _, _, _ = self.run_strategy("patch")
        self.assertEqual((s["total_moved"], s["total_dropped"], s["broken_commitments"]), (0, 0, 0))

    def test_stable_strategy_never_breaks_commitments(self):
        _, s, _, _, _ = self.run_strategy("stable")
        self.assertEqual(s["broken_commitments"], 0)

    def test_single_worker_runs_are_reproducible(self):
        a = self.run_strategy("stable")[1]
        b = self.run_strategy("stable")[1]
        self.assertEqual(a, b)

    def test_emergency_response_is_measured_from_arrival(self):
        ev, s, plan, _, _ = self.run_strategy("stable")
        day = int((plan[plan.task_id == "EMG-001"].date.min() - D0).days)
        self.assertEqual(ev.emergency_response_days.dropna().iloc[0], day - 2)
        self.assertEqual(s["mean_emergency_response_days"], day - 2)


_HAS_DATA = (m.OUT / "block_plan_4w.csv").exists() and (o.DATA / "tasks_common_schema.csv").exists()


@unittest.skipUnless(_HAS_DATA, "run data_gen.py, priority.py and horizons.py first")
class TestArrivalsAndRealScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T, cls.W = m.load_inputs()
        cls.raw = pd.read_csv(o.DATA / "tasks_common_schema.csv")
        cls.sections = pd.read_csv(o.DATA / "sections.csv")
        cls.plan = m.with_week(m.load_results()["plan"])
        cls.events = ar.generate_events(cls.T, cls.raw, cls.sections, spec=[(2, "emergency", "S&T"), (4, "routine", 2)])

    def test_events_are_deterministic_and_well_formed(self):
        again = ar.generate_events(self.T, self.raw, self.sections, spec=[(2, "emergency", "S&T"), (4, "routine", 2)])
        for a, b in zip(self.events, again):
            pd.testing.assert_frame_equal(a["tasks"], b["tasks"])
        emg, routine = self.events[0]["tasks"].iloc[0], self.events[1]["tasks"]
        self.assertEqual((emg.tier, emg.priority_score, emg.severity), ("CRITICAL", 0.95, 5))
        self.assertEqual(list(routine.columns), list(self.T.columns))
        self.assertTrue(routine.tier.isin(ar.TIER_LABELS).all())
        self.assertTrue((routine.duration_min <= 150).all())
        due = pd.to_datetime(routine.due_date)
        self.assertTrue(((due - o.START).dt.days == -routine.overdue_days).all())
        self.assertFalse(set(routine.task_id) & set(self.T.task_id))

    def test_adding_new_tasks_does_not_change_the_scores_of_the_original_tasks(self):
        new = pd.DataFrame([ar._raw_row("NEW-X", "ENGG", self.sections.section_id[0], 3.0, "Weld repair", 3, 90, 15, 1)])
        scored = ar.score_routine(self.raw, new)
        saved = pd.read_csv(o.OUT / "prioritized_tasks.csv").set_index("task_id").priority_score
        common = saved.index.intersection(scored.index)
        self.assertTrue(((scored.priority_score[common] - saved[common]).abs() < 0.0015).all())

    def test_stable_replan_on_the_real_plan(self):
        e = self.events[0]
        T2 = pd.concat([self.T, e["tasks"]], ignore_index=True)
        ids = list(e["tasks"].task_id)
        r = rp.replan(T2, self.W, self.plan, e["day"], new_ids=ids, force_ids=ids, mode="stable", time_limit=20)
        self.assertEqual(r["validation_errors"], [])
        self.assertEqual(r["diff"]["broken_commitments"], [])
        self.assertEqual(r["diff"]["dropped"], [])
        self.assertIn(ids[0], r["diff"]["new_scheduled"])
        self.assertLessEqual(len(r["diff"]["moved"]), 10)                   # small change, not a reshuffle


if __name__ == "__main__":
    unittest.main(verbosity=2)
