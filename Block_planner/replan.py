"""
Step 6 - rolling re-planning when new defects / emergencies arrive, with plan stability.

At day `now_day` the published plan is split into
    executed   blocks on days < now_day            history, never touched
    frozen     blocks on days now_day .. now_day+LOCK_DAYS-1
                                                   already communicated to crews and the control
                                                   office: every task stays in its block
    open       blocks up to now_day+LOOKAHEAD_DAYS-1   re-optimised together with the new tasks
    future     blocks beyond the look-ahead        left exactly as they are

Strategies compared on the same event sequence
    stable  re-optimise `open` + new tasks with the Step 4 weekly model, frozen tasks fixed, and a
            price on changes: MOVE_PENALTY per task moved to another block, DROP_PENALTY per task
            that falls out of the plan
    full    the same re-optimisation WITHOUT freezing or prices (what the Step 5 what-if did):
            best value, but anything can move, including committed blocks
    patch   manual-style: add the new task to the earliest block that can take it, change nothing else

ASSUMPTIONS (state them in the report)
    R1  LOCK_DAYS = 2, LOOKAHEAD_DAYS = 14, MOVE_PENALTY = 40, DROP_PENALTY = 150 are assumed values
        (a move costs about a block's worth of disruption for the crew; a drop also loses the task's
        value, which is already in the objective). Sensitivity is reported by `python replan.py`.
    R2  Re-planning happens at the START of the day an event arrives, so blocks of earlier days are
        executed. Blocks of that day count as frozen if LOCK_DAYS >= 1.
    R3  Frozen = the task stays in the same BLOCK (window). New tasks may join a frozen block if the
        hard rules allow, so the order of work inside it may change.
    R4  The stability prices only shape the re-plan. Reported net value is the Step 4 definition
        (rewards - block costs); the stability penalty is reported separately.
    R5  Tasks that were never scheduled (declined or too long) are not re-offered; only new tasks
        and tasks already planned are in the pool.
    R6  If an emergency cannot be placed it is reported (emergency_unplaced), never dropped silently.
    R7  An emergency must be scheduled no later than EMERGENCY_MAX_DELAY_DAYS (= 1) after it arrives;
        if no window meets that it is relaxed and reported (deadline_relaxed). In the scenario below
        a 1-day deadline changed no decision (responses were already <= 1 day), so it is a guarantee,
        not an improvement. A same-day deadline (0) cut the mean response from 0.67 to 0.33 days for
        24 points of net value (sensitivity table). The objective itself prices a day of delay at only
        about 10 points, far less than a block (~150), which is why an explicit deadline is needed.
    R8  broken_commitments is always measured against a 2-day reference lock (LOCK_DAYS), so runs
        with different lock settings are comparable.
"""
import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import horizons as h
import metrics as m
import optimizer as o

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
START = o.START

LOCK_DAYS = 2
LOOKAHEAD_DAYS = 14
MOVE_PENALTY = 40
DROP_PENALTY = 150
EMERGENCY_MAX_DELAY_DAYS = 1
ROW_COLS = ["window_id", "section_id", "date", "win_start", "win_end", "task_id"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def day_of(plan, start=START):
    return (pd.to_datetime(plan.date) - start).dt.days


def split_plan(plan, now_day, lock_days=LOCK_DAYS, lookahead_days=LOOKAHEAD_DAYS, start=START):
    d = day_of(plan, start)
    return dict(executed=plan[d < now_day],
                frozen=plan[(d >= now_day) & (d < now_day + lock_days)],
                open=plan[(d >= now_day + lock_days) & (d < now_day + lookahead_days)],
                future=plan[d >= now_day + lookahead_days])


def rebuild(plan_rows, T_all, windows, start=START):
    """Union of plan rows -> consistent (plan, blocks) in the Step 4 format, with a week column."""
    if plan_rows is None or len(plan_rows) == 0:
        return pd.DataFrame(), pd.DataFrame()
    plan, blocks = h.finish_plan_parallel(plan_rows[ROW_COLS].reset_index(drop=True), T_all, windows)
    return m.with_week(plan, start), m.with_week(blocks, start)


def diff_plans(old, new, now_day, lock_days, new_ids=(), start=START):
    """What changed for the work from now_day on."""
    o_f = old[day_of(old, start) >= now_day] if len(old) else old
    n_f = new[day_of(new, start) >= now_day] if len(new) else new
    om, nm = dict(zip(o_f.task_id, o_f.window_id)), dict(zip(n_f.task_id, n_f.window_id))
    moved = sorted(t for t in om if t in nm and om[t] != nm[t])
    dropped = sorted(t for t in om if t not in nm)
    d_old = day_of(o_f, start) if len(o_f) else pd.Series(dtype=int)
    frozen_ids = set(o_f[d_old < now_day + lock_days].task_id) if len(o_f) else set()
    broken = sorted(t for t in frozen_ids if t not in nm or nm[t] != om[t])
    ow = o_f.groupby("window_id").task_id.apply(frozenset) if len(o_f) else pd.Series(dtype=object)
    nw = n_f.groupby("window_id").task_id.apply(frozenset) if len(n_f) else pd.Series(dtype=object)
    wins = set(ow.index) | set(nw.index)
    changed = sum(1 for w in wins if ow.get(w, frozenset()) != nw.get(w, frozenset()))
    return dict(moved=moved, dropped=dropped, broken_commitments=broken, blocks_changed=changed,
                blocks_opened=len(set(nw.index) - set(ow.index)), blocks_closed=len(set(ow.index) - set(nw.index)),
                new_scheduled=[t for t in new_ids if t in nm])


def _from_now(df, now_day, start):
    return df[day_of(df, start) >= now_day] if len(df) else df


def _kpi_from_now(blocks, now_day, start):
    b = _from_now(blocks, now_day, start)
    return int(b.trains_affected.sum()) if len(b) else 0


# --------------------------------------------------------------------------
# strategies
# --------------------------------------------------------------------------
def replan(T_all, windows, plan, now_day, new_ids=(), force_ids=(), mode="stable", start=START,
           lock_days=LOCK_DAYS, lookahead_days=LOOKAHEAD_DAYS, move_penalty=MOVE_PENALTY,
           drop_penalty=DROP_PENALTY, emergency_max_delay=EMERGENCY_MAX_DELAY_DAYS, time_limit=20, workers=8):
    """Re-optimise the open part of the plan. mode='stable' (frozen + prices) or 'full' (neither)."""
    if mode not in ("stable", "full"):
        raise ValueError("mode must be 'stable' or 'full'")
    parts = split_plan(plan, now_day, lock_days, lookahead_days, start)
    last_day = int((windows.date.max() - start).days)
    horizon = max(1, min(lookahead_days, last_day + 1 - now_day))
    ws = start + pd.Timedelta(days=now_day)
    active = pd.concat([parts["frozen"], parts["open"]]) if len(plan) else plan
    pool = list(active.task_id) + [t for t in new_ids if t not in set(active.task_id)]
    stable = mode == "stable"
    kw = dict(horizon_days=horizon, start=ws, origin=start, task_ids=pool, time_limit=time_limit, workers=workers)
    if stable:
        kw.update(prior=dict(zip(active.task_id, active.window_id)), move_penalty=move_penalty,
                  drop_penalty=drop_penalty, freeze=dict(zip(parts["frozen"].task_id, parts["frozen"].window_id)))
    latest = ({t: now_day + int(emergency_max_delay) for t in force_ids}
              if force_ids and emergency_max_delay is not None else None)
    relaxed, unplaced = False, []
    new_part, _, info, _, _ = h.solve_weekly(T_all, windows, force_ids=list(force_ids), latest=latest, **kw)
    if info["objective"] is None and latest:                             # R7: no window meets the deadline
        relaxed = True
        new_part, _, info, _, _ = h.solve_weekly(T_all, windows, force_ids=list(force_ids), **kw)
    if info["objective"] is None:                                        # R6: emergency cannot be forced in
        unplaced = list(force_ids)
        new_part, _, info, _, _ = h.solve_weekly(T_all, windows, **kw)
    rows = pd.concat([parts["executed"], new_part if len(new_part) else None, parts["future"]], ignore_index=True)
    plan2, blocks2 = rebuild(rows, T_all, windows, start)
    d = diff_plans(plan, plan2, now_day, LOCK_DAYS, new_ids, start)
    errs = h.validate_parallel(new_part, windows, T_all, start=ws, horizon_days=horizon, origin=start) if len(new_part) else []
    return dict(plan=plan2, blocks=blocks2, info=info, diff=d, validation_errors=errs, emergency_unplaced=unplaced,
                deadline_relaxed=relaxed, mode=mode, feasible=info["objective"] is not None)


class _Usage:
    """Resource usage of an existing plan, for the manual-style patch."""

    def __init__(self, plan, T_all, windows):
        self.w = windows.set_index("window_id")
        self.T = T_all.set_index("task_id")
        self.crew_min = Counter()                # (window, gang) -> setup + work
        self.slot = Counter()                    # (date, hour, gang)
        self.day_blocks = Counter()              # (section, date)
        self.tasks = defaultdict(list)           # window -> task ids
        setups = T_all.groupby("gang_type").crew_setup_min.max().to_dict()
        for r in plan.itertuples():
            t = self.T.loc[r.task_id]
            if (r.window_id, t.gang_type) not in self.crew_min:
                self.crew_min[r.window_id, t.gang_type] = int(setups[t.gang_type])
                wd = self.w.loc[r.window_id]
                for hh in range(int(wd.start_hour), int(wd.end_hour)):
                    self.slot[wd.date, hh, t.gang_type] += 1
            self.crew_min[r.window_id, t.gang_type] += int(t.duration_min)
            self.tasks[r.window_id].append(r.task_id)
        for wid in self.tasks:
            wd = self.w.loc[wid]
            self.day_blocks[wd.section_id, wd.date] += 1
        self.setups = setups

    def feasible(self, t, wid):
        wd = self.w.loc[wid]
        if (wid, t.gang_type) in self.crew_min:
            if self.crew_min[wid, t.gang_type] + int(t.duration_min) > int(wd.length_min):
                return False
        else:
            if int(t.duration_min) + int(self.setups[t.gang_type]) > int(wd.length_min):
                return False
            if any(self.slot[wd.date, hh, t.gang_type] >= o.GANGS[t.gang_type]
                   for hh in range(int(wd.start_hour), int(wd.end_hour))):
                return False
            if wid not in self.tasks and self.day_blocks[wd.section_id, wd.date] >= o.MAX_BLOCKS_PER_DAY:
                return False
        for other in self.tasks.get(wid, []):                            # safety rule
            ot = self.T.loc[other]
            lim = h.CONFLICT_PAIRS.get(frozenset({t.dept, ot.dept}))
            if t.dept != ot.dept and lim is not None and abs(t.km - ot.km) < lim:
                return False
        return True

    def add(self, t, wid):
        wd = self.w.loc[wid]
        if (wid, t.gang_type) not in self.crew_min:
            self.crew_min[wid, t.gang_type] = int(self.setups[t.gang_type])
            for hh in range(int(wd.start_hour), int(wd.end_hour)):
                self.slot[wd.date, hh, t.gang_type] += 1
        if wid not in self.tasks:
            self.day_blocks[wd.section_id, wd.date] += 1
        self.crew_min[wid, t.gang_type] += int(t.duration_min)
        self.tasks[wid].append(t.task_id)


def patch_insert(T_all, windows, plan, now_day, new_ids, start=START, lock_days=LOCK_DAYS, **_):
    """Manual-style: each new task (highest priority first) goes into the earliest window from now_day
    on that can take it without moving anything. Nothing existing is changed."""
    use = _Usage(plan, T_all, windows)
    Ti = T_all.set_index("task_id")
    rows, unplaced = [], []
    cand_w = windows[(windows.date >= start + pd.Timedelta(days=now_day))].sort_values(["date", "start_hour", "window_id"])
    for tid in sorted(new_ids, key=lambda t: -Ti.priority_score[t]):
        t = Ti.loc[tid].copy()
        t["task_id"] = tid
        placed = False
        for r in cand_w[cand_w.section_id == t.section_id].itertuples():
            if (r.date - start).days < t.earliest_idx or r.length_min < t.need_min:
                continue
            if o.needs_power(t.block_type) and not r.power_block_possible:
                continue
            if use.feasible(t, r.window_id):
                use.add(t, r.window_id)
                rows.append(dict(window_id=r.window_id, section_id=r.section_id, date=r.date,
                                 win_start=int(r.start_hour) * 60, win_end=int(r.end_hour) * 60, task_id=tid))
                placed = True
                break
        if not placed:
            unplaced.append(tid)
    allrows = pd.concat([plan, pd.DataFrame(rows) if rows else None], ignore_index=True)
    plan2, blocks2 = rebuild(allrows, T_all, windows, start)
    d = diff_plans(plan, plan2, now_day, LOCK_DAYS, new_ids, start)
    errs = h.validate_parallel(plan2, windows, T_all, start=start, horizon_days=int((windows.date.max() - start).days) + 1,
                               origin=start)
    return dict(plan=plan2, blocks=blocks2, info=dict(status="PATCH", objective=None, wall_time_s=0.0), diff=d,
                validation_errors=errs, emergency_unplaced=unplaced, mode="patch", feasible=True)


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def simulate(T, windows, plan, events, strategy="stable", start=START, time_limit=20, workers=8, **params):
    """Apply the events in order with one strategy. Returns (event table, summary dict, final plan, final blocks, T_all)."""
    T_all, cur = T.copy(), plan.copy()
    rows, totals = [], Counter()
    for e in sorted(events, key=lambda e: e["day"]):
        day, new = e["day"], e["tasks"]
        T_all = pd.concat([T_all, new], ignore_index=True)
        ids = list(new.task_id)
        force = ids if e["kind"] == "emergency" else []
        old_blocks = rebuild(cur[ROW_COLS], T_all, windows, start)[1]
        t0 = time.time()
        if strategy == "patch":
            r = patch_insert(T_all, windows, cur, day, ids, start=start, lock_days=params.get("lock_days", LOCK_DAYS))
        else:
            r = replan(T_all, windows, cur, day, new_ids=ids, force_ids=force, mode=strategy, start=start,
                       time_limit=time_limit, workers=workers, **params)
        wall = time.time() - t0
        d = r["diff"]
        resp = []
        for tid in force:
            sub = r["plan"][r["plan"].task_id == tid] if len(r["plan"]) else r["plan"]
            if len(sub):
                resp.append(int((sub.date.min() - start).days) - day)
        row = dict(day=day, kind=e["kind"], new_tasks=len(ids), new_scheduled=len(d["new_scheduled"]),
                   moved=len(d["moved"]), dropped=len(d["dropped"]), blocks_changed=d["blocks_changed"],
                   broken_commitments=len(d["broken_commitments"]), blocks_opened=d["blocks_opened"],
                   trains_change=_kpi_from_now(r["blocks"], day, start) - _kpi_from_now(old_blocks, day, start),
                   emergency_response_days=(resp[0] if len(resp) == 1 else resp or None),
                   emergency_unplaced=len(r["emergency_unplaced"]), deadline_relaxed=int(r.get("deadline_relaxed", False)),
                   net_value=int(o.net_value(r["plan"], r["blocks"], start)),
                   violations=len(r["validation_errors"]), solve_s=round(wall, 1))
        rows.append(row)
        for k in ("moved", "dropped", "blocks_changed", "broken_commitments", "emergency_unplaced", "deadline_relaxed"):
            totals[k] += row[k]
        cur = r["plan"]
    final_blocks = rebuild(cur[ROW_COLS], T_all, windows, start)[1]
    horizon = int((windows.date.max() - start).days) + 1
    final_errs = h.validate_parallel(cur, windows, T_all, start=start, horizon_days=horizon, origin=start)
    k = m.kpis(cur, final_blocks, T_all, windows, start, horizon)
    resp_all = [x for r_ in rows for x in ([r_["emergency_response_days"]] if isinstance(r_["emergency_response_days"], int)
                                           else (r_["emergency_response_days"] or []))]
    summary = dict(strategy=strategy, events=len(rows), total_moved=totals["moved"], total_dropped=totals["dropped"],
                   total_blocks_changed=totals["blocks_changed"], broken_commitments=totals["broken_commitments"],
                   emergencies_unplaced=totals["emergency_unplaced"], deadlines_relaxed=totals["deadline_relaxed"],
                   max_emergency_response_days=max(resp_all) if resp_all else None,
                   mean_emergency_response_days=round(sum(resp_all) / len(resp_all), 2) if resp_all else None,
                   tasks_scheduled=k["tasks_scheduled"], blocks=k["blocks"], trains_disrupted=k["trains_disrupted"],
                   net_value=k["net_value"], days_past_due_total=k["days_past_due_total"],
                   final_violations=len(final_errs))
    return pd.DataFrame(rows), summary, cur, final_blocks, T_all


def main():
    import arrivals as ar
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-sensitivity", action="store_true")
    a = ap.parse_args()

    T, windows = m.load_inputs()
    res = m.load_results()
    if res["plan"] is None:
        raise SystemExit("run horizons.py first")
    raw = pd.read_csv(o.DATA / "tasks_common_schema.csv")
    sections = pd.read_csv(o.DATA / "sections.csv")
    plan = m.with_week(res["plan"])
    events = ar.generate_events(T, raw, sections, seed=a.seed)
    print("Events:")
    for e in events:
        print(f"  day {e['day']:>2} {e['kind']:<9} " + ", ".join(
            f"{r.task_id} {r.dept} {r.section_id} {r.duration_min}min [{r.tier}]" for r in e["tasks"].itertuples()))
    OUT.mkdir(exist_ok=True)
    summaries, ev_tables = [], []
    for strat in ("stable", "full", "patch"):
        ev, s, p, b, _ = simulate(T, windows, plan, events, strat, time_limit=a.time_limit, workers=a.workers)
        summaries.append(s)
        ev_tables.append(ev.assign(strategy=strat))
        if strat == "stable":
            p.to_csv(OUT / "rolling_plan_stable.csv", index=False)
            b.to_csv(OUT / "rolling_blocks_stable.csv", index=False)
    summ = pd.DataFrame(summaries)
    ev = pd.concat(ev_tables, ignore_index=True)
    pd.set_option("display.width", 250)
    print("\nPer event:\n" + ev.to_string(index=False))
    print("\nFinal comparison:\n" + summ.T.to_string())
    summ.to_csv(OUT / "rolling_summary.csv", index=False)
    ev.to_csv(OUT / "rolling_events.csv", index=False)

    if not a.no_sensitivity:
        rows = []
        for mp in (0, 20, 40, 100, 300):
            _, s, *_ = simulate(T, windows, plan, events, "stable", time_limit=a.time_limit, workers=a.workers,
                                move_penalty=mp)
            rows.append(dict(vary="move_penalty", value=mp, **s))
        for ld in (0, 1, 2, 3):
            _, s, *_ = simulate(T, windows, plan, events, "stable", time_limit=a.time_limit, workers=a.workers,
                                lock_days=ld, move_penalty=0, drop_penalty=0)
            rows.append(dict(vary="lock_days (no prices)", value=ld, **s))
        for dl in (0, 1, 2, None):
            _, s, *_ = simulate(T, windows, plan, events, "stable", time_limit=a.time_limit, workers=a.workers,
                                emergency_max_delay=dl)
            rows.append(dict(vary="emergency_max_delay", value=-1 if dl is None else dl, **s))
        sens = pd.DataFrame(rows)
        print("\nSensitivity (stable strategy):\n" + sens[[
            "vary", "value", "total_moved", "total_dropped", "total_blocks_changed", "broken_commitments",
            "mean_emergency_response_days", "max_emergency_response_days", "tasks_scheduled", "blocks", "net_value",
            "final_violations"]].to_string(index=False))
        sens.to_csv(OUT / "rolling_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
