"""
Step 4 - Coordinated (multi-department) block planning over weekly and monthly horizons.
Builds on optimizer.py (Step 3): same data, same objective definition, same hard rules for
section / power block / blocks per day / gang pool. What is NEW here:

  1. PARALLEL work inside a block: each department's crew (gang type) works at the same
     time as the others, on its own tasks, one after another.
  2. SHARED setup: a crew pays its setup time once per block, not once per task.
  3. SAFETY rule (hard): S&T and Traction tasks in the same block must be >= SAFE_KM apart.
  4. MONTHLY plan (tasks -> weeks, coarse) refined by WEEKLY plans (tasks -> windows, fine),
     with unplaced work carried over to the next week.

Weekly model (CP-SAT)
    x[i,w] task i in window w | y[w] block taken | u[w,g] crew g works in window w
    crew capacity   setup_g*u[w,g] + sum(duration_i * x[i,w], crew g) <= length_w * u[w,g]
    safety          x[i,w] + x[j,w] <= 1  for S&T/Traction pairs closer than SAFE_KM
    (plus the Step 3 rules: at most once, section, power, blocks per section-day, gang pool
     per (date, hour), block only if used, gang only if block taken)
    objective       identical to Step 3: sum assignment_value - sum block_cost (integers)

Monthly model (CP-SAT, coarse; see B6, B11-B14 for the current 'blocks' version)
    z[i,k] task i in week k | b[s,k,j] the j-th cheapest window of section s in week k is taken
    per (section, week, crew, line/power) task minutes <= (1 - EMERGENCY_RESERVE) * capacity
    per (week, crew) network task minutes <= (1 - EMERGENCY_RESERVE) * pool size * covered minutes
    crew work must fit the blocks taken (indivisible tasks); optional deferral / balancing
    objective  sum month_value(first usable day) - sum block_cost * b - balance penalty
    model='simple' keeps the first version (one shared visit cost per section-week).

ASSUMPTIONS (in addition to A1-A10 of optimizer.py; A1 and A2 are replaced by B1/B2, A9 by B3)
    B1  Crews of different departments work in parallel in one block; a crew's tasks are
        sequential. Crew time = setup + sum of durations must fit the window length.
    B2  Crew setup = the largest setup_min among that crew's tasks in the task table
        (equal for all tasks of a department in the synthetic data). Charged once per
        block per crew. Travel between tasks of one crew is ignored (sections are 22 km).
    B3  Safety: only S&T vs Traction is restricted (CONFLICT_PAIRS); minimum distance
        SAFE_KM = 1.0 km is an ASSUMED value. Engineering is unrestricted.
    B4  A crew is still reserved for the whole window network-wide (A4), so parallel work
        does not relax the gang pool.
    B5  Tasks longer than any eligible window (setup + duration > window) are still
        reported as needing an extended block; parallel work does not change that.
    B6  Monthly model is an approximation. REVISED (model='blocks', default): the model counts
        BLOCKS per section-week (cheapest windows first, at most 2 per day, cost = block_cost) and
        crew work must fit those blocks (shortest window - crew setup, per crew; tasks are indivisible,
        so at most m tasks longer than usable/(m+1) share a block; power work must fit the
        power-capable blocks). A task is only offered weeks with a usable window on/after its
        earliest day and its value uses the day of that window. WHY: the first version charged ONE
        shared 'visit' per section-week and valued tasks at the week start; diagnosis (forced
        re-solves) showed 11 monthly-assigned tasks were left out by the weekly model because each
        needed its own block, started later, or had no usable window that week. Still not visible
        at this level: safety conflicts, exact gang timing, window-by-window travel.
        model='simple' keeps the first version for comparison. Setup is charged per task in the
        legacy minute-capacity rows (conservative). EMERGENCY_RESERVE = 20% is assumed, unchanged.
    B7  Weekly plans refine the monthly plan: a task is only considered in the week the
        monthly plan gave it (or later, if carried over). Unplaced tasks are carried over;
        nothing is dropped silently.
    B8  Delay penalty uses days since the planning origin (week 1 day 1) and the overdue
        flag at the origin, as in Step 3, so values from different weeks are comparable.
    B9  Synthetic data, expert-set weights: results show the method works on this data,
        not what a real railway would gain.
    B10 Preventive-maintenance window: a task may not be scheduled earlier than
        EARLY_DAYS (assumed 14) before its due date; overdue tasks may go any time.
        WHY: without it the objective (which only penalises delay) puts every task into
        week 1 - observed in the first run: all 136 schedulable tasks in week 1, weeks 2-4
        empty - and the monthly plan has nothing to decide. Due dates are NOT hard
        deadlines (A7): finishing after the due date is allowed and reported as lateness.
    B11 Deferral (defer=True): non-overdue MEDIUM/LOW tasks may be moved to later weeks, but only to
        weeks that END on/before their due date. They pay the normal delay price, so smoothing must
        be worth more than the delay it causes. Overdue, CRITICAL and HIGH tasks are PINNED to their
        first usable week, so balancing cannot delay them.
    B12 Balancing (balance > 0): the monthly model pays `balance` points per 10 minutes of weekly
        workload above 1.25 x the average weekly load. Expert-set, not calibrated; it only shapes
        the monthly plan - weekly values and net_value are computed exactly as before. Deferring
        work lowers net_value by the extra delay penalty (that is how net_value is defined).
    B13 Lateness-aware monthly value (model='blocks'): a not-yet-overdue task is charged x3 per day
        once it passes its due date, as overdue tasks are. Weekly objective unchanged.
    B14 Re-plan (default): at the start of week k the monthly model is re-solved for weeks k.. over
        all still-unscheduled tasks and its week-k assignment is the pool, instead of pushing
        unplaced tasks blindly into the next week. --no-replan gives the static carry-over.
    B15 Due backfill (default in main): see run_horizon(backfill=True). Added after the diagnosis showed
        one task due inside the horizon (+55 net value if placed) that the monthly plan had skipped.

Diagnostics: explain_unscheduled() re-solves a week with a task FORCED in and reports what that
costs; final_unscheduled() lists every task still unscheduled after the last week with a reason
(never placeable / declined as net-negative / indifferent / pool effect) and an action;
capacity_diagnosis() shows eligible work vs scheduled work vs free window time per week.

Run:  python horizons.py [--time-limit 30] [--workers 8] [--reserve 0.2] [--weeks 4]
                         [--monthly-model blocks|simple] [--no-replan] [--no-defer] [--balance 10] [--no-backfill]
                         [--compare-variants] [--compare-myopic] [--no-explain] [--no-ablation]
"""
import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

import optimizer as o

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"

SAFE_KM = 1.0
CONFLICT_PAIRS = {frozenset({"S&T", "TRD"}): SAFE_KM}       # dept pair -> minimum km apart
EMERGENCY_RESERVE = 0.20
BALANCE_PTS = 0          # B12: points per 10 min of weekly load above the soft cap (0 = off)
BALANCE_CAP = 1.25       # B12: soft cap = this x average weekly load
DEFAULT_BALANCE = 10     # B12 default used by main() = 1 point per minute of excess weekly load; in a sensitivity run 20+ delayed overdue work
EARLY_DAYS = 14          # B10: work may start at most this many days before its due date
N_WEEKS = 4
START = o.START

PLAN_COLS = o.TASK_COLS + ["km", "duration_min", "crew_setup_min"]


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------
def prepare_tasks(tasks, early_days=EARLY_DAYS):
    """Crew setup (B2), stand-alone need = duration + crew setup, earliest start (B10).
    earliest_idx = days from the plan origin (day 0 = START, where overdue_days is measured)."""
    T = tasks.copy()
    T["earliest_idx"] = -T.overdue_days - int(early_days)
    setups = T.groupby("gang_type").setup_min.max()
    T["crew_setup_min"] = T.gang_type.map(setups).astype(int)
    T["need_min"] = T.duration_min + T.crew_setup_min
    return T


def conflict_pairs(T):
    """Index pairs of tasks (same section, restricted departments) closer than the limit."""
    out = []
    for _, g in T.groupby("section_id"):
        rows = list(g.itertuples())
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                ra, rb = rows[a], rows[b]
                lim = CONFLICT_PAIRS.get(frozenset({ra.dept, rb.dept}))
                if ra.dept != rb.dept and lim is not None and abs(ra.km - rb.km) < lim:
                    out.append((ra.Index, rb.Index))
    return out


# --------------------------------------------------------------------------
# Weekly (window-level) parallel model
# --------------------------------------------------------------------------
def finish_plan_parallel(rows, T, W):
    """Attach attributes, sequence each crew inside its window (after its setup), summarise blocks."""
    cols = ["window_id", "section_id", "date", "win_start", "win_end"]
    if rows is None or len(rows) == 0:
        return pd.DataFrame(), pd.DataFrame()
    plan = rows.merge(T[PLAN_COLS], on="task_id")
    plan = plan.sort_values(["date", "window_id", "gang_type", "priority_score", "task_id"],
                            ascending=[True, True, True, False, True])
    cum = plan.groupby(["window_id", "gang_type"]).duration_min.cumsum() - plan.duration_min
    plan["start_min"] = plan.win_start + plan.crew_setup_min + cum
    plan["end_min"] = plan.start_min + plan.duration_min

    crew = (plan.groupby(["window_id", "gang_type"])
            .agg(crew_work=("duration_min", "sum"), setup=("crew_setup_min", "first")).reset_index())
    crew["crew_total"] = crew.crew_work + crew.setup
    cs = crew.groupby("window_id").agg(longest_crew_min=("crew_total", "max"),
                                       crew_total_min=("crew_total", "sum"),
                                       n_crews=("gang_type", "count")).reset_index()
    blocks = (plan.groupby("window_id")
              .agg(section_id=("section_id", "first"), date=("date", "first"),
                   win_start=("win_start", "first"), win_end=("win_end", "first"),
                   n_tasks=("task_id", "count"), n_depts=("dept", "nunique"),
                   work_min=("duration_min", "sum"),
                   depts=("dept", lambda d: "+".join(sorted(set(d)))))
              .reset_index().merge(cs, on="window_id")
              .merge(W[["window_id", "trains_affected", "goods_trains_affected"]], on="window_id"))
    blocks["merged"] = blocks.n_depts > 1
    blocks["length_min"] = blocks.win_end - blocks.win_start
    blocks["utilization"] = (blocks.longest_crew_min / blocks.length_min).round(3)   # busiest crew
    blocks["crew_utilization"] = (blocks.crew_total_min / (blocks.length_min * blocks.n_crews)).round(3)
    return plan, blocks


def solve_weekly(tasks, windows, horizon_days=7, start=START, origin=None, task_ids=None,
                 section=None, merge=True, time_limit=30, workers=8, rel_gap_limit=0.0, seed=1,
                 force_ids=()):
    """`tasks` must come from prepare_tasks(). `origin` = day 0 of the whole plan (B8).
    force_ids: tasks that MUST be scheduled (diagnosis only: shows what placing a task costs)."""
    origin = origin if origin is not None else start
    T0 = tasks if task_ids is None else tasks[tasks.task_id.isin(set(task_ids))]
    W, T = o.scope(T0, windows, horizon_days, start, section)
    pairs = o.candidate_pairs(T, W)
    if "earliest_idx" in T:                                             # B10
        pairs = [(i, w) for (i, w) in pairs if (W.date[w] - origin).days >= T.earliest_idx[i]]
    unsched = o.find_unschedulable(T, W)
    unknown = sorted(set(T.gang_type) - set(o.GANGS))
    if unknown:
        raise ValueError(f"gang types missing from GANGS: {unknown}")

    m = cp_model.CpModel()
    x = {p: m.NewBoolVar(f"x{p[0]}_{p[1]}") for p in pairs}
    y = {w: m.NewBoolVar(f"y{w}") for w in W.index}
    u = {(w, g): m.NewBoolVar(f"u{w}_{g}") for w in W.index for g in o.GANGS}
    gang_of, dur = T.gang_type.to_dict(), T.duration_min.to_dict()
    setup_g = T.groupby("gang_type").crew_setup_min.max().to_dict()

    by_task, by_win = defaultdict(list), defaultdict(list)
    for (i, w) in pairs:
        by_task[i].append(w)
        by_win[w].append(i)

    for i, ws in by_task.items():                                       # at most once
        m.Add(sum(x[i, w] for w in ws) <= 1)
    forced_missing = []
    for tid in force_ids:                                               # diagnosis: force placement
        idx = T.index[T.task_id == tid]
        if len(idx) == 0 or idx[0] not in by_task:
            forced_missing.append(tid)
        else:
            m.Add(sum(x[idx[0], w] for w in by_task[idx[0]]) == 1)
    for (i, w) in pairs:
        m.Add(x[i, w] <= y[w])
        m.Add(x[i, w] <= u[w, gang_of[i]])
    for w in W.index:
        m.Add(y[w] <= sum(x[i, w] for i in by_win.get(w, [])))         # block only if used
        for g in o.GANGS:
            m.Add(u[w, g] <= y[w])                                      # crew only if block taken
            tg = [i for i in by_win.get(w, []) if gang_of[i] == g]
            if tg:                                                      # B1/B2 crew capacity
                m.Add(sum(dur[i] * x[i, w] for i in tg) + setup_g[g] * u[w, g]
                      <= int(W.length_min[w]) * u[w, g])
        if not merge:                                                   # ablation: one dept per block
            m.Add(sum(u[w, g] for g in o.GANGS) <= 1)
    for _, g in W.groupby(["section_id", "date"]):                      # blocks per section-day
        m.Add(sum(y[w] for w in g.index) <= o.MAX_BLOCKS_PER_DAY)
    slots = defaultdict(list)                                           # crews per (date, hour)
    for w, r in W.iterrows():
        for h in range(int(r.start_hour), int(r.end_hour)):
            slots[(r.date, h)].append(w)
    for ws in slots.values():
        for g, n in o.GANGS.items():
            if len(ws) > n:
                m.Add(sum(u[w, g] for w in ws) <= n)
    cpairs = conflict_pairs(T)                                          # B3 safety
    for (i, j) in cpairs:
        for w in set(by_task.get(i, [])) & set(by_task.get(j, [])):
            m.Add(x[i, w] + x[j, w] <= 1)

    obj = []
    for (i, w) in pairs:
        obj.append(o.assignment_value(T.priority_score[i], T.severity[i], T.overdue_days[i],
                                      (W.date[w] - origin).days) * x[i, w])
    for w in W.index:
        obj.append(-o.block_cost(W.trains_affected[w], W.goods_trains_affected[w]) * y[w])
    m.Maximize(sum(obj))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    s.parameters.random_seed = seed
    s.parameters.relative_gap_limit = rel_gap_limit
    t0 = time.time()
    status = s.Solve(m)
    name = s.StatusName(status)
    info = dict(status=name, proven_optimal=(name == "OPTIMAL"), wall_time_s=round(time.time() - t0, 2),
                n_vars=len(pairs) + len(y) + len(u), unschedulable=unsched,
                start=start, origin=origin, horizon_days=horizon_days, merge=merge,
                n_conflict_pairs=len(cpairs), n_tasks=len(T),
                objective=None, bound=None, abs_gap=None, rel_gap=None)
    info["forced_missing"] = forced_missing
    if forced_missing:                                                  # no eligible window at all
        info["status"] = "INFEASIBLE_FORCED"
        return pd.DataFrame(), pd.DataFrame(), info, W, T
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return pd.DataFrame(), pd.DataFrame(), info, W, T
    info["objective"] = int(round(s.ObjectiveValue()))
    info["bound"] = s.BestObjectiveBound()
    info["abs_gap"], info["rel_gap"] = o.compute_gap(info["objective"], info["bound"])
    rows = pd.DataFrame([o._row(W, w, T.task_id[i]) for (i, w), var in x.items() if s.Value(var)])
    plan, blocks = finish_plan_parallel(rows, T, W)
    return plan, blocks, info, W, T


# --------------------------------------------------------------------------
# Independent validator for parallel plans (rebuilt from tasks + windows tables)
# --------------------------------------------------------------------------
def validate_parallel(plan, windows, tasks, start=START, horizon_days=7, merge=True, origin=START):
    errs = []
    if plan is None or len(plan) == 0:
        return errs
    t, w = tasks.set_index("task_id"), windows.set_index("window_id")
    setups = tasks.groupby("gang_type").setup_min.max().to_dict()
    end = start + pd.Timedelta(days=horizon_days)

    for tid in plan.task_id[plan.task_id.duplicated()].unique():
        errs.append(f"[DUPLICATE] {tid} scheduled more than once")
    crew_work, crew_tasks, win_depts, win_tasks = Counter(), defaultdict(list), defaultdict(set), defaultdict(list)
    for r in plan.itertuples():
        if r.task_id not in t.index or r.window_id not in w.index:
            errs.append(f"[UNKNOWN] {r.task_id} / {r.window_id}")
            continue
        tk, wd = t.loc[r.task_id], w.loc[r.window_id]
        if tk.section_id != wd.section_id:
            errs.append(f"[SECTION] {r.task_id} ({tk.section_id}) in window of {wd.section_id}")
        if not (start <= wd.date < end):
            errs.append(f"[HORIZON] {r.task_id} in {r.window_id} on {wd.date.date()}")
        if o.needs_power(tk.block_type) and not wd.power_block_possible:
            errs.append(f"[POWER] {r.task_id} needs a power block, {r.window_id} has none")
        if "earliest_idx" in tasks.columns and (wd.date - origin).days < tk.earliest_idx:
            errs.append(f"[EARLY] {r.task_id} on day {(wd.date - origin).days}, "
                        f"not allowed before day {int(tk.earliest_idx)}")
        if tk.gang_type not in o.GANGS:
            errs.append(f"[GANG_UNKNOWN] {r.task_id}: {tk.gang_type}")
            continue
        crew_work[(r.window_id, tk.gang_type)] += int(tk.duration_min)
        crew_tasks[(r.window_id, tk.gang_type)].append(r)
        win_depts[r.window_id].add(tk.dept)
        win_tasks[r.window_id].append(r.task_id)

    for (wid, g), work in crew_work.items():                            # B1/B2
        total = work + int(setups[g])
        if total > w.length_min[wid]:
            errs.append(f"[CREW_CAPACITY] {wid} crew {g}: setup+work {total} min > window {w.length_min[wid]} min")
    opened = set(win_depts)
    for (sec, d), n in Counter((w.section_id[x_], w.date[x_]) for x_ in opened).items():
        if n > o.MAX_BLOCKS_PER_DAY:
            errs.append(f"[BLOCKS_PER_DAY] {sec} {d.date()}: {n} blocks > {o.MAX_BLOCKS_PER_DAY}")
    slot = Counter()
    for (wid, g) in crew_work:
        for h in range(int(w.start_hour[wid]), int(w.end_hour[wid])):
            slot[(w.date[wid], h, g)] += 1
    for (d, h, g), n in sorted(slot.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if n > o.GANGS[g]:
            errs.append(f"[GANG] {g} on {d.date()} {h:02d}:00 - {n} reserved, only {o.GANGS[g]} exist")
    for wid, tids in win_tasks.items():                                 # B3 safety
        for a in range(len(tids)):
            for b in range(a + 1, len(tids)):
                ta, tb = t.loc[tids[a]], t.loc[tids[b]]
                lim = CONFLICT_PAIRS.get(frozenset({ta.dept, tb.dept}))
                if ta.dept != tb.dept and lim is not None and abs(ta.km - tb.km) < lim:
                    errs.append(f"[SAFETY] {wid}: {tids[a]} ({ta.dept}) and {tids[b]} ({tb.dept}) "
                                f"only {abs(ta.km - tb.km):.2f} km apart")
    if not merge:
        for wid, ds in win_depts.items():
            if len(ds) > 1:
                errs.append(f"[NO_MERGE] {wid} has several departments: {sorted(ds)}")
    if {"start_min", "end_min"} <= set(plan.columns):                   # crew sequencing
        for (wid, g), rs in crew_tasks.items():
            lo, hi = int(w.start_hour[wid]) * 60, int(w.end_hour[wid]) * 60
            prev = lo + int(setups[g])
            for r in sorted(rs, key=lambda r: r.start_min):
                dur = int(t.duration_min[r.task_id])
                if r.start_min < prev or r.end_min - r.start_min != dur or r.end_min > hi:
                    errs.append(f"[SEQUENCE] {wid} crew {g}: {r.task_id} {r.start_min}-{r.end_min} "
                                f"overlaps, starts before setup ends, or leaves the window")
                prev = max(prev, r.end_min)
    return errs


# --------------------------------------------------------------------------
# Monthly (week-level) model
# --------------------------------------------------------------------------
def month_value(t, day, late_aware=True):
    """Monthly value of doing task t on (estimated) day `day`: the SAME delay price as the weekly
    model, so 'is it worth a block' is decided consistently. (An earlier version charged deferrable
    tasks only for lateness; the weekly model then declined 12 of the tasks it had assigned.)
    late_aware (B13): a not-yet-overdue task is charged x3 per day once it passes its due date, like
    an overdue task (without it 'finished after due date' rose from 4 to 9 tasks in the first
    block-aware run). late_aware=False reproduces o.assignment_value exactly (model='simple')."""
    coef, due, d = 3 + 4 * int(t.severity), int(-t.overdue_days), int(day)
    if late_aware and t.overdue_days <= 0:
        before, late = max(0, min(d, due)), max(0, d - max(due, 0))
        return int(round(1000 * float(t.priority_score))) - (coef * (before + 3 * late)) // 2
    return o.assignment_value(t.priority_score, t.severity, t.overdue_days, day)


def is_deferrable(t):
    """B11: not overdue and not CRITICAL/HIGH -> may be pushed later, but never past its due date."""
    return t.overdue_days <= 0 and t.tier not in ("CRITICAL", "HIGH")


def solve_monthly(T, windows, start=START, n_weeks=N_WEEKS, reserve=EMERGENCY_RESERVE,
                  time_limit=30, workers=8, seed=1, model="blocks", first_week=0, task_ids=None,
                  balance=BALANCE_PTS, balance_cap=BALANCE_CAP, defer=False):
    """Assign tasks to weeks first_week..n_weeks-1 (0-based).
    model='simple' : first version - one shared visit cost per section-week, value at week start.
    model='blocks' : (B6 revised) blocks are counted. Per section-week the model chooses how many
                     blocks n it takes (cheapest windows first, at most 2 per day); crew work must
                     fit n blocks of (shortest window - crew setup) minutes AND tasks are indivisible
                     (at most m tasks longer than usable/(m+1) per block, m = 1..3); power work must
                     fit the power-capable blocks; value uses the day of the first usable window; a
                     task is only offered weeks that have a usable window on/after its earliest day.
    defer=True     : deferrable tasks (B11) may only go to weeks that END on/before their due date
                     (so deferral never makes them late); all other tasks (overdue, CRITICAL, HIGH)
                     are pinned to their first usable week, so balancing can never delay them;
                     balance > 0 adds a penalty of `balance`
                     points per 10 minutes of weekly workload above balance_cap x average load (B12)."""
    if model not in ("simple", "blocks"):
        raise ValueError("model must be 'simple' or 'blocks'")
    T = T if task_ids is None else T[T.task_id.isin(set(task_ids))]
    weeks = list(range(first_week, n_weeks))
    t_lo, t_hi = start + pd.Timedelta(days=7 * first_week), start + pd.Timedelta(days=7 * n_weeks)
    Wm = windows[(windows.date >= t_lo) & (windows.date < t_hi)].copy()
    Wm["day"] = (Wm.date - start).dt.days
    Wm["week"] = Wm.day // 7
    Wm["cost"] = [o.block_cost(a, b) for a, b in zip(Wm.trains_affected, Wm.goods_trains_affected)]

    cap, maxlen, visit, cover = defaultdict(int), {}, {}, defaultdict(int)
    for (sec, wk, _), g in Wm.groupby(["section_id", "week", "date"]):
        for kind, gg in (("any", g), ("power", g[g.power_block_possible])):
            lens = sorted(gg.length_min, reverse=True)[:o.MAX_BLOCKS_PER_DAY]
            cap[(sec, wk, kind)] += int(sum(lens))
    for (sec, wk), g in Wm.groupby(["section_id", "week"]):
        maxlen[(sec, wk, "any")] = int(g.length_min.max())
        p = g[g.power_block_possible]
        maxlen[(sec, wk, "power")] = int(p.length_min.max()) if len(p) else 0
        visit[(sec, wk)] = int(g.cost.min())
    for (wk, _), g in Wm.groupby(["week", "date"]):                     # minutes covered by any window
        hrs = set()
        for r in g.itertuples():
            hrs |= set(range(int(r.start_hour), int(r.end_hour)))
        cover[wk] += 60 * len(hrs)
    Wsec = {sec: g for sec, g in Wm.groupby("section_id")}

    m = cp_model.CpModel()
    z, dday, defer_of = {}, {}, {}
    for i, t in T.iterrows():
        pw = o.needs_power(t.block_type)
        kind = "power" if pw else "any"
        defer_of[i] = bool(defer and is_deferrable(t))
        ks = []
        for k in weeks:
            if model == "simple":
                if 7 * k + 6 >= t.earliest_idx and maxlen.get((t.section_id, k, kind), 0) >= t.need_min:
                    ks.append(k)
                    dday[(i, k)] = 7 * k
            else:
                sw = Wsec.get(t.section_id)
                if sw is None:
                    continue
                el = sw[(sw.week == k) & (sw.day >= t.earliest_idx) & (sw.length_min >= t.need_min)]
                if pw:
                    el = el[el.power_block_possible]
                if len(el):
                    ks.append(k)
                    dday[(i, k)] = int(el.day.min())
        if defer and ks:                                                # B11
            if defer_of[i]:                                             # deferrable: weeks ending by due date
                ok = [k for k in ks if 7 * k + 6 <= -t.overdue_days]
                ks = ok or ks[:1]
            else:                                                       # overdue / CRITICAL / HIGH: never moved
                ks = ks[:1]
        for k in ks:
            z[i, k] = m.NewBoolVar(f"z{i}_{k}")
    by_task = defaultdict(list)
    for (i, k) in z:
        by_task[i].append(k)
    for i, ks in by_task.items():
        m.Add(sum(z[i, k] for k in ks) <= 1)

    grp = defaultdict(list)
    for (i, k) in z:
        t = T.loc[i]
        grp[(t.section_id, k, t.gang_type, "power" if o.needs_power(t.block_type) else "any")].append(i)
    for (sec, k, g, kind), ids in grp.items():                          # reserve-adjusted minutes
        m.Add(sum(int(T.need_min[i]) * z[i, k] for i in ids) <= int((1 - reserve) * cap[(sec, k, kind)]))
    for k in weeks:                                                     # network crew pool per week
        for g, n in o.GANGS.items():
            ids = [i for (i, kk) in z if kk == k and T.gang_type[i] == g]
            if ids:
                m.Add(sum(int(T.need_min[i]) * z[i, k] for i in ids) <= int((1 - reserve) * n * cover[k]))

    late_aware = model == "blocks"
    obj = [month_value(T.loc[i], dday[i, k], late_aware) * var for (i, k), var in z.items()]
    n_blocks = {}
    if model == "simple":
        v = {sk: m.NewBoolVar(f"v{sk}") for sk in visit}
        for (i, k), var in z.items():
            m.Add(var <= v[(T.section_id[i], k)])
        obj += [-visit[sk] * var for sk, var in v.items()]
    else:
        sec_week = sorted({(T.section_id[i], k) for (i, k) in z})
        for (sec, k) in sec_week:
            ids = [i for (i, kk) in z if kk == k and T.section_id[i] == sec]
            lo = min(int(T.earliest_idx[i]) for i in ids)
            sw = Wsec[sec][(Wsec[sec].week == k) & (Wsec[sec].day >= lo)]
            per_day = sw.groupby("day").size().clip(upper=o.MAX_BLOCKS_PER_DAY)
            J = int(per_day.sum())
            costs = sorted(sw.cost)[:J]
            b = [m.NewBoolVar(f"b{sec}_{k}_{j}") for j in range(len(costs))]
            n = sum(b)
            obj += [-c * bj for c, bj in zip(costs, b)]
            lbar = int(sw.length_min.min())
            pw = sw[sw.power_block_possible]
            pdays = pw.groupby("day").size().clip(upper=o.MAX_BLOCKS_PER_DAY)
            npv = m.NewIntVar(0, int(pdays.sum()), f"np{sec}_{k}")
            m.Add(npv <= n)
            lpbar = int(pw.length_min.min()) if len(pw) else 0
            for g in {T.gang_type[i] for i in ids}:
                setup = int(T[T.gang_type == g].crew_setup_min.max())
                mine = [i for i in ids if T.gang_type[i] == g]
                pmine = [i for i in mine if o.needs_power(T.block_type[i])]
                for group, nvar, length in ((mine, n, lbar), (pmine, npv, lpbar)):
                    if not group:
                        continue
                    usable = max(0, length - setup)
                    m.Add(sum(int(T.duration_min[i]) * z[i, k] for i in group) <= usable * nvar)
                    for per_block in (1, 2, 3):          # tasks are indivisible: more than `per_block`
                        big = [i for i in group if int(T.duration_min[i]) * (per_block + 1) > usable]
                        if len(big) > per_block:         # tasks longer than usable/(per_block+1) cannot share
                            m.Add(sum(z[i, k] for i in big) <= per_block * nvar)
            n_blocks[(sec, k)] = n
    if balance and len(z):                                              # B12 workload smoothing
        total = sum(int(T.need_min[i]) for i in by_task)
        soft = int(balance_cap * total / max(1, len(weeks)))
        for k in weeks:
            ids = [i for (i, kk) in z if kk == k]
            if not ids:
                continue
            ex = m.NewIntVar(0, total // 10 + 1, f"ex{k}")
            m.Add(sum(int(T.need_min[i]) * z[i, k] for i in ids) <= soft + 10 * ex)
            obj.append(-int(balance) * ex)
    m.Maximize(sum(obj))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    s.parameters.random_seed = seed
    s.parameters.relative_gap_limit = 0.0
    t0 = time.time()
    status = s.Solve(m)
    name = s.StatusName(status)
    info = dict(status=name, proven_optimal=(name == "OPTIMAL"), wall_time_s=round(time.time() - t0, 2),
                n_vars=len(z) + len(n_blocks), reserve=reserve, model=model, first_week=first_week,
                objective=None, bound=None, abs_gap=None, rel_gap=None)
    cols = ["task_id", "week", "dept", "section_id", "task", "tier", "need_min"]
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return pd.DataFrame(columns=cols), info
    info["objective"] = int(round(s.ObjectiveValue()))
    info["bound"] = s.BestObjectiveBound()
    info["abs_gap"], info["rel_gap"] = o.compute_gap(info["objective"], info["bound"])
    rows = [dict(task_id=T.task_id[i], week=k) for (i, k), var in z.items() if s.Value(var)]
    assign = pd.DataFrame(rows, columns=["task_id", "week"])
    assign = assign.merge(T[["task_id", "dept", "section_id", "task", "tier", "need_min"]], on="task_id")
    return assign.sort_values(["week", "section_id", "task_id"]).reset_index(drop=True)[cols], info


# --------------------------------------------------------------------------
# Rolling pipeline: monthly -> weekly refinement with carry-over
# --------------------------------------------------------------------------
def run_horizon(T, windows, assign=None, start=START, n_weeks=N_WEEKS, time_limit=30, workers=8,
                replan=None, monthly_time_limit=15, backfill=False):
    """Weekly refinement, week by week.
      assign + replan=None : static - week k pool = monthly assignment for week k + carried-over tasks
      assign + replan=dict : rolling - at the start of week k the monthly model is RE-SOLVED for weeks
                             k.. over every still-unscheduled task (dict = solve_monthly kwargs); its week-k
                             assignment is the pool. Tasks the weekly plan could not place are therefore
                             re-planned instead of being pushed blindly into the next week.
      assign=None          : 'myopic' reference - every week sees all unscheduled tasks.
    backfill=True (B15)    : tasks that are still unscheduled and already DUE by the end of week k are
                             added to that week's pool even if the monthly plan did not assign them, so
                             an approximate monthly plan can never silently skip work that is due; the
                             weekly objective still decides whether it is worth a block. Deferral is
                             not undone: deferred tasks are never due before their assigned week.
    Returns (plan, blocks, summary, done). summary.attrs['monthly_log'] lists the monthly plans used."""
    done, carry, plans, blocks_all, rows, log = set(), [], [], [], [], []
    due_idx = dict(zip(T.task_id, -T.overdue_days))
    for k in range(n_weeks):
        ws = start + pd.Timedelta(days=7 * k)
        remaining = [t for t in T.task_id if t not in done]
        if assign is None:
            pool = remaining
        elif replan is not None:
            if k == 0:
                a = assign
            else:
                a, _ = solve_monthly(T, windows, start=start, n_weeks=n_weeks, first_week=k,
                                     task_ids=remaining, time_limit=monthly_time_limit, workers=workers, **replan)
            log.append(a)
            pool = list(a[a.week == k].task_id)
        else:
            pool = list(assign[assign.week == k].task_id) + carry
        n_back = 0
        if backfill and assign is not None:
            extra = [t for t in remaining if t not in set(pool) and due_idx[t] <= 7 * k + 6]
            pool, n_back = pool + extra, len(extra)
        plan, blocks, info, W, Tk = solve_weekly(T, windows, start=ws, origin=start, task_ids=pool,
                                                 time_limit=time_limit, workers=workers)
        errs = validate_parallel(plan, windows, T, start=ws, origin=start)
        if info["objective"] is not None and len(plan):
            nv = o.net_value(plan, blocks, start)
            if nv != info["objective"]:
                raise AssertionError(f"week {k + 1}: solver objective {info['objective']} != net_value {nv}")
        got = set(plan.task_id) if len(plan) else set()
        unplaced = [t for t in pool if t not in got]
        carry = unplaced if (assign is not None and replan is None) else []
        done |= got
        if len(plan):
            plans.append(plan.assign(week=k + 1))
            blocks_all.append(blocks.assign(week=k + 1))
        rows.append(dict(week=k + 1, status=info["status"], pool=len(pool), backfilled=n_back,
                         scheduled=len(got), carried_over=len(unplaced), blocks=len(blocks),
                         solver_objective=info["objective"],
                         merged_blocks=int(blocks.merged.sum()) if len(blocks) else 0,
                         trains_disrupted=int(blocks.trains_affected.sum()) if len(blocks) else 0,
                         net_value=o.net_value(plan, blocks, start) if len(plan) else 0,
                         rel_gap=round(info["rel_gap"], 4) if info["rel_gap"] is not None else None,
                         proven_optimal=info["proven_optimal"], violations=len(errs),
                         wall_time_s=info["wall_time_s"]))
        if errs:
            print(f"!! week {k + 1}: {len(errs)} violations, first: {errs[0]}")
    plan = pd.concat(plans, ignore_index=True) if plans else pd.DataFrame()
    blocks = pd.concat(blocks_all, ignore_index=True) if blocks_all else pd.DataFrame()
    summ = pd.DataFrame(rows)
    summ.attrs["monthly_log"] = log
    return plan, blocks, summ, done


def realization(assign, plan):
    """How closely did the weekly plans follow the INITIAL monthly plan?
    ON_TIME = scheduled in its assigned week | SLIPPED = later | PULLED_FORWARD = earlier (re-plan only)
    NOT_SCHEDULED = never scheduled."""
    first = plan.groupby("task_id").week.min() if len(plan) else pd.Series(dtype=int)
    r = assign.copy()
    r["realized_week"] = r.task_id.map(first)
    r["status"] = "NOT_SCHEDULED"
    r.loc[r.realized_week == r.week + 1, "status"] = "ON_TIME"
    r.loc[r.realized_week > r.week + 1, "status"] = "SLIPPED"
    r.loc[r.realized_week < r.week + 1, "status"] = "PULLED_FORWARD"
    r.loc[r.realized_week.isna(), "status"] = "NOT_SCHEDULED"
    return r


def realization_summary(assign, plan):
    r = realization(assign, plan)
    cnt = r.status.value_counts().to_dict()
    added = (set(plan.task_id) - set(assign.task_id)) if len(plan) else set()
    return dict(assigned=len(assign), on_time=cnt.get("ON_TIME", 0), slipped=cnt.get("SLIPPED", 0),
                pulled_forward=cnt.get("PULLED_FORWARD", 0), not_scheduled=cnt.get("NOT_SCHEDULED", 0),
                added_after_initial_plan=len(added),
                realization_pct=round(100 * cnt.get("ON_TIME", 0) / max(1, len(assign)), 1))


def free_capacity(windows, blocks, start=START, n_weeks=N_WEEKS):
    """Window minutes that were NOT taken as blocks, per week: the real emergency reserve."""
    W = windows[(windows.date >= start) & (windows.date < start + pd.Timedelta(days=7 * n_weeks))].copy()
    W["week"] = ((W.date - start).dt.days // 7) + 1
    used = set(blocks.window_id) if len(blocks) else set()
    W["taken"] = W.window_id.isin(used)
    g = W.groupby("week").apply(lambda d: pd.Series(dict(
        window_min=int(d.length_min.sum()), free_min=int(d[~d.taken].length_min.sum()))),
        include_groups=False)
    g["free_pct"] = (100 * g.free_min / g.window_min).round(1)
    return g


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def explain_unscheduled(T, windows, plan, ids, start=START, n_weeks=N_WEEKS, time_limit=15, workers=8):
    """Solver evidence for tasks left unscheduled: for every week the task could be done in,
    re-solve that week with the task FORCED in and compare with the same week without it.
    delta = objective(forced) - objective(not forced): what placing the task costs (or gains).
    reason:
      NOT_YET_ELIGIBLE       its earliest start (B10) is after the end of the horizon
      NO_ELIGIBLE_WINDOW     no window it may use in any week (length / power / earliest start)
      DECLINED_NET_NEGATIVE  best delta < 0: placing it lowers the net value (block cost, bumped
                             tasks or delay outweigh its value) - the objective declines it
      INDIFFERENT_TIE        best delta == 0
      WEEKLY_POOL_EFFECT     best delta > 0: placing it would RAISE the week's value, so only the
                             pool / carry-over rules kept it out (a monthly-to-weekly mismatch)
    delta is exact only when both solves are proven optimal (column `proven`)."""
    rows = []
    for tid in ids:
        t = T[T.task_id == tid].iloc[0]
        best, weeks_ok, proven_all = None, [], True
        if t.earliest_idx > 7 * n_weeks - 1:
            rows.append(dict(task_id=tid, dept=t.dept, section_id=t.section_id, tier=t.tier,
                             due_idx=int(-t.overdue_days), earliest_idx=int(t.earliest_idx), weeks_possible="-",
                             best_week=None, delta_if_forced=None, proven=True, reason="NOT_YET_ELIGIBLE"))
            continue
        for k in range(n_weeks):
            if 7 * k + 6 < t.earliest_idx:
                continue
            ws = start + pd.Timedelta(days=7 * k)
            base = list(plan[plan.week == k + 1].task_id) if len(plan) else []
            if base:
                _, _, i0, _, _ = solve_weekly(T, windows, start=ws, origin=start, task_ids=base,
                                              time_limit=time_limit, workers=workers)
                obj0, p0 = i0["objective"], i0["proven_optimal"]
            else:
                obj0, p0 = 0, True
            _, _, i1, _, _ = solve_weekly(T, windows, start=ws, origin=start, task_ids=base + [tid],
                                          force_ids=[tid], time_limit=time_limit, workers=workers)
            if i1["objective"] is None:
                continue
            weeks_ok.append(k + 1)
            delta = i1["objective"] - obj0
            proven_all &= bool(p0 and i1["proven_optimal"])
            if best is None or delta > best[1]:
                best = (k + 1, delta)
        if best is None:
            reason, bw, delta = "NO_ELIGIBLE_WINDOW", None, None
        else:
            bw, delta = best
            reason = ("DECLINED_NET_NEGATIVE" if delta < 0 else
                      "INDIFFERENT_TIE" if delta == 0 else "WEEKLY_POOL_EFFECT")
        rows.append(dict(task_id=tid, dept=t.dept, section_id=t.section_id, tier=t.tier,
                         due_idx=int(-t.overdue_days), earliest_idx=int(t.earliest_idx),
                         weeks_possible=",".join(map(str, weeks_ok)) or "-", best_week=bw,
                         delta_if_forced=delta, proven=proven_all, reason=reason))
    return pd.DataFrame(rows)


def lateness_report(plan, T, windows, start=START, horizon_days=28):
    """Timing of the plan against due dates (A7: due dates are reported, not enforced)."""
    ok = ~T.task_id.isin(o.find_unschedulable(T, o.scope(T, windows, horizon_days, start)[0]).task_id)
    cand = T[ok].copy()
    first = plan.groupby("task_id").date.min() if len(plan) else pd.Series(dtype="datetime64[ns]")
    cand["exec_day"] = cand.task_id.map(first).sub(start).dt.days
    cand["due_idx"] = -cand.overdue_days
    ov, nv = cand[cand.overdue_days > 0], cand[cand.overdue_days <= 0]
    sched = cand[cand.exec_day.notna()]
    return dict(
        placeable_tasks=len(cand), scheduled=len(sched),
        overdue_at_start=len(ov), overdue_scheduled=int(ov.exec_day.notna().sum()),
        overdue_mean_exec_day=round(float(ov.exec_day.mean()), 1) if ov.exec_day.notna().any() else None,
        not_overdue_scheduled_after_due=int(((nv.exec_day - nv.due_idx) > 0).sum()),
        due_in_horizon_not_scheduled=int(((nv.due_idx < horizon_days) & nv.exec_day.isna()).sum()),
        not_scheduled_total=int(cand.exec_day.isna().sum()))


UNSCHED_FINAL_COLS = ["task_id", "dept", "section_id", "task", "tier", "group", "reason", "due_idx",
                      "due_after_horizon", "earliest_idx", "initially_assigned_week", "delta_if_forced",
                      "proven", "recommended_action"]


def final_unscheduled(T, windows, plan, assign=None, start=START, n_weeks=N_WEEKS,
                      time_limit=15, workers=8, explain=True):
    """Every task that is NOT scheduled after the final week, with a reason and an action.
    group NEVER_PLACEABLE        no window can ever hold it (needs_extended_block logic)
    group PLACEABLE_NOT_SCHEDULED could be placed, but is not: reason from explain_unscheduled()"""
    horizon = 7 * n_weeks
    never = o.find_unschedulable(T, o.scope(T, windows, horizon, start)[0])
    done = set(plan.task_id) if len(plan) else set()
    init = dict(zip(assign.task_id, assign.week + 1)) if assign is not None and len(assign) else {}
    Ti = T.set_index("task_id")
    rows = []
    for r in never.itertuples():
        t = Ti.loc[r.task_id]
        action = {"TOO_LONG": f"request an extended block (+{r.shortfall_min} min)",
                  "NO_POWER_WINDOW": "ask COA for a power-capable window",
                  "NO_WINDOW_IN_HORIZON": "ask COA for any window on this section"}[r.reason]
        rows.append(dict(task_id=r.task_id, dept=r.dept, section_id=r.section_id, task=r.task, tier=r.tier,
                         group="NEVER_PLACEABLE", reason=r.reason, due_idx=int(-t.overdue_days),
                         due_after_horizon=bool(-t.overdue_days >= horizon), earliest_idx=int(t.earliest_idx),
                         initially_assigned_week=init.get(r.task_id), delta_if_forced=None, proven=None,
                         recommended_action=action))
    ids = [x for x in T.task_id if x not in done and x not in set(never.task_id)]
    if ids:
        ex = (explain_unscheduled(T, windows, plan, ids, start, n_weeks, time_limit, workers) if explain
              else pd.DataFrame(dict(task_id=ids)).assign(reason="NOT_EXPLAINED", delta_if_forced=None, proven=None))
        for r in ex.itertuples():
            t = Ti.loc[r.task_id]
            due = int(-t.overdue_days)
            if r.reason in ("DECLINED_NET_NEGATIVE", "INDIFFERENT_TIE", "NOT_EXPLAINED"):
                action = (f"carry to the next planning cycle (due day {due})" if due >= horizon
                          else "REVIEW: due inside the horizon but not worth a block under the objective")
            elif r.reason == "NOT_YET_ELIGIBLE":
                action = f"carry to the next planning cycle (earliest start day {int(t.earliest_idx)})"
            elif r.reason == "NO_ELIGIBLE_WINDOW":
                action = "no usable window after its earliest start: carry over or ask COA for a window"
            else:
                action = "re-run with a larger pool: placing it would raise the week's value"
            rows.append(dict(task_id=r.task_id, dept=t.dept, section_id=t.section_id, task=t.task, tier=t.tier,
                             group="PLACEABLE_NOT_SCHEDULED", reason=r.reason, due_idx=due,
                             due_after_horizon=bool(due >= horizon), earliest_idx=int(t.earliest_idx),
                             initially_assigned_week=init.get(r.task_id), delta_if_forced=r.delta_if_forced,
                             proven=r.proven, recommended_action=action))
    return pd.DataFrame(rows, columns=UNSCHED_FINAL_COLS)


def print_unscheduled(u, n_tasks, n_scheduled):
    print(f"\nUNSCHEDULED AFTER THE FINAL WEEK: {len(u)} of {n_tasks} tasks "
          f"(scheduled {n_scheduled} + unscheduled {len(u)} = {n_scheduled + len(u)})")
    if u.empty:
        return
    print(u.groupby(["group", "reason", "tier"]).size().rename("tasks").to_string())
    pl = u[u.group == "PLACEABLE_NOT_SCHEDULED"]
    if len(pl):
        print(f"  placeable-but-unscheduled: {len(pl)}; due after the horizon: {int(pl.due_after_horizon.sum())}; "
              f"due INSIDE the horizon: {int((~pl.due_after_horizon).sum())}; "
              f"all explanations proven: {bool(pl.proven.fillna(False).all())}")
        d = pl.delta_if_forced.dropna()
        if len(d):
            print(f"  net-value change if each were forced in: min {d.min():.0f}, max {d.max():.0f}")


def capacity_diagnosis(T, windows, plan, blocks, start=START, n_weeks=N_WEEKS):
    """Explain unused capacity without treating every task as eligible for every window.

    free_window_min is time in windows not selected as blocks. Individually feasible tasks
    are counted only if at least one window in that week matches section, earliest-start,
    duration/setup, and power requirements. This is a per-task diagnostic, not a guarantee
    that all eligible tasks can be scheduled together.
    """
    never = set(o.find_unschedulable(T, o.scope(T, windows, 7 * n_weeks, start)[0]).task_id)
    P = T[~T.task_id.isin(never)].copy()
    first = plan.groupby("task_id").week.min() if len(plan) else pd.Series(dtype=int)
    fc = free_capacity(windows, blocks, start, n_weeks).reindex(range(1, n_weeks + 1), fill_value=0)
    rows = []
    for k in range(1, n_weeks + 1):
        before = set(first[first < k].index) if len(first) else set()
        remaining = P[~P.task_id.isin(before)]
        week_start = start + pd.Timedelta(days=7 * (k - 1))
        week_end = week_start + pd.Timedelta(days=7)
        Wk = windows[(windows.date >= week_start) & (windows.date < week_end)].copy()
        eligible_ids = []
        for t in remaining.itertuples():
            eligible = Wk[(Wk.section_id == t.section_id)
                          & (Wk.date >= start + pd.Timedelta(days=int(t.earliest_idx)))
                          & (Wk.length_min >= int(t.need_min))]
            if o.needs_power(t.block_type):
                eligible = eligible[eligible.power_block_possible]
            if len(eligible):
                eligible_ids.append(t.task_id)
        elig = remaining[remaining.task_id.isin(eligible_ids)]
        sched = int((first == k).sum()) if len(first) else 0
        wb = blocks[blocks.week == k] if len(blocks) else blocks
        crew = float(wb.crew_total_min.sum()) if len(wb) else 0.0
        work = int(wb.work_min.sum()) if len(wb) else 0              # block table has work_min / crew_total_min
        window_min = int(fc.window_min[k])
        rows.append(dict(
            week=k,
            window_count=len(Wk),
            window_min=window_min,
            selected_block_min=int(wb.length_min.sum()) if len(wb) else 0,
            free_window_min=int(fc.free_min[k]),
            free_pct=float(fc.free_pct[k]),
            blocks=len(wb),
            eligible_at_week_end=len(elig),
            eligible_work_min=int(elig.need_min.sum()),
            new_supply=int((P.earliest_idx.clip(lower=0) // 7 == k - 1).sum()),
            scheduled=sched,
            left_for_later=max(0, len(elig) - sched),
            crew_work_min=work,
            crew_setup_min=int(crew - work),
            crew_util_pct=round(100 * crew / max(1, window_min * 3), 1),
        ))
    return pd.DataFrame(rows)


def workload_profile(plan, blocks, n_weeks=N_WEEKS):
    rows = []
    for k in range(1, n_weeks + 1):
        p = plan[plan.week == k] if len(plan) else plan
        b = blocks[blocks.week == k] if len(blocks) else blocks
        rows.append(dict(week=k, tasks=p.task_id.nunique() if len(p) else 0,
                         work_min=int(p.duration_min.sum()) if len(p) else 0, blocks=len(b),
                         merged_blocks=int(b.merged.sum()) if len(b) else 0,
                         trains=int(b.trains_affected.sum()) if len(b) else 0))
    df = pd.DataFrame(rows)
    tot = max(1, df.tasks.sum())
    return df, dict(week1_tasks=int(df.tasks[0]), week1_share_pct=round(float(100 * df.tasks[0] / tot), 1),
                    peak_to_mean_work=round(float(df.work_min.max() / max(1.0, df.work_min.mean())), 2))


def priority_check(plan, T, windows, start=START, horizon_days=28):
    """Are critical/overdue tasks still done first? Execution day = first day of the task's block."""
    never = set(o.find_unschedulable(T, o.scope(T, windows, horizon_days, start)[0]).task_id)
    P = T[~T.task_id.isin(never)].copy()
    first = plan.groupby("task_id").date.min().sub(start).dt.days if len(plan) else pd.Series(dtype=int)
    P["exec_day"] = P.task_id.map(first)
    ch = P[P.tier.isin(["CRITICAL", "HIGH"])]
    ov = P[P.overdue_days > 0]
    nv = P[(P.overdue_days <= 0) & P.exec_day.notna()]
    mean = lambda d: round(float(d.exec_day.mean()), 2) if d.exec_day.notna().any() else None
    return dict(crit_high=f"{int(ch.exec_day.notna().sum())}/{len(ch)}", crit_high_mean_day=mean(ch),
                overdue=f"{int(ov.exec_day.notna().sum())}/{len(ov)}", overdue_mean_day=mean(ov),
                newly_late=int(((nv.exec_day - (-nv.overdue_days)) > 0).sum()))


def compare_variants(T, windows, a, balance_pts):
    """Same data, same weekly model and objective. Only the monthly stage / carry-over differs."""
    variants = [("v1: simple monthly, static carry-over", dict(model="simple"), False),
                ("blocks monthly, static carry-over", dict(model="blocks"), False),
                ("blocks monthly + weekly re-plan", dict(model="blocks"), True),
                ("blocks + re-plan + deferral & balancing", dict(model="blocks", defer=True, balance=balance_pts), True),
                ("... + due backfill (= default)", dict(model="blocks", defer=True, balance=balance_pts), True)]
    rows = []
    for vi, (name, kw, rp) in enumerate(variants):
        t0 = time.time()
        assign, minfo = solve_monthly(T, windows, n_weeks=a.weeks, reserve=a.reserve,
                                      time_limit=a.time_limit, workers=a.workers, **kw)
        plan, blocks, summ, done = run_horizon(T, windows, assign, n_weeks=a.weeks, time_limit=a.time_limit,
                                               workers=a.workers, replan=dict(reserve=a.reserve, **kw) if rp else None,
                                               backfill=(vi == len(variants) - 1))
        errs = validate_parallel(plan, windows, T, horizon_days=7 * a.weeks)
        never = o.find_unschedulable(T, o.scope(T, windows, 7 * a.weeks, START)[0])
        real = realization_summary(assign, plan)
        _, prof = workload_profile(plan, blocks, a.weeks)
        pc = priority_check(plan, T, windows, horizon_days=7 * a.weeks)
        rows.append(dict(variant=name, monthly_assigned=len(assign), scheduled=plan.task_id.nunique(),
                         placeable_unscheduled=len(T) - len(never) - plan.task_id.nunique(),
                         assigned_not_scheduled=real["not_scheduled"], on_time_pct=real["realization_pct"],
                         week1_tasks=prof["week1_tasks"], week1_share_pct=prof["week1_share_pct"],
                         peak_to_mean_work=prof["peak_to_mean_work"], blocks=len(blocks),
                         merged_blocks=int(blocks.merged.sum()), trains=int(blocks.trains_affected.sum()),
                         net_value=int(summ.net_value.sum()), crit_high=pc["crit_high"],
                         crit_high_mean_day=pc["crit_high_mean_day"], overdue_mean_day=pc["overdue_mean_day"],
                         newly_late=pc["newly_late"], violations=len(errs), runtime_s=round(time.time() - t0, 1)))
    return pd.DataFrame(rows)


def objective_audit(plan, blocks, T, start=START, summ=None):
    """Decompose the weekly objective into assignment reward minus block costs (same integer
    formulas as the solver). With `summ` (run_horizon summary) the recomputed value is compared with
    the objective the CP-SAT solver itself reported for that week (column solver_objective)."""
    rows = []
    Ti = T.set_index("task_id")
    weeks = sorted(set(plan.week) | set(blocks.week)) if len(plan) or len(blocks) else []
    solver = {} if summ is None or "solver_objective" not in summ else dict(zip(summ.week, summ.solver_objective))
    for k in weeks:
        pk = plan[plan.week == k] if len(plan) else plan
        bk = blocks[blocks.week == k] if len(blocks) else blocks
        reward = 0
        for r in pk.itertuples():
            t = Ti.loc[r.task_id]
            day = int((pd.Timestamp(r.date) - start).days)
            reward += o.assignment_value(t.priority_score, t.severity, t.overdue_days, day)
        cost = sum(o.block_cost(r.trains_affected, r.goods_trains_affected) for r in bk.itertuples())
        net = int(reward - cost)
        ref = solver.get(k, int(o.net_value(pk, bk, start)) if len(pk) else 0)
        rows.append(dict(week=int(k), assignment_reward=int(reward), block_cost=int(cost),
                         recomputed_net_value=net, reference_value=None if ref is None else int(ref),
                         reference="solver objective" if k in solver else "net_value()",
                         matches=(ref is not None and net == int(ref))))
    return pd.DataFrame(rows)


def gap_line(info):
    if info["objective"] is None:
        return f"{info['status']} - no feasible plan"
    tag = "OPTIMAL (proven)" if info["proven_optimal"] else "FEASIBLE, NOT proven optimal"
    return (f"{tag} | objective {info['objective']} | best bound {info['bound']:.1f} | "
            f"gap {info['abs_gap']:.1f} ({info['rel_gap']:.2%})")


def ablation_week1(T, windows, raw_tasks, time_limit, workers):
    """Same data, same objective; only the merging rule differs. The Step 3 row uses a
    DIFFERENT model (sequential, setup per task) and is a reference only."""
    rows = []
    for label, merge in (("Step 4 coordinated (parallel, shared setup)", True),
                         ("Step 4, one department per block", False)):
        plan, blocks, info, _, Tk = solve_weekly(T, windows, merge=merge, time_limit=time_limit, workers=workers)
        errs = validate_parallel(plan, windows, T, merge=merge)
        rows.append(dict(model=label, tasks=plan.task_id.nunique(), blocks=len(blocks),
                         merged_blocks=int(blocks.merged.sum()), trains_disrupted=int(blocks.trains_affected.sum()),
                         net_value=o.net_value(plan, blocks), bound=round(info["bound"], 1),
                         rel_gap=f"{info['rel_gap']:.2%}", proven=info["proven_optimal"], violations=len(errs)))
    plan, blocks, info, _, Ts = o.solve(raw_tasks, windows, time_limit=time_limit, workers=workers)
    errs = o.validate(plan, blocks, windows, raw_tasks)
    rows.append(dict(model="Step 3 sequential (reference, different model)", tasks=plan.task_id.nunique(),
                     blocks=len(blocks), merged_blocks=int((blocks.depts.str.contains(r"\+")).sum()),
                     trains_disrupted=int(blocks.trains_affected.sum()), net_value=o.net_value(plan, blocks),
                     bound=round(info["bound"], 1), rel_gap=f"{info['rel_gap']:.2%}",
                     proven=info["proven_optimal"], violations=len(errs)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=30, help="per weekly / monthly solve (s)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reserve", type=float, default=EMERGENCY_RESERVE)
    ap.add_argument("--weeks", type=int, default=N_WEEKS)
    ap.add_argument("--early-days", type=int, default=EARLY_DAYS)
    ap.add_argument("--monthly-model", choices=["simple", "blocks"], default="blocks")
    ap.add_argument("--no-replan", action="store_true", help="static carry-over instead of weekly re-planning")
    ap.add_argument("--no-defer", action="store_true", help="do not defer non-critical work (B11)")
    ap.add_argument("--balance", type=int, default=DEFAULT_BALANCE,
                    help="B12: points per 10 min of weekly load above the soft cap (0 = off)")
    ap.add_argument("--balance-cap", type=float, default=BALANCE_CAP,
                    help="weekly workload soft cap as a multiple of the average weekly workload (default 1.25)")
    ap.add_argument("--no-backfill", action="store_true", help="do not offer due-by-week-end tasks to the weekly solve (B15)")
    ap.add_argument("--no-explain", action="store_true", help="skip the forced re-solves that explain unscheduled tasks")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--compare-variants", action="store_true", help="run v1 / blocks / re-plan / deferral side by side")
    ap.add_argument("--compare-myopic", action="store_true")
    a = ap.parse_args()

    raw, windows = o.load()
    T = prepare_tasks(raw, a.early_days)
    OUT.mkdir(exist_ok=True)
    horizon = a.weeks * 7
    if a.balance_cap <= 0:
        ap.error("--balance-cap must be greater than 0")
    if a.balance < 0:
        ap.error("--balance must be >= 0")
    mkw = dict(model=a.monthly_model, defer=not a.no_defer, balance=a.balance,
               balance_cap=a.balance_cap, reserve=a.reserve)
    if a.monthly_model == "simple":
        mkw.update(defer=False, balance=0)

    print(f"Tasks {len(T)} | safety pairs closer than {SAFE_KM} km (same section): {len(conflict_pairs(T))}")
    print(f"settings: monthly={mkw} | re-plan={'off' if a.no_replan else 'on'} | early window {a.early_days} d")

    assign, minfo = solve_monthly(T, windows, n_weeks=a.weeks, time_limit=a.time_limit, workers=a.workers, **mkw)
    print(f"\n=== MONTHLY plan (tasks -> weeks, reserve {a.reserve:.0%}) ===\n{gap_line(minfo)} "
          f"| {minfo['n_vars']} vars | {minfo['wall_time_s']}s")
    print("tasks per week:", assign.groupby(assign.week + 1).size().to_dict())
    assign.to_csv(OUT / "monthly_plan.csv", index=False)

    plan, blocks, summ, done = run_horizon(T, windows, assign, n_weeks=a.weeks, time_limit=a.time_limit,
                                           workers=a.workers, replan=None if a.no_replan else mkw,
                                           backfill=not a.no_backfill)
    print("\n=== WEEKLY refinement ===")
    print(summ.to_string(index=False))
    errs = validate_parallel(plan, windows, T, horizon_days=horizon)
    print(f"\nwhole-horizon validation: {errs[:3] or 'all hard constraints OK'}")
    print("monthly vs weekly realization:", realization_summary(assign, plan))
    wp, prof = workload_profile(plan, blocks, a.weeks)
    print("\nworkload per week:\n" + wp.to_string(index=False))
    print("concentration:", prof, "\npriorities:", priority_check(plan, T, windows, horizon_days=horizon))
    print("total: scheduled", plan.task_id.nunique(), "of", len(T), "| blocks", len(blocks),
          "| merged", int(blocks.merged.sum()), "| trains", int(blocks.trains_affected.sum()),
          "| net value", int(summ.net_value.sum()))
    audit = objective_audit(plan, blocks, T, summ=summ)
    print("\nweekly objective audit (assignment reward - block cost, vs the solver's own objective):\n"
          + audit.to_string(index=False))
    if len(audit) and not audit.matches.all():
        print("WARNING: objective audit does not match the solver objective; inspect the objective definitions.")
    audit.to_csv(OUT / "objective_audit.csv", index=False)

    cd = capacity_diagnosis(T, windows, plan, blocks, n_weeks=a.weeks)
    print("\ncapacity diagnosis (eligible = at least one individually compatible window that week):\n"
          + cd.to_string(index=False))

    un = final_unscheduled(T, windows, plan, assign, n_weeks=a.weeks, time_limit=min(a.time_limit, 15),
                           workers=a.workers, explain=not a.no_explain)
    print_unscheduled(un, len(T), plan.task_id.nunique())
    un[un.group == "NEVER_PLACEABLE"].to_csv(OUT / "needs_extended_block_4w.csv", index=False)
    un.to_csv(OUT / "unscheduled_final.csv", index=False)
    plan.to_csv(OUT / "block_plan_4w.csv", index=False)
    blocks.to_csv(OUT / "block_summary_4w.csv", index=False)
    summ.to_csv(OUT / "horizon_summary.csv", index=False)
    realization(assign, plan).to_csv(OUT / "monthly_realization.csv", index=False)
    cd.to_csv(OUT / "capacity_diagnosis.csv", index=False)
    wp.to_csv(OUT / "workload_profile.csv", index=False)

    if a.compare_variants:
        cv = compare_variants(T, windows, a, a.balance or DEFAULT_BALANCE)
        print("\n=== monthly / carry-over variants (same data, same weekly model and objective) ===")
        print(cv.T.to_string())
        cv.to_csv(OUT / "variant_comparison.csv", index=False)

    if a.compare_myopic:
        p2, b2, s2, _ = run_horizon(T, windows, None, n_weeks=a.weeks, time_limit=a.time_limit, workers=a.workers)
        print("\n=== reference: weekly only, no monthly plan ===")
        print(s2.to_string(index=False))
        print(f"scheduled {p2.task_id.nunique()} | blocks {len(b2)} | trains {int(b2.trains_affected.sum())} "
              f"| net value {int(s2.net_value.sum())}  (this plan: {plan.task_id.nunique()} | "
              f"{len(blocks)} | {int(blocks.trains_affected.sum())} | {int(summ.net_value.sum())})")

    if not a.no_ablation:
        ab = ablation_week1(T, windows, raw, a.time_limit, a.workers)
        print("\n=== week 1: effect of merging (same data and objective) ===")
        print(ab.to_string(index=False))
        ab.to_csv(OUT / "merge_ablation_week1.csv", index=False)


if __name__ == "__main__":
    main()
