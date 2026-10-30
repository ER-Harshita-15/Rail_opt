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

Monthly model (CP-SAT, coarse)
    z[i,k] task i in week k | v[s,k] section s is visited in week k
    per (section, week, crew, line/power) task minutes <= (1 - EMERGENCY_RESERVE) * capacity
    per (week, crew) network task minutes <= (1 - EMERGENCY_RESERVE) * pool size * covered minutes
    objective  sum assignment_value(week start) - sum visit_cost * v[s,k]
    -> work is grouped into the same section-week (shared blocks) when the delay it causes
       is cheaper than another visit.

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
    B6  Monthly model is an approximation: setup is charged per task (conservative), the
        capacity of a section-week is the sum over days of its 2 longest eligible windows,
        safety conflicts and exact gang timing are not visible at this level, and a visit
        costs the cheapest window of that section-week. EMERGENCY_RESERVE = 20% is assumed.
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

Run:  python horizons.py [--time-limit 30] [--workers 8] [--reserve 0.2] [--weeks 4]
                         [--no-ablation] [--compare-myopic]
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
                 section=None, merge=True, time_limit=30, workers=8, rel_gap_limit=0.0, seed=1):
    """`tasks` must come from prepare_tasks(). `origin` = day 0 of the whole plan (B8)."""
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
def solve_monthly(T, windows, start=START, n_weeks=N_WEEKS, reserve=EMERGENCY_RESERVE,
                  time_limit=30, workers=8, seed=1):
    end = start + pd.Timedelta(days=7 * n_weeks)
    Wm = windows[(windows.date >= start) & (windows.date < end)].copy()
    Wm["week"] = ((Wm.date - start).dt.days // 7).astype(int)

    cap, maxlen, visit, cover = defaultdict(int), {}, {}, defaultdict(int)
    for (sec, wk, _), g in Wm.groupby(["section_id", "week", "date"]):
        for kind, gg in (("any", g), ("power", g[g.power_block_possible])):
            lens = sorted(gg.length_min, reverse=True)[:o.MAX_BLOCKS_PER_DAY]
            cap[(sec, wk, kind)] += int(sum(lens))
    for (sec, wk), g in Wm.groupby(["section_id", "week"]):
        maxlen[(sec, wk, "any")] = int(g.length_min.max())
        p = g[g.power_block_possible]
        maxlen[(sec, wk, "power")] = int(p.length_min.max()) if len(p) else 0
        visit[(sec, wk)] = min(o.block_cost(r.trains_affected, r.goods_trains_affected)
                               for r in g.itertuples())
    for (wk, _), g in Wm.groupby(["week", "date"]):                     # minutes covered by any window
        hrs = set()
        for r in g.itertuples():
            hrs |= set(range(int(r.start_hour), int(r.end_hour)))
        cover[wk] += 60 * len(hrs)

    m = cp_model.CpModel()
    z, grp = {}, defaultdict(list)
    for i, t in T.iterrows():
        kind = "power" if o.needs_power(t.block_type) else "any"
        for k in range(n_weeks):
            late_enough = 7 * k + 6 >= t.get("earliest_idx", -10**6)    # B10: week must end after earliest start
            if late_enough and maxlen.get((t.section_id, k, kind), 0) >= t.need_min:
                z[i, k] = m.NewBoolVar(f"z{i}_{k}")
    v = {sk: m.NewBoolVar(f"v{sk}") for sk in visit}
    by_task = defaultdict(list)
    for (i, k) in z:
        by_task[i].append(k)
    for i, ks in by_task.items():
        m.Add(sum(z[i, k] for k in ks) <= 1)
    for (i, k), var in z.items():
        t = T.loc[i]
        m.Add(var <= v[(t.section_id, k)])
        kind = "power" if o.needs_power(t.block_type) else "any"
        grp[(t.section_id, k, t.gang_type, kind)].append(i)
    for (sec, k, g, kind), ids in grp.items():                          # section-week crew capacity
        m.Add(sum(int(T.need_min[i]) * z[i, k] for i in ids) <= int((1 - reserve) * cap[(sec, k, kind)]))
    for k in range(n_weeks):                                            # network crew pool per week
        for g, n in o.GANGS.items():
            ids = [i for (i, kk) in z if kk == k and T.gang_type[i] == g]
            if ids:
                m.Add(sum(int(T.need_min[i]) * z[i, k] for i in ids) <= int((1 - reserve) * n * cover[k]))
    obj = [o.assignment_value(T.priority_score[i], T.severity[i], T.overdue_days[i], 7 * k) * var
           for (i, k), var in z.items()]
    obj += [-visit[sk] * var for sk, var in v.items()]
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
                n_vars=len(z) + len(v), reserve=reserve, objective=None, bound=None, abs_gap=None, rel_gap=None)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return pd.DataFrame(columns=["task_id", "week"]), info
    info["objective"] = int(round(s.ObjectiveValue()))
    info["bound"] = s.BestObjectiveBound()
    info["abs_gap"], info["rel_gap"] = o.compute_gap(info["objective"], info["bound"])
    rows = [dict(task_id=T.task_id[i], week=k) for (i, k), var in z.items() if s.Value(var)]
    assign = pd.DataFrame(rows, columns=["task_id", "week"])
    assign = assign.merge(T[["task_id", "dept", "section_id", "task", "tier", "need_min"]], on="task_id")
    return assign.sort_values(["week", "section_id", "task_id"]).reset_index(drop=True), info


# --------------------------------------------------------------------------
# Rolling pipeline: monthly -> weekly refinement with carry-over
# --------------------------------------------------------------------------
def run_horizon(T, windows, assign=None, start=START, n_weeks=N_WEEKS, time_limit=30, workers=8,
                label="monthly->weekly"):
    """assign=None -> 'myopic' mode: every week sees all tasks still unscheduled."""
    done, carry, plans, blocks_all, rows = set(), [], [], [], []
    for k in range(n_weeks):
        ws = start + pd.Timedelta(days=7 * k)
        if assign is None:
            pool = [t for t in T.task_id if t not in done]
        else:
            pool = list(assign[assign.week == k].task_id) + carry
        plan, blocks, info, W, Tk = solve_weekly(T, windows, start=ws, origin=start, task_ids=pool,
                                                 time_limit=time_limit, workers=workers)
        errs = validate_parallel(plan, windows, T, start=ws, origin=start)
        if info["objective"] is not None and len(plan):
            nv = o.net_value(plan, blocks, start)
            if nv != info["objective"]:
                raise AssertionError(f"week {k + 1}: solver objective {info['objective']} != net_value {nv}")
        got = set(plan.task_id) if len(plan) else set()
        carry = [t for t in pool if t not in got] if assign is not None else []
        done |= got
        if len(plan):
            plans.append(plan.assign(week=k + 1))
            blocks_all.append(blocks.assign(week=k + 1))
        rows.append(dict(week=k + 1, status=info["status"], pool=len(pool), scheduled=len(got),
                         carried_over=len(carry), blocks=len(blocks),
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
    return plan, blocks, pd.DataFrame(rows), done


def realization(assign, plan):
    """How closely did the weekly plans follow the monthly plan?"""
    first = plan.groupby("task_id").week.min() if len(plan) else pd.Series(dtype=int)
    r = assign.copy()
    r["realized_week"] = r.task_id.map(first)
    r["status"] = "NOT_SCHEDULED"
    r.loc[r.realized_week == r.week + 1, "status"] = "ON_TIME"
    r.loc[r.realized_week > r.week + 1, "status"] = "SLIPPED"
    return r


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
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--compare-myopic", action="store_true")
    a = ap.parse_args()

    raw, windows = o.load()
    T = prepare_tasks(raw, a.early_days)
    OUT.mkdir(exist_ok=True)
    horizon = a.weeks * 7
    unsched = o.find_unschedulable(T, o.scope(T, windows, horizon, START)[0])
    unsched.assign(horizon_start=START.date(), horizon_days=horizon).to_csv(
        OUT / "needs_extended_block_4w.csv", index=False)

    print(f"Tasks {len(T)} | cannot be placed in {horizon} days: {len(unsched)} "
          f"({unsched.reason.value_counts().to_dict()}) -> needs_extended_block_4w.csv")
    print(f"S&T/Traction safety pairs closer than {SAFE_KM} km (same section): {len(conflict_pairs(T))}")

    # ---- monthly plan ----
    assign, minfo = solve_monthly(T, windows, n_weeks=a.weeks, reserve=a.reserve,
                                  time_limit=a.time_limit, workers=a.workers)
    print(f"\n=== MONTHLY plan (tasks -> weeks, reserve {a.reserve:.0%}) ===\n{gap_line(minfo)} "
          f"| {minfo['n_vars']} vars | {minfo['wall_time_s']}s")
    print("tasks per week:", assign.groupby(assign.week + 1).size().to_dict())
    assign.to_csv(OUT / "monthly_plan.csv", index=False)

    # ---- weekly refinement ----
    plan, blocks, summ, done = run_horizon(T, windows, assign, n_weeks=a.weeks,
                                           time_limit=a.time_limit, workers=a.workers)
    print("\n=== WEEKLY refinement of the monthly plan ===")
    print(summ.to_string(index=False))
    errs = validate_parallel(plan, windows, T, horizon_days=horizon)
    print(f"\nwhole-horizon validation: {errs[:3] or 'all hard constraints OK'}")
    real = realization(assign, plan)
    print("monthly vs weekly:", real.status.value_counts().to_dict())
    print("scheduled in total:", plan.task_id.nunique(), "of", len(T),
          "| merged blocks:", int(blocks.merged.sum()), "of", len(blocks),
          "| trains disrupted:", int(blocks.trains_affected.sum()),
          "| total net value:", int(summ.net_value.sum()))
    print("timing vs due dates:", lateness_report(plan, T, windows, horizon_days=horizon))
    print("\nfree window minutes (practical emergency reserve):")
    print(free_capacity(windows, blocks, n_weeks=a.weeks).to_string())
    plan.to_csv(OUT / "block_plan_4w.csv", index=False)
    blocks.to_csv(OUT / "block_summary_4w.csv", index=False)
    summ.to_csv(OUT / "horizon_summary.csv", index=False)
    real.to_csv(OUT / "monthly_realization.csv", index=False)

    if a.compare_myopic:
        p2, b2, s2, _ = run_horizon(T, windows, None, n_weeks=a.weeks, time_limit=a.time_limit,
                                    workers=a.workers)
        print("\n=== reference: weekly only, no monthly plan (every week sees all open tasks) ===")
        print(s2.to_string(index=False))
        print(f"scheduled {p2.task_id.nunique()} | blocks {len(b2)} | trains {int(b2.trains_affected.sum())} "
              f"| net value {int(s2.net_value.sum())}  (monthly->weekly: {plan.task_id.nunique()} | "
              f"{len(blocks)} | {int(blocks.trains_affected.sum())} | {int(summ.net_value.sum())})")

    if not a.no_ablation:
        ab = ablation_week1(T, windows, raw, a.time_limit, a.workers)
        print("\n=== week 1: effect of merging (same data and objective) ===")
        print(ab.to_string(index=False))
        ab.to_csv(OUT / "merge_ablation_week1.csv", index=False)


if __name__ == "__main__":
    main()
