"""
Step 3 (revised) - CP-SAT block scheduler for RailOpt
Assigns prioritized maintenance tasks to COA corridor windows.

Decision variables
    x[i,w]  task i is done in window w          (only for eligible pairs)
    y[w]    window w is actually taken as a block (only if >= 1 task is in it)
    u[w,g]  gang type g is reserved for window w

Hard constraints (identical for CP-SAT and both greedy baselines)
    1. each task is scheduled at most once
    2. a task needs a window on ITS section, inside the planning horizon
    3. sum(duration + setup) of tasks in a window <= window length
    4. POWER / LINE+POWER tasks only in windows where a power block is possible
    5. at most MAX_BLOCKS_PER_DAY blocks per section per day
    6. gang availability: at every (date, hour), gangs of type g reserved <= GANGS[g]

Objective (maximise) - ONE definition, shared by the solver, net_value() and the baselines
    value = sum_i  assignment_value(i, day)  -  sum_w  block_cost(w) * y[w]
    assignment_value = round(1000 * priority_score) - delay_penalty      (integer)
    delay_penalty    = day_idx * (3 + 4*severity) * (3 if overdue else 1) // 2
    block_cost       = W_FIXED + W_TRAIN * trains + W_GOODS * goods trains   (integer)

ASSUMPTIONS (all synthetic / modelling choices - state them in the report)
    A1  A block is one COA window taken whole. Tasks inside a block run one after
        another (sequential), so capacity = window length. Parallel work by several
        departments in the same block is NOT modelled (Step 4).
    A2  Setup time is charged per task (not shared inside a block). Shared setup is Step 4.
    A3  A task is done entirely inside ONE window (no splitting across windows/days).
        A task longer than every eligible window is never scheduled; it is reported
        in needs_extended_block.csv with the reason and the shortfall.
    A4  Gang reservation: if any task in a window needs gang type g, ONE gang of
        type g is reserved for the WHOLE window, in every hour it covers, network-wide.
        This is conservative (a gang may idle part of the window). Pool sizes in GANGS
        are assumed numbers. gang_size (people) is not used. No travel time between
        windows; back-to-back windows (13:00 end / 13:00 start) do not overlap.
    A5  LINE tasks can use any window; POWER tasks need power_block_possible. A power
        block is assumed not to conflict with line-block work in another window.
    A6  Disruption is measured by trains_affected / goods_trains_affected stored in the
        COA windows file (synthetic here). Same numbers for every method.
    A7  Due dates are not hard deadlines: urgency enters only through priority_score
        and the delay penalty. Leaving a task unscheduled has no explicit penalty
        beyond the value it would have earned.
    A8  Objective weights are expert-chosen, not calibrated. "Value" is in arbitrary
        points; compare plans only with the same weights.
    A9  S&T-vs-Traction safety conflicts are not needed while work is sequential;
        they become a hard constraint in Step 4 with parallel scheduling.
    A10 With workers > 1 CP-SAT is not deterministic: repeated runs can return
        slightly different feasible plans. Use --workers 1 for reproducible runs.

Run:  python optimizer.py [--horizon-days 7] [--time-limit 60] [--workers 8]
                          [--rel-gap 0.0] [--section TLD-SKR]
"""
import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "DATA", ROOT / "outputs"

MAX_BLOCKS_PER_DAY = 2
GANGS = {"track_gang": 3, "signal_team": 3, "ohe_tower_wagon": 2}   # per network (A4)
W_FIXED, W_TRAIN, W_GOODS = 60, 10, 5                              # block cost weights
START = pd.Timestamp("2026-10-26")

UNSCHED_COLS = ["task_id", "dept", "section_id", "task", "tier", "need_min",
                "longest_eligible_window_min", "shortfall_min", "reason"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load():
    tasks = pd.read_csv(DATA / "tasks_common_schema.csv")
    pri = pd.read_csv(OUT / "prioritized_tasks.csv")[
        ["task_id", "priority_score", "priority_rank", "tier"]]
    tasks = tasks.merge(pri, on="task_id")
    tasks["need_min"] = tasks.duration_min + tasks.setup_min
    windows = pd.read_csv(DATA / "coa_corridor_windows.csv", parse_dates=["date"])
    return tasks, windows


def needs_power(block_type):
    return block_type in ("POWER", "LINE+POWER")


def scope(tasks, windows, horizon_days=7, start=START, section=None):
    """Windows/tasks that belong to one planning run. Used by CP-SAT AND the greedy
    baselines, so every method sees exactly the same data."""
    end = start + pd.Timedelta(days=horizon_days)
    W = windows[(windows.date >= start) & (windows.date < end)]
    T = tasks
    if section:
        W, T = W[W.section_id == section], T[T.section_id == section]
    return W.reset_index(drop=True), T.reset_index(drop=True)


def candidate_pairs(T, W):
    """Eligible (task index, window index) pairs: constraints 2, 3-per-task and 4."""
    pairs = []
    for i, t in T.iterrows():
        ok = W[(W.section_id == t.section_id) & (W.length_min >= t.need_min)]
        if needs_power(t.block_type):
            ok = ok[ok.power_block_possible]
        pairs += [(i, w) for w in ok.index]
    return pairs


def find_unschedulable(T, W):
    """Tasks that can never be placed in this horizon, with an explicit reason.
    TOO_LONG              need_min exceeds the longest eligible window -> needs an extended block
    NO_POWER_WINDOW       power task, but no power-capable window on its section
    NO_WINDOW_IN_HORIZON  the section has no window at all in the horizon"""
    rows = []
    for _, t in T.iterrows():
        sec = W[W.section_id == t.section_id]
        elig = sec[sec.power_block_possible] if needs_power(t.block_type) else sec
        if sec.empty:
            reason, longest = "NO_WINDOW_IN_HORIZON", 0
        elif elig.empty:
            reason, longest = "NO_POWER_WINDOW", 0
        else:
            longest = int(elig.length_min.max())
            reason = "TOO_LONG" if t.need_min > longest else None
        if reason:
            rows.append(dict(task_id=t.task_id, dept=t.dept, section_id=t.section_id,
                             task=t.task, tier=t.tier, need_min=int(t.need_min),
                             longest_eligible_window_min=longest,
                             shortfall_min=int(t.need_min - longest) if reason == "TOO_LONG" else None,
                             reason=reason))
    return pd.DataFrame(rows, columns=UNSCHED_COLS)


# --------------------------------------------------------------------------
# ONE objective definition (integer arithmetic) - used by solver, baselines, net_value
# --------------------------------------------------------------------------
def assignment_value(priority_score, severity, overdue_days, day_idx):
    delay = (int(day_idx) * (3 + 4 * int(severity)) * (3 if overdue_days > 0 else 1)) // 2
    return int(round(1000 * float(priority_score))) - delay


def block_cost(trains_affected, goods_trains_affected):
    return int(W_FIXED + W_TRAIN * int(trains_affected) + W_GOODS * int(goods_trains_affected))


def gain_and_cost(plan, blocks, start=START):
    if plan is None or len(plan) == 0:
        return 0, 0
    gain = sum(assignment_value(r.priority_score, r.severity, r.overdue_days,
                                (r.date - start).days) for r in plan.itertuples())
    cost = sum(block_cost(b.trains_affected, b.goods_trains_affected) for b in blocks.itertuples())
    return int(gain), int(cost)


def net_value(plan, blocks, start=START):
    """The exact quantity the CP-SAT objective maximises."""
    g, c = gain_and_cost(plan, blocks, start)
    return g - c


def compute_gap(objective, bound):
    """Maximisation: bound >= objective. Absolute gap = bound - objective.
    Relative gap uses CP-SAT's definition |bound - objective| / max(1, |objective|)."""
    abs_gap = bound - objective
    return abs_gap, abs(abs_gap) / max(1.0, abs(objective))


# --------------------------------------------------------------------------
# Plan post-processing
# --------------------------------------------------------------------------
TASK_COLS = ["task_id", "dept", "task", "tier", "priority_score", "need_min",
             "gang_type", "block_type", "overdue_days", "severity"]
BLOCK_COLS = ["window_id", "section_id", "date", "win_start", "win_end", "n_tasks", "used_min",
              "depts", "trains_affected", "goods_trains_affected", "length_min", "utilization"]


def finish_plan(plan, T, W):
    """Attach task attributes, sequence tasks inside each window, build block summary."""
    if plan is None or len(plan) == 0:
        return pd.DataFrame(), pd.DataFrame(columns=BLOCK_COLS)
    plan = plan.merge(T[TASK_COLS], on="task_id")
    plan = plan.sort_values(["date", "window_id", "priority_score", "task_id"],
                            ascending=[True, True, False, True])
    plan["offset"] = plan.groupby("window_id").need_min.cumsum() - plan.need_min
    plan["start_min"] = plan.win_start + plan.offset
    plan["end_min"] = plan.start_min + plan.need_min
    plan = plan.drop(columns="offset")
    blocks = (plan.groupby("window_id")
              .agg(section_id=("section_id", "first"), date=("date", "first"),
                   win_start=("win_start", "first"), win_end=("win_end", "first"),
                   n_tasks=("task_id", "count"), used_min=("need_min", "sum"),
                   depts=("dept", lambda d: "+".join(sorted(set(d)))))
              .reset_index()
              .merge(W[["window_id", "trains_affected", "goods_trains_affected"]], on="window_id"))
    blocks["length_min"] = blocks.win_end - blocks.win_start
    blocks["utilization"] = (blocks.used_min / blocks.length_min).round(3)
    return plan, blocks[BLOCK_COLS]


def _row(W, w, task_id):
    return dict(window_id=W.window_id[w], section_id=W.section_id[w], date=W.date[w],
                win_start=int(W.start_hour[w]) * 60, win_end=int(W.end_hour[w]) * 60,
                task_id=task_id)


# --------------------------------------------------------------------------
# CP-SAT model
# --------------------------------------------------------------------------
def solve(tasks, windows, horizon_days=7, start=START, section=None,
          time_limit=30, workers=8, rel_gap_limit=0.0, seed=1):
    W, T = scope(tasks, windows, horizon_days, start, section)
    pairs = candidate_pairs(T, W)
    unsched = find_unschedulable(T, W)

    m = cp_model.CpModel()
    x = {p: m.NewBoolVar(f"x{p[0]}_{p[1]}") for p in pairs}
    y = {w: m.NewBoolVar(f"y{w}") for w in W.index}
    gang_of = T.gang_type.to_dict()
    unknown = sorted(set(T.gang_type) - set(GANGS))
    if unknown:
        raise ValueError(f"gang types missing from GANGS: {unknown}")
    u = {(w, g): m.NewBoolVar(f"u{w}_{g}") for w in W.index for g in GANGS}

    by_task, by_win = defaultdict(list), defaultdict(list)
    for (i, w) in pairs:
        by_task[i].append(w)
        by_win[w].append(i)

    for i, ws in by_task.items():                                   # (1) at most once
        m.Add(sum(x[i, w] for w in ws) <= 1)
    for (i, w) in pairs:                                            # link x -> y, u
        m.Add(x[i, w] <= y[w])
        m.Add(x[i, w] <= u[w, gang_of[i]])
    for w in W.index:                                               # block open only if used
        m.Add(y[w] <= sum(x[i, w] for i in by_win.get(w, [])))
        for g in GANGS:                                             # gang reserved only if block open
            m.Add(u[w, g] <= y[w])
    for w, ts in by_win.items():       # (3) capacity; '* y[w]' is equivalent for integers (x<=y)
                                       #     but gives a tighter LP relaxation / better bound
        m.Add(sum(int(T.need_min[i]) * x[i, w] for i in ts) <= int(W.length_min[w]) * y[w])
    for _, g in W.groupby(["section_id", "date"]):                  # (5) blocks per section-day
        m.Add(sum(y[w] for w in g.index) <= MAX_BLOCKS_PER_DAY)
    slots = defaultdict(list)                                       # (6) gangs per (date, hour)
    for w, r in W.iterrows():
        for h in range(int(r.start_hour), int(r.end_hour)):
            slots[(r.date, h)].append(w)
    for ws in slots.values():
        for g, n in GANGS.items():
            if len(ws) > n:                                         # can only bind if more windows than gangs
                m.Add(sum(u[w, g] for w in ws) <= n)

    obj = []                                                        # same terms as net_value()
    for (i, w) in pairs:
        day_idx = (W.date[w] - start).days
        obj.append(assignment_value(T.priority_score[i], T.severity[i],
                                    T.overdue_days[i], day_idx) * x[i, w])
    for w in W.index:
        obj.append(-block_cost(W.trains_affected[w], W.goods_trains_affected[w]) * y[w])
    m.Maximize(sum(obj))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    s.parameters.random_seed = seed
    s.parameters.relative_gap_limit = rel_gap_limit
    t0 = time.time()
    status = s.Solve(m)
    wall = time.time() - t0

    name = s.StatusName(status)
    info = dict(status=name, proven_optimal=(name == "OPTIMAL"), wall_time_s=round(wall, 2),
                n_vars=len(pairs) + len(y) + len(u), unschedulable=unsched,
                too_long=unsched[unsched.reason == "TOO_LONG"].task_id.tolist(),
                start=start, horizon_days=horizon_days, objective=None, bound=None,
                abs_gap=None, rel_gap=None)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return pd.DataFrame(), pd.DataFrame(columns=BLOCK_COLS), info, W, T

    objective = int(round(s.ObjectiveValue()))
    bound = s.BestObjectiveBound()
    info["objective"], info["bound"] = objective, bound
    info["abs_gap"], info["rel_gap"] = compute_gap(objective, bound)

    rows = [_row(W, w, T.task_id[i]) for (i, w), var in x.items() if s.Value(var)]
    plan, blocks = finish_plan(pd.DataFrame(rows), T, W)
    return plan, blocks, info, W, T


# --------------------------------------------------------------------------
# Independent validator - rebuilds everything from tasks + windows (NOT from the
# attributes copied into the plan), so it can catch errors in the plan itself.
# --------------------------------------------------------------------------
def validate(plan, blocks, windows, tasks, start=START, horizon_days=7):
    errs = []
    if plan is None or len(plan) == 0:
        return errs
    t, w = tasks.set_index("task_id"), windows.set_index("window_id")
    end = start + pd.Timedelta(days=horizon_days)

    for tid in plan.task_id[plan.task_id.duplicated()].unique():
        errs.append(f"[DUPLICATE] {tid} scheduled more than once")

    used, win_gangs = Counter(), defaultdict(set)
    for r in plan.itertuples():
        if r.task_id not in t.index:
            errs.append(f"[UNKNOWN_TASK] {r.task_id}")
            continue
        if r.window_id not in w.index:
            errs.append(f"[UNKNOWN_WINDOW] {r.window_id}")
            continue
        tk, wd = t.loc[r.task_id], w.loc[r.window_id]
        if tk.section_id != wd.section_id:
            errs.append(f"[SECTION] {r.task_id} ({tk.section_id}) in window of {wd.section_id}")
        if not (start <= wd.date < end):
            errs.append(f"[HORIZON] {r.task_id} in {r.window_id} on {wd.date.date()}")
        if needs_power(tk.block_type) and not wd.power_block_possible:
            errs.append(f"[POWER] {r.task_id} needs a power block, {r.window_id} has none")
        used[r.window_id] += int(tk.duration_min + tk.setup_min)
        if tk.gang_type not in GANGS:
            errs.append(f"[GANG_UNKNOWN] {r.task_id}: {tk.gang_type}")
        else:
            win_gangs[r.window_id].add(tk.gang_type)

    for wid, mins in used.items():
        if wid in w.index and mins > w.length_min[wid]:
            errs.append(f"[CAPACITY] {wid}: {mins} min of work in a {w.length_min[wid]} min window")

    per_day = Counter((w.section_id[wid], w.date[wid]) for wid in used if wid in w.index)
    for (sec, d), n in per_day.items():
        if n > MAX_BLOCKS_PER_DAY:
            errs.append(f"[BLOCKS_PER_DAY] {sec} {d.date()}: {n} blocks > {MAX_BLOCKS_PER_DAY}")

    slot_use = Counter()                      # gangs reserved per (date, hour, gang type)
    for wid, gs in win_gangs.items():
        if wid not in w.index:
            continue
        for h in range(int(w.start_hour[wid]), int(w.end_hour[wid])):
            for g in gs:
                slot_use[(w.date[wid], h, g)] += 1
    for (d, h, g), n in sorted(slot_use.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if n > GANGS[g]:
            errs.append(f"[GANG] {g} on {d.date()} {h:02d}:00 - {n} reserved, only {GANGS[g]} exist")

    if {"start_min", "end_min"} <= set(plan.columns):         # sequencing inside each window
        for wid, g in plan.groupby("window_id"):
            if wid not in w.index:
                continue
            g = g.sort_values("start_min")
            lo, hi = int(w.start_hour[wid]) * 60, int(w.end_hour[wid]) * 60
            prev_end = lo
            for r in g.itertuples():
                need = int(t.duration_min[r.task_id] + t.setup_min[r.task_id])
                if r.start_min < prev_end or r.end_min - r.start_min != need or r.end_min > hi:
                    errs.append(f"[SEQUENCE] {wid}: {r.task_id} {r.start_min}-{r.end_min} "
                                f"overlaps, has wrong length or leaves the window")
                prev_end = max(prev_end, r.end_min)

    if blocks is not None and len(blocks):                    # block table must match the plan
        if set(blocks.window_id) != set(used):
            errs.append("[BLOCKS_MISMATCH] block table and plan use different windows")
        else:
            for b in blocks.itertuples():
                if b.used_min != used[b.window_id]:
                    errs.append(f"[BLOCKS_MISMATCH] {b.window_id}: used_min {b.used_min} "
                                f"!= {used[b.window_id]}")
    return errs


# --------------------------------------------------------------------------
# Constraint-compliant greedy baselines (same data, same constraints, same objective)
# --------------------------------------------------------------------------
class _State:
    """Tracks capacity, open blocks and gang reservations while a greedy plan is built.
    Mirrors constraints 3, 5 and 6 exactly as the CP-SAT model states them."""

    def __init__(self, W):
        self.W = W
        self.left = dict(zip(W.index, W.length_min.astype(int)))
        self.gangs = defaultdict(set)         # window -> gang types reserved
        self.opened = set()
        self.day_blocks = Counter()           # (section, date) -> open blocks
        self.slot = Counter()                 # (date, hour, gang) -> reserved

    def _hours(self, w):
        r = self.W.loc[w]
        return [(r.date, h) for h in range(int(r.start_hour), int(r.end_hour))]

    def feasible(self, t, w):
        if self.left[w] < t.need_min:
            return False
        r = self.W.loc[w]
        if w not in self.opened and self.day_blocks[(r.section_id, r.date)] >= MAX_BLOCKS_PER_DAY:
            return False
        if t.gang_type not in self.gangs[w]:
            if any(self.slot[(d, h, t.gang_type)] >= GANGS[t.gang_type] for d, h in self._hours(w)):
                return False
        return True

    def add(self, t, w):
        r = self.W.loc[w]
        if w not in self.opened:
            self.opened.add(w)
            self.day_blocks[(r.section_id, r.date)] += 1
        if t.gang_type not in self.gangs[w]:
            self.gangs[w].add(t.gang_type)
            for d, h in self._hours(w):
                self.slot[(d, h, t.gang_type)] += 1
        self.left[w] -= int(t.need_min)


def greedy_plan(tasks, windows, horizon_days=7, start=START, section=None, rule="first_fit"):
    """Tasks are taken in priority order (ties by task_id) and placed in a window that
    satisfies ALL hard constraints. Returns (plan, blocks).
      rule='first_fit' : earliest feasible window; ignores disruption cost
      rule='value'     : feasible window with the best marginal value (value of the task
                         minus the block cost if the window is not yet open); a task is
                         skipped when its best marginal value is <= 0"""
    if rule not in ("first_fit", "value"):
        raise ValueError("rule must be 'first_fit' or 'value'")
    W, T = scope(tasks, windows, horizon_days, start, section)
    by_task = defaultdict(list)
    for (i, w) in candidate_pairs(T, W):
        by_task[i].append(w)
    state, rows = _State(W), []
    for i in sorted(T.index, key=lambda i: (-T.priority_score[i], T.task_id[i])):
        t, best = T.loc[i], None
        for w in by_task.get(i, []):
            if not state.feasible(t, w):
                continue
            order = (W.date[w], int(W.start_hour[w]), W.window_id[w])
            if rule == "first_fit":
                key = order
            else:
                gain = assignment_value(t.priority_score, t.severity, t.overdue_days,
                                        (W.date[w] - start).days)
                cost = 0 if w in state.opened else block_cost(W.trains_affected[w],
                                                              W.goods_trains_affected[w])
                if gain - cost <= 0:
                    continue
                key = (-(gain - cost),) + order
            if best is None or key < best[0]:
                best = (key, w)
        if best:
            state.add(t, best[1])
            rows.append(_row(W, best[1], t.task_id))
    return finish_plan(pd.DataFrame(rows), T, W)


# --------------------------------------------------------------------------
# Reporting - identical metrics for every method
# --------------------------------------------------------------------------
def evaluate(name, plan, blocks, tasks, windows, start=START, horizon_days=7,
             section=None, runtime=None):
    W, T = scope(tasks, windows, horizon_days, start, section)
    errs = validate(plan, blocks, windows, tasks, start, horizon_days)
    gain, cost = gain_and_cost(plan, blocks, start)
    n = plan.task_id.nunique() if len(plan) else 0
    row = {"method": name, "tasks_scheduled": f"{n}/{len(T)}", "blocks": len(blocks),
           "avg_utilization": f"{blocks.utilization.mean():.0%}" if len(blocks) else "-",
           "trains_disrupted": int(blocks.trains_affected.sum()) if len(blocks) else 0,
           "goods_trains_disrupted": int(blocks.goods_trains_affected.sum()) if len(blocks) else 0,
           "gain": gain, "block_cost": cost, "net_value": gain - cost}
    for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        row[f"{tier.lower()}_sched"] = (
            f"{plan[plan.tier == tier].task_id.nunique() if len(plan) else 0}/{(T.tier == tier).sum()}")
    row["overdue_sched"] = (
        f"{plan[plan.overdue_days > 0].task_id.nunique() if len(plan) else 0}/{(T.overdue_days > 0).sum()}")
    row["violations"] = len(errs)
    row["runtime_s"] = round(runtime, 2) if runtime is not None else None
    return row, errs


def print_gap(info):
    print(f"CP-SAT status: {info['status']} | wall time {info['wall_time_s']}s | "
          f"{info['n_vars']} boolean variables")
    if info["objective"] is None:
        print("  no feasible plan returned - no objective, bound or gap available")
        return
    print(f"  objective (net value): {info['objective']} | best bound: {info['bound']:.1f}")
    print(f"  absolute gap: {info['abs_gap']:.1f} | relative gap: {info['rel_gap']:.2%}")
    if info["proven_optimal"]:
        print("  OPTIMALITY PROVEN by CP-SAT for this model and data (relative to modelling assumptions).")
    else:
        print(f"  NOT proven optimal: the best possible value is at most {info['bound']:.1f}; "
              f"the plan is within {info['rel_gap']:.2%} of that bound, no better claim is made.")


def print_unschedulable(u):
    print(f"\nTasks that cannot be placed in this horizon: {len(u)}")
    if u.empty:
        return
    print(u.groupby("reason").size().rename("tasks").to_string())
    tl = u[u.reason == "TOO_LONG"]
    if len(tl):
        print(f"  TOO_LONG: {len(tl)} tasks; shortfall min/median/max = "
              f"{tl.shortfall_min.min():.0f}/{tl.shortfall_min.median():.0f}/{tl.shortfall_min.max():.0f} min "
              f"-> need extended blocks. By task type:")
        print("   ", tl.groupby("task").size().to_dict())


def summarize(label, plan, blocks, info, T):
    print(f"\n=== {label} ===")
    print_gap(info)
    print(f"tasks in scope: {len(T)} | scheduled: {plan.task_id.nunique() if len(plan) else 0} | "
          f"cannot be placed: {len(info['unschedulable'])}")
    print_unschedulable(info["unschedulable"])


def check_objective(info, plan, blocks):
    """The solver objective and net_value() must be the same integer."""
    if info["objective"] is None:
        return True
    nv = net_value(plan, blocks, info["start"])
    ok = nv == info["objective"]
    print(f"objective consistency: solver {info['objective']} vs net_value() {nv} -> "
          f"{'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        raise AssertionError("CP-SAT objective != net_value(); the model and the metric diverged")
    return ok


def run_case(label, tasks, windows, args, section=None, save=False):
    kw = dict(horizon_days=args.horizon_days, section=section)
    plan, blocks, info, W, T = solve(tasks, windows, time_limit=args.time_limit,
                                     workers=args.workers, rel_gap_limit=args.rel_gap, **kw)
    summarize(label, plan, blocks, info, T)
    check_objective(info, plan, blocks)

    rows, all_errs = [], {}
    r, e = evaluate("CP-SAT", plan, blocks, tasks, windows, runtime=info["wall_time_s"], **kw)
    rows.append(r), all_errs.update({"CP-SAT": e})
    for name, rule in (("greedy first-fit", "first_fit"), ("greedy value", "value")):
        t0 = time.time()
        gp, gb = greedy_plan(tasks, windows, rule=rule, **kw)
        r, e = evaluate(name, gp, gb, tasks, windows, runtime=time.time() - t0, **kw)
        rows.append(r), all_errs.update({name: e})
        if save:
            gp.to_csv(OUT / f"weekly_plan_{rule}.csv", index=False)
    table = pd.DataFrame(rows).set_index("method").T
    print("\nSame data, same constraints, same objective:")
    print(table.to_string())
    for name, e in all_errs.items():
        if e:
            print(f"!! {name}: {len(e)} violations, first: {e[0]}")
    if info["bound"] is not None:
        beyond = [r["method"] for r in rows if r["net_value"] > info["bound"] + 1e-6]
        if beyond:
            raise AssertionError(f"{beyond} exceed the CP-SAT bound - model/baseline mismatch")
    if save:
        OUT.mkdir(exist_ok=True)
        plan.to_csv(OUT / "weekly_plan_v1.csv", index=False)
        blocks.to_csv(OUT / "weekly_blocks_v1.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT / "weekly_method_comparison.csv", index=False)
        u = info["unschedulable"].assign(horizon_start=info["start"].date(),
                                         horizon_days=info["horizon_days"])
        u.to_csv(OUT / "needs_extended_block.csv", index=False)
    return plan, blocks, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-days", type=int, default=7)
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rel-gap", type=float, default=0.0)
    ap.add_argument("--section", default=None, help="single-section run (default: auto + network)")
    args = ap.parse_args()

    tasks, windows = load()
    sec = args.section or tasks.section_id.value_counts().index[0]
    plan, blocks, info = run_case(f"Single section {sec}", tasks, windows, args, section=sec)
    if len(plan):
        show = plan[["window_id", "date", "dept", "task_id", "task", "tier", "start_min", "end_min"]].copy()
        show["date"] = show.date.dt.strftime("%a %d-%b")
        for c in ("start_min", "end_min"):
            show[c] = show[c].map(lambda m: f"{m // 60:02d}:{m % 60:02d}")
        print("\nFirst blocks of the single-section plan:")
        print(show.head(12).to_string(index=False))
    if not args.section:
        run_case("Network", tasks, windows, args, save=True)
