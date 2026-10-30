"""
Step 5 - simulated DECENTRALISED (manual) block planning, for a fair comparison.

How today's process is modelled (all of this is an ASSUMPTION, state it in the report)
    M1  Each department plans alone, using only its own task list. It does not see the other
        departments' tasks, the AI priority score, disruption cost, or merge opportunities.
    M2  A department works through its tasks most-overdue-first (ties: higher severity), the way a
        manual planner would clear a backlog, and requests the EARLIEST corridor window that
        works. Tasks of the same department are batched in one window when they fit (one crew,
        setup paid once), because that is natural manual practice.
    M3  Every request is a separate BDMS block: a window is used by ONE department only (no
        cross-department merging, no parallel crews in one block).
    M4  The control office (COA) grants a request only if the hard rules hold: window on the task's
        section, long enough, power window for power work, at most 2 blocks per section per day,
        crews available (gang pool), and the preventive-maintenance start rule (B10). A refused
        request is re-requested for the next window. So the final manual plan satisfies the SAME hard
        constraints as the optimised plan (checked by validate_parallel(merge=False)).
    M5  Departments submit one after another; the order matters (first come, first served), so the
        baseline is run for all 6 orders and reported as mean / best / worst.
    M6  A manual planner requests every placeable task: it never declines a task because the block
        costs more than the task is worth, and it does not defer work to smooth the load.

Fair-comparison rules: same task table, same COA windows, same hard constraints, same objective
definition, same KPI function (metrics.kpis).

Run:  python baseline.py            (needs outputs/ from horizons.py)
"""
import argparse
import itertools
from collections import Counter
from pathlib import Path

import pandas as pd

import horizons as h
import metrics as m
import optimizer as o

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
START = o.START
DEPTS = ("ENGG", "S&T", "TRD")


class _Bookings:
    """Capacity bookkeeping for the manual process (one department per window, M3)."""

    def __init__(self, W):
        self.W = W
        self.owner = {}                      # window index -> department that requested it
        self.used = Counter()                # window index -> crew minutes (setup + work)
        self.day_blocks = Counter()          # (section, date) -> opened blocks
        self.slot = Counter()                # (date, hour, gang) -> crews reserved

    def feasible(self, t, w):
        r = self.W.loc[w]
        if self.owner.get(w, t.dept) != t.dept:                         # window belongs to another dept
            return False
        opened = w in self.owner
        add = int(t.duration_min) + (0 if opened else int(t.crew_setup_min))
        if self.used[w] + add > int(r.length_min):
            return False
        if not opened:
            if self.day_blocks[(r.section_id, r.date)] >= o.MAX_BLOCKS_PER_DAY:
                return False
            if any(self.slot[(r.date, hh, t.gang_type)] >= o.GANGS[t.gang_type]
                   for hh in range(int(r.start_hour), int(r.end_hour))):
                return False
        return True

    def add(self, t, w):
        r = self.W.loc[w]
        if w not in self.owner:
            self.owner[w] = t.dept
            self.day_blocks[(r.section_id, r.date)] += 1
            for hh in range(int(r.start_hour), int(r.end_hour)):
                self.slot[(r.date, hh, t.gang_type)] += 1
            self.used[w] += int(t.crew_setup_min)
        self.used[w] += int(t.duration_min)


def manual_baseline(T, windows, start=START, horizon_days=28, dept_order=DEPTS):
    """One manual planning round. Returns (plan, blocks) in the Step 4 format."""
    W, Ts = o.scope(T, windows, horizon_days, start)
    by_task = {}
    for (i, w) in o.candidate_pairs(Ts, W):                              # section, length, power
        if (W.date[w] - start).days >= Ts.earliest_idx[i]:               # B10 start rule (M4)
            by_task.setdefault(i, []).append(w)
    book, rows = _Bookings(W), []
    for dept in dept_order:
        mine = Ts.index[Ts.dept == dept]
        for i in sorted(mine, key=lambda i: (-Ts.overdue_days[i], -Ts.severity[i], Ts.task_id[i])):
            t = Ts.loc[i]
            for w in sorted(by_task.get(i, []), key=lambda w: (W.date[w], int(W.start_hour[w]), W.window_id[w])):
                if book.feasible(t, w):
                    book.add(t, w)
                    rows.append(o._row(W, w, t.task_id))
                    break
    plan, blocks = h.finish_plan_parallel(pd.DataFrame(rows), Ts, W)
    if len(plan):
        plan, blocks = m.with_week(plan, start), m.with_week(blocks, start)
    return plan, blocks


def run_orders(T, windows, start=START, horizon_days=28):
    """All 6 submission orders. Returns (kpi table, {order: (plan, blocks)})."""
    rows, plans = [], {}
    for perm in itertools.permutations(DEPTS):
        plan, blocks = manual_baseline(T, windows, start, horizon_days, perm)
        errs = h.validate_parallel(plan, windows, T, start=start, horizon_days=horizon_days,
                                   merge=False, origin=start)
        label = " > ".join(perm)
        k = m.kpis(plan, blocks, T, windows, start, horizon_days)
        rows.append(dict(order=label, violations=len(errs), **k))
        plans[label] = (plan, blocks)
    return pd.DataFrame(rows), plans


NUMERIC_KPIS = [k for k in m.KPI_LABELS]


def compare(ai, base):
    """Optimised plan vs the range of manual outcomes. verdict compares against ALL 6 orders."""
    rows = []
    for k in NUMERIC_KPIS:
        vals = pd.to_numeric(base[k], errors="coerce").dropna()
        a = ai.get(k)
        if a is None or vals.empty:
            continue
        d = m.KPI_DIRECTION.get(k, 0)
        lo, hi = float(vals.min()), float(vals.max())
        if d == 0:
            verdict = "info"
        elif d > 0:
            verdict = "better than all orders" if a > hi else "worse than all orders" if a < lo else \
                      "equal to all orders" if lo == hi == a else "within the manual range"
        else:
            verdict = "better than all orders" if a < lo else "worse than all orders" if a > hi else \
                      "equal to all orders" if lo == hi == a else "within the manual range"
        rows.append(dict(kpi=m.KPI_LABELS[k], key=k, optimised=a, manual_mean=round(float(vals.mean()), 2),
                         manual_best_order=(lo if d < 0 else hi if d > 0 else None),
                         manual_worst_order=(hi if d < 0 else lo if d > 0 else None), verdict=verdict))
    return pd.DataFrame(rows)


def best_order(base):
    """Submission order with the highest manual net value (used for the dashboard's manual Gantt)."""
    return base.sort_values(["net_value", "order"], ascending=[False, True]).iloc[0].order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--early-days", type=int, default=h.EARLY_DAYS)
    a = ap.parse_args()
    T, windows = m.load_inputs(a.early_days)
    res = m.load_results()
    if res["plan"] is None:
        raise SystemExit("run horizons.py first (outputs/block_plan_4w.csv is missing)")
    ai = m.kpis(res["plan"], res["blocks"], T, windows)
    base, plans = run_orders(T, windows)
    cmp = compare(ai, base)
    pd.set_option("display.width", 220)
    print("Manual baseline: 6 submission orders, same data / constraints / objective / KPIs")
    print(base[["order", "tasks_scheduled", "blocks", "trains_disrupted", "net_value",
                "days_past_due_total", "week1_share_pct", "violations"]].to_string(index=False))
    print("\nOptimised plan vs manual:\n" + cmp.drop(columns="key").to_string(index=False))
    OUT.mkdir(exist_ok=True)
    base.to_csv(OUT / "baseline_orders.csv", index=False)
    cmp.to_csv(OUT / "baseline_comparison.csv", index=False)
    bo = best_order(base)
    p, b = plans[bo]
    p.to_csv(OUT / "manual_plan_best_order.csv", index=False)
    b.to_csv(OUT / "manual_blocks_best_order.csv", index=False)
    print(f"\nmanual plan of the best order ({bo}) saved for the dashboard")


if __name__ == "__main__":
    main()
