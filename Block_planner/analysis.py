"""
Open questions from the Build Plan, answered on the current data (Step 7 analyses).

  1. Myopic reference  - does the monthly stage help, or would "weekly only" do as well?
  2. Newly late tasks  - why are some tasks that were NOT overdue at the start finished after their due date?
  3. Safety rule (B3)  - how often does the S&T / Traction 1 km rule actually change the plan?

Run:   python Block_planner/analysis.py            (about 3-4 minutes; needs outputs/ from horizons.py)
Writes outputs/analysis_myopic.csv, analysis_late_tasks.csv, analysis_safety.csv and analysis_summary.txt.

Nothing here changes the planner. The saved plan in outputs/ is the reference; the comparison runs re-solve
from scratch with the same default settings that `horizons.py` uses.
"""
import argparse
import contextlib

import pandas as pd

import horizons as h
import metrics as m
import optimizer as o

OUT = h.OUT
KPI_COLS = ["tasks_scheduled", "blocks", "merged_blocks", "trains_disrupted", "net_value", "overdue_mean_day",
            "crit_high_mean_day", "days_past_due_total", "newly_late", "week1_share_pct", "peak_to_mean_work"]


def default_settings():
    """The settings `python horizons.py` uses when run with no options."""
    return dict(model="blocks", defer=True, balance=h.DEFAULT_BALANCE, balance_cap=h.BALANCE_CAP,
                reserve=h.EMERGENCY_RESERVE)


def full_run(T, windows, time_limit=30, workers=8):
    """Monthly plan + rolling weekly refinement, exactly as horizons.main() does it."""
    mkw = default_settings()
    assign, _ = h.solve_monthly(T, windows, n_weeks=h.N_WEEKS, time_limit=time_limit, workers=workers, **mkw)
    plan, blocks, summ, _ = h.run_horizon(T, windows, assign, n_weeks=h.N_WEEKS, time_limit=time_limit,
                                          workers=workers, replan=mkw, backfill=True)
    return plan, blocks, summ


def kpi_row(name, plan, blocks, T, windows):
    k = m.kpis(m.with_week(plan), m.with_week(blocks), T, windows)
    return dict(plan=name, **{c: k[c] for c in KPI_COLS})


@contextlib.contextmanager
def no_safety_rule():
    saved = dict(h.CONFLICT_PAIRS)
    h.CONFLICT_PAIRS.clear()
    try:
        yield
    finally:
        h.CONFLICT_PAIRS.update(saved)


# ---------------------------------------------------------------- 1. myopic
def myopic_comparison(T, windows, plan, blocks, time_limit, workers):
    p2, b2, _, _ = h.run_horizon(T, windows, None, n_weeks=h.N_WEEKS, time_limit=time_limit, workers=workers)
    return pd.DataFrame([kpi_row("monthly + weekly (saved plan)", plan, blocks, T, windows),
                         kpi_row("weekly only (myopic)", p2, b2, T, windows)])


# ---------------------------------------------------------------- 2. newly late
def late_tasks(T, windows, plan, blocks, start=o.START):
    """Every not-yet-overdue task that starts after its due day, with the evidence for why."""
    Ti = T.set_index("task_id")
    W = windows.assign(day=(windows.date - start).dt.days)
    first = plan.sort_values("date").drop_duplicates("task_id").set_index("task_id")
    used = set(plan.window_id)
    rows = []
    for t, r in first.iterrows():
        tk = Ti.loc[t]
        due, day = int(-tk.overdue_days), int((r.date - start).days)
        if tk.overdue_days > 0 or day <= due:
            continue
        ok = W[(W.section_id == tk.section_id) & (W.length_min >= tk.need_min)
               & (W.day >= max(int(tk.earliest_idx), 0)) & (W.day <= due)]
        if o.needs_power(tk.block_type):
            ok = ok[ok.power_block_possible]
        crew_full, crew_busy = 0, 0                   # why the already-opened on-time blocks could not take it
        for w in ok[ok.window_id.isin(used)].itertuples():
            s, e = w.start_hour * 60, w.end_hour * 60
            same_time = plan[(plan.date == w.date) & (plan.gang_type == tk.gang_type)
                             & (plan.win_start < e) & (plan.win_end > s)].window_id.nunique()
            in_window = plan[(plan.window_id == w.window_id) & (plan.gang_type == tk.gang_type)]
            if len(in_window):
                crew_busy += 1
            elif same_time >= o.GANGS[tk.gang_type]:
                crew_full += 1
        unopened = ok[~ok.window_id.isin(used)]
        new_block = (int(unopened.apply(lambda w: o.block_cost(w.trains_affected, w.goods_trains_affected),
                                        axis=1).min()) if len(unopened) else None)
        delay_cost = (o.assignment_value(tk.priority_score, tk.severity, tk.overdue_days, due)
                      - o.assignment_value(tk.priority_score, tk.severity, tk.overdue_days, day))
        b = blocks.set_index("window_id").loc[r.window_id]
        rows.append(dict(task_id=t, dept=tk.dept, section_id=tk.section_id, task=tk.task, tier=tk.tier,
                         due_day=due, start_day=day, days_late=day - due,
                         ontime_windows=len(ok), ontime_opened=int(ok.window_id.isin(used).sum()),
                         opened_but_same_crew_busy=crew_busy, opened_but_crews_all_out=crew_full,
                         cheapest_new_ontime_block=new_block, cost_of_lateness=delay_cost,
                         placed_in_joint_block=bool(b.merged)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. safety rule
def safety_effect(T, windows, plan, blocks, time_limit, workers):
    pairs = h.conflict_pairs(T)
    ids = [(T.task_id[i], T.task_id[j]) for i, j in pairs]
    win = plan.drop_duplicates("task_id").set_index("task_id").window_id
    with no_safety_rule():
        p2, b2, _ = full_run(T, windows, time_limit, workers)
    win2 = p2.drop_duplicates("task_id").set_index("task_id").window_id

    def count(wmap):
        both = [(a, b) for a, b in ids if a in wmap.index and b in wmap.index]
        return len(both), sum(wmap[a] == wmap[b] for a, b in both)

    sched_with, same_with = count(win)
    sched_without, same_without = count(win2)
    kp = pd.DataFrame([kpi_row("with safety rule (saved plan)", plan, blocks, T, windows),
                       kpi_row("without safety rule", p2, b2, T, windows)])
    kp["close_pairs"] = len(ids)
    kp["pairs_both_scheduled"] = [sched_with, sched_without]
    kp["pairs_in_same_block"] = [same_with, same_without]
    return kp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=30)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    T, windows = m.load_inputs()
    res = m.load_results()
    if res["plan"] is None:
        raise SystemExit("run horizons.py first (outputs/block_plan_4w.csv is missing)")
    plan, blocks = res["plan"], res["blocks"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    lines = []

    print("1/3  weekly-only (myopic) reference ...")
    my = myopic_comparison(T, windows, plan, blocks, a.time_limit, a.workers)
    my.to_csv(OUT / "analysis_myopic.csv", index=False)
    lines += ["== 1. Monthly + weekly vs weekly only ==", my.T.to_string(header=False), ""]

    print("2/3  newly late tasks ...")
    lt = late_tasks(T, windows, plan, blocks)
    lt.to_csv(OUT / "analysis_late_tasks.csv", index=False)
    lines += [f"== 2. Not-yet-overdue tasks finished late: {len(lt)} ==", lt.to_string(index=False), ""]

    print("3/3  plan without the S&T / Traction safety rule ...")
    sf = safety_effect(T, windows, plan, blocks, a.time_limit, a.workers)
    sf.to_csv(OUT / "analysis_safety.csv", index=False)
    lines += ["== 3. Effect of the S&T / Traction 1 km safety rule ==", sf.T.to_string(header=False), ""]

    text = "\n".join(lines)
    (OUT / "analysis_summary.txt").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
