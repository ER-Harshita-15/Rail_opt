"""
Step 5 - what-if: impact preview of an EMERGENCY defect on an already planned week.

The chosen week is re-optimised with the emergency task FORCED in (same weekly model, same
objective, same hard constraints). The result is compared with the current plan of that week:
what moved, what was displaced, extra blocks / trains, change in net value.

Assumptions
    W1  The pool is the week's currently scheduled tasks + the emergency task. Tasks that were left
        unscheduled are not added, so the preview shows the cost of making room, nothing else.
    W2  The week is fully re-optimised, so unrelated tasks can move too. There is NO stability
        objective here (keeping the existing plan as it is = Step 6, rolling re-planning); the
        preview reports how many tasks moved so this churn is visible.
    W5  net_value_change includes the emergency task's own value. cost_to_rest_of_plan removes it:
        what the rest of the plan pays (extra blocks, delay, displaced work) to make room.
    W3  An emergency task gets the top tier (CRITICAL) and a user-set priority score (default 0.95);
        it is treated as already due (overdue_days = 1) and may use any window of the week.
    W4  Duration and department are entered by the user; setup, crew and block type are taken from
        that department's existing tasks.
"""
import pandas as pd

import horizons as h
import metrics as m
import optimizer as o

START = o.START
SOURCE = {"ENGG": "TMS", "S&T": "SMMS", "TRD": "TDMS"}


def make_emergency_task(T, dept, section_id, km, duration_min, task="Emergency defect",
                        task_id="EMG-001", severity=5, priority_score=0.95, overdue_days=1,
                        early_days=h.EARLY_DAYS, tier="CRITICAL", past_failures=0):
    """One-row task table in the same format as the prepared task table (W3, W4)."""
    like = T[T.dept == dept]
    if like.empty:
        raise ValueError(f"no existing tasks for department {dept!r}")
    row = like.iloc[0].to_dict()
    gang_setup = int(like.crew_setup_min.max())
    row.update(task_id=task_id, dept=dept, source_system=SOURCE.get(dept, row.get("source_system")),
               section_id=section_id, km=float(km), task=task, severity=int(severity),
               due_date="emergency", overdue_days=int(overdue_days), duration_min=int(duration_min),
               past_failures=int(past_failures), priority_score=float(priority_score), priority_rank=0, tier=tier,
               crew_setup_min=gang_setup, need_min=int(duration_min) + gang_setup,
               earliest_idx=-int(overdue_days) - int(early_days))
    return pd.DataFrame([row])[list(T.columns)]


def _times(df):
    return df.assign(slot=df.date.dt.strftime("%a %d-%b") + " " + (df.start_min // 60).astype(int).map("{:02d}".format)
                     + ":" + (df.start_min % 60).astype(int).map("{:02d}".format))


def impact_in_week(T, windows, plan, blocks, week, emg, start=START, time_limit=15, workers=8):
    """Re-plan `week` (1-based) with the emergency task(s) forced in. Returns a dict."""
    emg_ids = list(emg.task_id)
    T2 = pd.concat([T, emg], ignore_index=True)
    old_p, old_b = plan[plan.week == week], blocks[blocks.week == week]
    ws = start + pd.Timedelta(days=7 * (week - 1))
    new_p, new_b, info, _, _ = h.solve_weekly(T2, windows, start=ws, origin=start,
                                              task_ids=list(old_p.task_id) + emg_ids, force_ids=emg_ids,
                                              time_limit=time_limit, workers=workers)
    out = dict(week=week, info=info, feasible=info["objective"] is not None)
    if not out["feasible"]:
        why = info["unschedulable"]
        out["reason"] = ("no window in this week can hold the emergency task (length, power block, section "
                         "or the earliest-start rule)" if info["status"] == "INFEASIBLE_FORCED" else info["status"])
        out["unschedulable"] = why
        return out
    new_p, new_b = m.with_week(new_p, start), m.with_week(new_b, start)
    old_t, new_t = _times(old_p), _times(new_p)
    o_idx, n_idx = old_t.set_index("task_id"), new_t.set_index("task_id")
    kept_ids, dropped_ids = set(o_idx.index) & set(n_idx.index), set(o_idx.index) - set(n_idx.index)
    moved = [t for t in kept_ids if o_idx.window_id[t] != n_idx.window_id[t]]
    out["emergency"] = new_t[new_t.task_id.isin(emg_ids)][["task_id", "dept", "section_id", "task", "slot",
                                                            "window_id", "start_min", "end_min"]]
    out["moved"] = pd.DataFrame([dict(task_id=t, tier=o_idx.tier[t], dept=o_idx.dept[t], was=o_idx.slot[t],
                                      now=n_idx.slot[t]) for t in sorted(moved)])
    out["displaced"] = pd.DataFrame([dict(task_id=t, tier=o_idx.tier[t], dept=o_idx.dept[t],
                                          section_id=o_idx.section_id[t], was=o_idx.slot[t])
                                     for t in sorted(dropped_ids)])
    ov, nv = o.net_value(old_p, old_b, start), o.net_value(new_p, new_b, start)
    emg_value = sum(o.assignment_value(r.priority_score, r.severity, r.overdue_days, (r.date - start).days)
                    for r in new_p[new_p.task_id.isin(emg_ids)].itertuples())
    out["summary"] = dict(emergency_value=int(emg_value), cost_to_rest_of_plan=int(nv - ov - emg_value),
        net_value_before=int(ov), net_value_after=int(nv), net_value_change=int(nv - ov),
        blocks_before=len(old_b), blocks_after=len(new_b), blocks_added=len(new_b) - len(old_b),
        trains_before=int(old_b.trains_affected.sum()), trains_after=int(new_b.trains_affected.sum()),
        tasks_before=old_p.task_id.nunique(), tasks_after=new_p.task_id.nunique() - len(emg_ids),
        tasks_moved=len(moved), tasks_displaced=len(dropped_ids),
        proven_optimal=bool(info["proven_optimal"]), rel_gap=info["rel_gap"])
    out["new_plan"], out["new_blocks"] = new_p, new_b
    return out


def scan_weeks(T, windows, plan, blocks, emg, weeks=(1, 2, 3, 4), start=START, time_limit=10, workers=8):
    """The same emergency task tried in each week: where does it hurt least?"""
    rows = []
    for k in weeks:
        r = impact_in_week(T, windows, plan, blocks, k, emg, start, time_limit, workers)
        if not r["feasible"]:
            rows.append(dict(week=k, feasible=False, net_value_change=None, cost_to_rest=None, blocks_added=None,
                             trains_added=None, tasks_displaced=None, tasks_moved=None, proven=None))
            continue
        s = r["summary"]
        rows.append(dict(week=k, feasible=True, net_value_change=s["net_value_change"],
                         cost_to_rest=s["cost_to_rest_of_plan"],
                         blocks_added=s["blocks_added"], trains_added=s["trains_after"] - s["trains_before"],
                         tasks_displaced=s["tasks_displaced"], tasks_moved=s["tasks_moved"],
                         proven=s["proven_optimal"]))
    return pd.DataFrame(rows)
