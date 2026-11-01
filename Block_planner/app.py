"""
RailOpt - Automatic Block Planning dashboard (Step 5).

Run:  streamlit run app.py        (needs outputs/ from `python horizons.py`)

Everything shown is built from SYNTHETIC data. The manual baseline is a SIMULATION of decentralised
planning under stated assumptions (see baseline.py). The system recommends; officials approve.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import baseline as bl
import bdms_export as bx
import horizons as h
import metrics as m
import optimizer as o
import replan as rp
import whatif as wi

ROOT = Path(__file__).parent
DEPT_COLORS = {"ENGG": "#2a6f97", "S&T": "#e07a1f", "TRD": "#6a994e"}

st.set_page_config(page_title="RailOpt - Automatic Block Planning", layout="wide")


@st.cache_data(show_spinner=False)
def load_all():
    T, windows = m.load_inputs()
    return T, windows, m.load_results(), pd.read_csv(o.DATA / "sections.csv")


@st.cache_data(show_spinner="Simulating manual planning (6 submission orders)...")
def manual_runs():
    T, windows = m.load_inputs()
    base, plans = bl.run_orders(T, windows)
    return base, plans


def gantt_frame(plan):
    d = plan.copy()
    d["start"] = d.date + pd.to_timedelta(d.start_min, unit="m")
    d["end"] = d.date + pd.to_timedelta(d.end_min, unit="m")
    d["lane"] = d.section_id + "  |  " + d.dept
    return d


def gantt(plan, title):
    d = gantt_frame(plan)
    if d.empty:
        st.info("Nothing to show for this selection.")
        return
    fig = px.timeline(d, x_start="start", x_end="end", y="lane", color="dept", color_discrete_map=DEPT_COLORS,
                      hover_data={"task_id": True, "task": True, "tier": True, "km": True, "start": "|%a %d-%b %H:%M",
                                  "end": "|%H:%M", "lane": False, "dept": False}, title=title)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title=None)
    fig.update_layout(height=max(380, 26 * d.lane.nunique() + 140), legend_title_text="Department",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


def delta(a, b):
    if b in (None, 0) or a is None:
        return None
    return f"{(a - b) / abs(b) * 100:+.0f}% vs manual mean"


if not (ROOT / "outputs" / "block_plan_4w.csv").exists():
    st.error("No plan found. Run `python horizons.py` first, then reload this page.")
    st.stop()

T, windows, res, sections = load_all()
plan, blocks = m.with_week(res["plan"]), m.with_week(res["blocks"])
base, mplans = manual_runs()
ai = m.kpis(plan, blocks, T, windows)
cmp = bl.compare(ai, base)
mean = base.select_dtypes("number").mean()
best = bl.best_order(base)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("RailOpt")
    st.caption("Automatic block planning - synthetic data, recommendations only")
    st.markdown(f"**Plan:** {plan.week.nunique()} weeks from {o.START:%d %b %Y}  \n"
                f"**Tasks:** {len(T)} | **COA windows:** {len(windows)}")
    with st.expander("Re-run the planner (about 1 minute)"):
        bal = st.number_input("Balance strength (B12)", 0, 50, h.DEFAULT_BALANCE, key="rerun_balance")
        res_pct = st.slider("Emergency reserve", 0.0, 0.5, h.EMERGENCY_RESERVE, 0.05, key="rerun_reserve")
        early = st.number_input("Earliest start, days before due (B10)", 0, 60, h.EARLY_DAYS, key="rerun_early")
        if st.button("Run horizons.py", key="rerun_btn"):
            with st.spinner("Planning 4 weeks..."):
                r = subprocess.run([sys.executable, "horizons.py", "--no-ablation", "--balance", str(bal),
                                    "--reserve", str(res_pct), "--early-days", str(early)],
                                   cwd=ROOT, capture_output=True, text=True, timeout=900)
            st.cache_data.clear()
            st.success("Done - reload the page.") if r.returncode == 0 else st.error(r.stderr[-800:])
    st.caption("The what-if and baseline use the default 14-day earliest-start rule.")

tabs = st.tabs(["Overview", "Block calendar", "Monthly plan", "What-if", "Rolling re-plan",
                "Unscheduled & diagnostics", "Export"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    st.subheader("Optimised plan vs simulated manual planning")
    c = st.columns(5)
    c[0].metric("Tasks scheduled", f"{ai['tasks_scheduled']} / {ai['placeable_tasks']}",
                f"{ai['tasks_scheduled'] - mean.tasks_scheduled:+.0f} vs manual mean")
    c[1].metric("Blocks opened", ai["blocks"], delta(ai["blocks"], mean.blocks), delta_color="inverse")
    c[2].metric("Trains disrupted", ai["trains_disrupted"], delta(ai["trains_disrupted"], mean.trains_disrupted),
                delta_color="inverse")
    c[3].metric("Overdue tasks scheduled", ai["overdue_cleared"], f"start day {ai['overdue_mean_day']} vs {mean.overdue_mean_day:.2f}",
                delta_color="off")
    c[4].metric("Net value", f"{ai['net_value']:,}", delta(ai["net_value"], mean.net_value))
    st.caption("Manual = mean over the 6 orders in which departments could submit requests. "
               "Same data, same hard constraints, same objective, same KPI code.")

    show = cmp.drop(columns="key").rename(columns={"kpi": "KPI", "optimised": "Optimised", "manual_mean": "Manual (mean)",
                                                   "manual_best_order": "Manual (best order)",
                                                   "manual_worst_order": "Manual (worst order)", "verdict": "Verdict"})
    st.dataframe(show, hide_index=True, width="stretch")

    left, right = st.columns(2)
    chart = pd.DataFrame({"KPI": ["Blocks", "Trains disrupted / 10", "Task-days past due / 10"] * 2,
                          "Plan": ["Optimised"] * 3 + ["Manual (mean)"] * 3,
                          "Value": [ai["blocks"], ai["trains_disrupted"] / 10, ai["days_past_due_total"] / 10,
                                    mean.blocks, mean.trains_disrupted / 10, mean.days_past_due_total / 10]})
    left.plotly_chart(px.bar(chart, x="KPI", y="Value", color="Plan", barmode="group",
                             color_discrete_sequence=["#2a6f97", "#b0b7bf"]), width="stretch")
    wk = h.workload_profile(plan, blocks, plan.week.max())[0].assign(Plan="Optimised")
    mp, mb = mplans[best]
    wm = h.workload_profile(mp, mb, plan.week.max())[0].assign(Plan="Manual (best order)")
    right.plotly_chart(px.bar(pd.concat([wk, wm]), x="week", y="tasks", color="Plan", barmode="group",
                              title="Tasks per week", color_discrete_sequence=["#2a6f97", "#b0b7bf"]),
                       width="stretch")
    st.info("**Read this carefully.** The manual planner requests every placeable task, so it schedules more tasks "
            f"({mean.tasks_scheduled:.0f} vs {ai['tasks_scheduled']}); the optimiser declines tasks whose block costs more than "
            "they are worth. Manual outcomes depend on the assumptions in `baseline.py` (M1-M6), above all that "
            "departments never share a block. 'Task-days past due' is a proxy, not measured downtime.")

# ---------------------------------------------------------------- calendar
with tabs[1]:
    st.subheader("Block calendar")
    c1, c2, c3, c4 = st.columns(4)
    source = c1.selectbox("Plan", ["Optimised", f"Manual ({best})"], key="cal_source")
    weeks = sorted(plan.week.unique())
    week = c2.selectbox("Week", ["All"] + weeks, index=1, key="cal_week")
    depts = c3.multiselect("Departments", list(DEPT_COLORS), default=list(DEPT_COLORS), key="cal_dept")
    secs_all = sorted(T.section_id.unique())
    secs = c4.multiselect("Sections", secs_all, default=secs_all[:4], key="cal_sec")
    p = plan if source == "Optimised" else m.with_week(mplans[best][0])
    p = p[p.dept.isin(depts) & p.section_id.isin(secs)]
    if week != "All":
        p = p[p.week == week]
    gantt(p, f"{source} - {'all weeks' if week == 'All' else f'week {week}'}")
    st.caption("One bar = one task. Bars of different departments on the same section are the same block: "
               "crews work in parallel, each after its own setup.")
    st.dataframe(p[["date", "section_id", "dept", "task_id", "task", "tier", "km", "start_min", "end_min"]]
                 .assign(date=lambda d: d.date.dt.strftime("%a %d-%b"),
                         start_min=lambda d: d.start_min.map(lambda x: f"{x // 60:02d}:{x % 60:02d}"),
                         end_min=lambda d: d.end_min.map(lambda x: f"{x // 60:02d}:{x % 60:02d}"))
                 .rename(columns={"start_min": "start", "end_min": "end"}), hide_index=True, width="stretch")

# ---------------------------------------------------------------- monthly
with tabs[2]:
    st.subheader("Weekly and monthly view")
    summ = res["summary"]
    st.dataframe(summ.drop(columns=[c for c in ("status",) if c in summ]), hide_index=True, width="stretch")
    a, b = st.columns(2)
    per = plan.drop_duplicates("task_id").groupby(["week", "dept"]).size().rename("tasks").reset_index()
    a.plotly_chart(px.bar(per, x="week", y="tasks", color="dept", color_discrete_map=DEPT_COLORS,
                          title="Tasks scheduled per week and department"), width="stretch")
    heat = plan.drop_duplicates("task_id").groupby(["section_id", "week"]).size().rename("tasks").reset_index()
    b.plotly_chart(px.density_heatmap(heat, x="week", y="section_id", z="tasks", histfunc="sum",
                                      color_continuous_scale="Blues", title="Tasks by section and week"),
                   width="stretch")
    if res["realization"] is not None:
        r = res["realization"]
        st.markdown("**Monthly plan vs what the weekly plans did** (relative to the initial monthly assignment)")
        st.dataframe(r.status.value_counts().rename("tasks").reset_index(), hide_index=True)
    if res["capacity"] is not None:
        st.markdown("**Why capacity is left over** (eligible = at least one compatible window that week)")
        st.dataframe(res["capacity"], hide_index=True, width="stretch")
        cap = res["capacity"]
        st.caption(f"Free window time by week: {', '.join(f'{x:.0f}%' for x in cap.free_pct)}. "
                   "The backlog is small and released gradually by the 14-day start rule, so later weeks are "
                   "demand-limited, not constraint-limited.")

# ---------------------------------------------------------------- what-if
with tabs[3]:
    st.subheader("What-if: emergency defect")
    st.caption("Re-optimises the chosen week with the emergency task forced in and compares it with the current plan. "
               "This preview has NO stability objective, so unrelated tasks may move - the 'Rolling re-plan' tab "
               "keeps committed blocks and prices changes.")
    f1, f2, f3 = st.columns(3)
    dept = f1.selectbox("Department", list(DEPT_COLORS), key="wi_dept")
    section = f2.selectbox("Section", sorted(T.section_id.unique()), key="wi_section")
    row = sections[sections.section_id == section].iloc[0]
    km = f3.slider("Location (km)", float(row.km_start), float(row.km_end), float(row.km_start) + 5.0, 0.5, key="wi_km")
    g1, g2, g3 = st.columns(3)
    dur = g1.slider("Work duration (min)", 30, 150, 90, 10, key="wi_dur")
    score = g2.slider("Priority score", 0.5, 1.0, 0.95, 0.05, key="wi_score")
    mode = g3.selectbox("Where", ["Scan all weeks", "Week 1", "Week 2", "Week 3", "Week 4"], key="wi_mode")
    if st.button("Preview impact", key="wi_btn"):
        emg = wi.make_emergency_task(T, dept, section, km, dur, priority_score=score)
        with st.spinner("Re-optimising..."):
            if mode == "Scan all weeks":
                scan = wi.scan_weeks(T, windows, plan, blocks, emg, weeks=sorted(plan.week.unique()))
                st.markdown("**The same defect in each week**")
                st.dataframe(scan, hide_index=True, width="stretch")
                ok = scan[scan.feasible]
                if ok.empty:
                    st.error("No week can take this task (too long for every window, or no power-capable window).")
                else:
                    best_nv = ok.sort_values(["net_value_change", "tasks_moved"], ascending=[False, True]).iloc[0]
                    least = ok.sort_values(["cost_to_rest", "tasks_moved"], ascending=[False, True]).iloc[0]
                    st.success(f"Highest overall net value: week {int(best_nv.week)} "
                               f"({int(best_nv.net_value_change):+d}, {int(best_nv.tasks_moved)} other tasks moved).")
                    if int(least.week) != int(best_nv.week):
                        st.info(f"Least cost to the rest of the plan: week {int(least.week)} "
                                f"({int(least.cost_to_rest)} points) - but the defect would wait longer, and "
                                "waiting lowers its own value (delay penalty).")
            else:
                r = wi.impact_in_week(T, windows, plan, blocks, int(mode.split()[1]), emg)
                if not r["feasible"]:
                    st.error(r["reason"])
                else:
                    s = r["summary"]
                    k = st.columns(4)
                    k[0].metric("Blocks", s["blocks_after"], f"{s['blocks_added']:+d}", delta_color="inverse")
                    k[1].metric("Trains disrupted", s["trains_after"], f"{s['trains_after'] - s['trains_before']:+d}",
                                delta_color="inverse")
                    k[2].metric("Cost to the rest of the plan", s["cost_to_rest_of_plan"])
                    k[3].metric("Tasks moved / displaced", f"{s['tasks_moved']} / {s['tasks_displaced']}")
                    st.markdown("**Emergency task placement**")
                    st.dataframe(r["emergency"], hide_index=True)
                    if s["tasks_moved"] > 10:
                        st.warning(f"{s['tasks_moved']} other tasks were re-assigned to other blocks - the week was fully "
                                   "re-optimised. A planner would want fewer changes (Step 6).")
                    if len(r["displaced"]):
                        st.markdown("**Displaced (no longer scheduled this week)**")
                        st.dataframe(r["displaced"], hide_index=True)
                    if len(r["moved"]):
                        with st.expander(f"Moved tasks ({len(r['moved'])})"):
                            st.dataframe(r["moved"], hide_index=True)
                    gantt(r["new_plan"], f"Week {r['week']} after the emergency task")

# ---------------------------------------------------------------- rolling re-plan
with tabs[4]:
    st.subheader("Rolling re-planning: new defects and emergencies")
    st.caption("At the start of the arrival day, blocks of earlier days are executed, blocks inside the lock window are "
               "frozen, the next 14 days are re-optimised with a price on every change, and later blocks are left alone. "
               "Three strategies on the same events: stable (this system), full re-optimisation, and a manual-style patch.")
    summ_path = ROOT / "outputs" / "rolling_summary.csv"
    if summ_path.exists():
        rs = pd.read_csv(summ_path)
        st.markdown("**Scenario run by `python replan.py`** (5 events: 3 emergencies, 4 routine defects, synthetic)")
        cols = ["strategy", "total_moved", "total_dropped", "total_blocks_changed", "broken_commitments",
                "mean_emergency_response_days", "tasks_scheduled", "blocks", "trains_disrupted", "net_value"]
        st.dataframe(rs[cols], hide_index=True, width="stretch")
        ev_path = ROOT / "outputs" / "rolling_events.csv"
        if ev_path.exists():
            with st.expander("Per event"):
                st.dataframe(pd.read_csv(ev_path), hide_index=True, width="stretch")
        sens_path = ROOT / "outputs" / "rolling_sensitivity.csv"
        if sens_path.exists():
            sens = pd.read_csv(sens_path)
            mp = sens[sens.vary == "move_penalty"]
            a1, a2 = st.columns(2)
            a1.plotly_chart(px.line(mp, x="value", y="total_moved", markers=True,
                                    title="Tasks moved vs price per move", labels={"value": "move penalty"}),
                            width="stretch")
            a2.plotly_chart(px.line(mp, x="value", y="net_value", markers=True,
                                    title="Net value vs price per move", labels={"value": "move penalty"}),
                            width="stretch")
    else:
        st.info("Run `python replan.py` to produce the scenario results.")

    st.markdown("---")
    st.markdown("**Try one event on the saved plan**")
    r1, r2, r3 = st.columns(3)
    now_day = r1.slider("Arrival day (0 = first day)", 0, 20, 3, key="rp_day")
    r_dept = r2.selectbox("Department", list(DEPT_COLORS), key="rp_dept")
    r_sec = r3.selectbox("Section", sorted(T.section_id.unique()), key="rp_section")
    q1, q2, q3, q4 = st.columns(4)
    r_dur = q1.slider("Duration (min)", 30, 150, 90, 10, key="rp_dur")
    r_lock = q2.slider("Lock days", 0, 4, rp.LOCK_DAYS, key="rp_lock")
    r_move = q3.slider("Price per move", 0, 300, rp.MOVE_PENALTY, 10, key="rp_move")
    r_kind = q4.selectbox("Kind", ["Emergency", "Routine"], key="rp_kind")
    if st.button("Compare strategies", key="rp_btn"):
        srow = sections[sections.section_id == r_sec].iloc[0]
        emg = wi.make_emergency_task(
            T, r_dept, r_sec, float(srow.km_start) + 5.0, r_dur, task_id="EMG-UI" if r_kind == "Emergency" else "NEW-UI",
            overdue_days=1 - now_day if r_kind == "Emergency" else -(now_day + 15),
            priority_score=0.95 if r_kind == "Emergency" else 0.45, tier="CRITICAL" if r_kind == "Emergency" else "MEDIUM",
            severity=5 if r_kind == "Emergency" else 3)
        event = [dict(day=now_day, kind="emergency" if r_kind == "Emergency" else "routine", tasks=emg)]
        out_rows, plans = [], {}
        with st.spinner("Re-planning with three strategies..."):
            for strat in ("stable", "full", "patch"):
                kw = dict(lock_days=r_lock, move_penalty=r_move) if strat == "stable" else {}
                ev, sm, p, b, _ = rp.simulate(T, windows, plan, event, strat, workers=8, time_limit=15, **kw)
                e = ev.iloc[0]
                out_rows.append(dict(strategy=strat, moved=e.moved, dropped=e.dropped, blocks_changed=e.blocks_changed,
                                     broken_commitments=e.broken_commitments, new_blocks=e.blocks_opened,
                                     trains_change=e.trains_change, emergency_response_days=e.emergency_response_days,
                                     net_value=e.net_value, valid=(e.violations == 0 and sm["final_violations"] == 0)))
                plans[strat] = p
        st.dataframe(pd.DataFrame(out_rows), hide_index=True, width="stretch")
        st.caption("broken_commitments is counted against a 2-day lock. 'stable' keeps committed blocks and prices every "
                   "move; 'full' may move anything; 'patch' only adds the new task.")
        view = plans["stable"]
        view = view[(view.date >= o.START + pd.Timedelta(days=now_day))
                    & (view.date < o.START + pd.Timedelta(days=now_day + rp.LOOKAHEAD_DAYS))]
        gantt(view, f"Stable plan, days {now_day}-{now_day + rp.LOOKAHEAD_DAYS - 1}")

# ---------------------------------------------------------------- unscheduled
with tabs[5]:
    st.subheader("Tasks not scheduled after the final week")
    u = res["unscheduled"]
    st.write(f"**{len(u)} of {len(T)} tasks** are unscheduled: "
             f"{(u.group == 'NEVER_PLACEABLE').sum()} can never be placed with the current windows, "
             f"{(u.group == 'PLACEABLE_NOT_SCHEDULED').sum()} could be placed but the objective declines them.")
    st.dataframe(u.groupby(["group", "reason", "tier"]).size().rename("tasks").reset_index(), hide_index=True)
    f = st.selectbox("Filter", ["All"] + sorted(u.reason.unique()), key="un_filter")
    st.dataframe(u if f == "All" else u[u.reason == f], hide_index=True, width="stretch")
    tl = u[u.reason == "TOO_LONG"]
    if len(tl):
        short = tl.recommended_action.str.extract(r"\+(\d+)")[0].astype(int)
        st.warning(f"{len(tl)} tasks need an extended block (shortfall {short.min()}-{short.max()} min). "
                   "Ask the control office for longer windows.")
    if res["audit"] is not None:
        st.markdown("**Objective audit** - recomputed value vs the value the solver reported")
        st.dataframe(res["audit"], hide_index=True)

# ---------------------------------------------------------------- export
with tabs[6]:
    st.subheader("BDMS-style export")
    st.caption("Illustrative layout - the real BDMS fields are not available. Recommendations only: every request "
               "needs official approval.")
    req = bx.to_bdms(plan, blocks)
    ew = st.selectbox("Week", ["All"] + sorted(plan.week.unique()), key="ex_week")
    if ew != "All":
        wb, wp = blocks[blocks.week == ew], plan[plan.week == ew]
        req = bx.to_bdms(wp, wb)
    else:
        wb, wp = blocks, plan
    st.dataframe(req, hide_index=True, width="stretch")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Block requests (CSV)", req.to_csv(index=False), "block_requests.csv", "text/csv", key="dl_csv")
    d2.download_button("Block requests (JSON)", bx.to_json(wp, wb), "block_requests.json", "application/json", key="dl_json")
    d3.download_button("Work items (CSV)", bx.work_items(wp).to_csv(index=False), "work_items.csv", "text/csv", key="dl_items")
