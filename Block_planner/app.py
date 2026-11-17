"""
RailOpt - Automatic Block Planning dashboard (Step 5).

Run:  streamlit run Block_planner/app.py   (needs Block_planner/outputs/ from `python horizons.py`)

Everything shown is built from SYNTHETIC data. The manual baseline is a SIMULATION of decentralised
planning under stated assumptions (see baseline.py). The system recommends; officials approve.

Layout: one Streamlit page per question a planner asks, grouped in the sidebar. Styling, colours and
plain-language labels live in ui.py.
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
import ui
import whatif as wi

ROOT = Path(__file__).parent
DEPT_COLORS = ui.DEPT_COLORS
CC = st.column_config

st.set_page_config(page_title="RailOpt - Automatic Block Planning", page_icon="🚆", layout="wide")
ui.setup()


# ================================================================ data
@st.cache_data(show_spinner="Loading the plan...")
def load_all():
    T, windows = m.load_inputs()
    return T, windows, m.load_results(), pd.read_csv(o.DATA / "sections.csv")


@st.cache_data(show_spinner="Simulating manual planning (6 submission orders)...")
def manual_runs():
    T, windows = m.load_inputs()
    base, plans = bl.run_orders(T, windows)
    return base, plans


if not (ROOT / "outputs" / "block_plan_4w.csv").exists():
    ui.hero("RailOpt", "Automatic railway maintenance block planning")
    st.error("**No plan found yet.** Generate one from the repository root, then reload this page:")
    st.code("python Block_planner/horizons.py --no-ablation", language="bash")
    st.stop()

T, windows, res, sections = load_all()
plan, blocks = m.with_week(res["plan"]), m.with_week(res["blocks"])
base, mplans = manual_runs()
ai = m.kpis(plan, blocks, T, windows)
cmp = bl.compare(ai, base)
mean = base.select_dtypes("number").mean()
best = bl.best_order(base)
WEEKS = sorted(int(w) for w in plan.week.unique())
SECTIONS = sorted(T.section_id.unique())
END = o.START + pd.Timedelta(days=7 * len(WEEKS) - 1)


# ================================================================ helpers
def pct(a, b):
    return None if b in (None, 0) or a is None else (a - b) / abs(b) * 100


def delta(a, b, suffix=" vs manual"):
    p = pct(a, b)
    return None if p is None else f"{p:+.0f}%{suffix}"


def tidy(fig):
    """Drop plotly-express legend titles (they collide with chart titles) and keep the legend top-left."""
    fig.update_layout(legend_title_text=None, legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                      uniformtext_minsize=10, uniformtext_mode="hide")
    fig.update_traces(textangle=0, selector=dict(type="bar"))
    return fig


def gantt(p, title, height_per_lane=24):
    if p.empty:
        st.info("Nothing to show for this selection - try more sections or another week.")
        return
    d = p.copy()
    d["start"] = d.date + pd.to_timedelta(d.start_min, unit="m")
    d["end"] = d.date + pd.to_timedelta(d.end_min, unit="m")
    d["Section"] = d.section_id + "  ·  " + d.dept
    d["Department"] = d.dept.map(ui.DEPT_NAMES)
    d["Time"] = d.start_min.map(ui.hhmm) + "-" + d.end_min.map(ui.hhmm)
    d = d.sort_values(["section_id", "dept"])
    fig = px.timeline(d, x_start="start", x_end="end", y="Section", color="dept", color_discrete_map=DEPT_COLORS,
                      custom_data=["task_id", "task", "tier", "km", "Time", "Department"], title=title,
                      category_orders={"Section": list(d.Section.drop_duplicates())})
    fig.update_traces(hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[0]} · %{customdata[5]}<br>"
                                    "Tier: %{customdata[2]} · km %{customdata[3]}<br>%{x|%a %d %b}, "
                                    "%{customdata[4]}<extra></extra>", marker_line_width=0)
    lanes = list(d.Section.drop_duplicates())
    for k, sec in enumerate(d.section_id.drop_duplicates()):          # light band behind every other section
        if k % 2 == 0:
            idx = [i for i, lane in enumerate(lanes) if lane.startswith(sec + "  ")]
            fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=min(idx) - 0.5, y1=max(idx) + 0.5,
                          fillcolor="#EEF3F8", line_width=0, layer="below")
    fig.update_yaxes(autorange=True, title=None)
    fig.update_xaxes(title=None, tickformat="%a %d %b", showgrid=True)
    fig.update_layout(height=max(360, height_per_lane * d.Section.nunique() + 140), showlegend=False)
    st.plotly_chart(tidy(fig), width="stretch")


def task_table(p):
    t = p.sort_values(["date", "start_min"]).assign(
        day=lambda d: d.date.dt.strftime("%a %d %b"), start=lambda d: d.start_min.map(ui.hhmm),
        end=lambda d: d.end_min.map(ui.hhmm))
    st.dataframe(t[["day", "start", "end", "section_id", "dept", "task_id", "task", "tier", "km"]], hide_index=True,
                 width="stretch", column_config={
                     "day": "Day", "start": "Start", "end": "End", "section_id": "Section", "dept": "Dept",
                     "task_id": "Task ID", "task": "Work", "tier": "Priority", "km": CC.NumberColumn("km", format="%.1f")})


def week_label(w):
    s = o.START + pd.Timedelta(days=7 * (int(w) - 1))
    return f"Week {int(w)}  ({s:%d %b} - {s + pd.Timedelta(days=6):%d %b})"


# ================================================================ pages
def page_start():
    ui.hero("RailOpt: one coordinated plan for railway maintenance blocks",
            "Today Engineering, Signal & Telecom and Traction each book their own track closures. RailOpt plans them "
            "together, so the same work needs fewer closures and stops fewer trains.")
    fewer_b = -pct(ai["blocks"], mean.blocks)
    fewer_t = -pct(ai["trains_disrupted"], mean.trains_disrupted)
    ui.section("The story in 30 seconds")
    ui.steps([("The problem", "Three departments plan separately, so the line is closed again and again for work "
                              "that could share one closure."),
              ("What RailOpt does", "Collects all maintenance tasks, ranks them by urgency and risk, and fits them into "
                                    "the corridor windows the control office allows."),
              ("The result", f"{fewer_b:.0f}% fewer blocks and {fewer_t:.0f}% fewer trains disrupted than manual "
                             "planning, with all overdue work still scheduled.")])

    ui.section("5-minute demo tour", "Open each page in order. The line in quotes is what you can say while showing it.")
    tour = [
        ("Overview", "overview", "Point at the two columns, manual vs RailOpt.",
         f"Same work, same safety rules. RailOpt needs {ai['blocks']} closures instead of about {mean.blocks:.0f}."),
        ("Block calendar", "calendar", "Pick one week, then switch Plan from RailOpt to Manual.",
         "Each bar is a job. Where colours line up, departments share one closure instead of booking three."),
        ("Weekly & monthly view", "monthly", "Show the four week cards.",
         "There is a monthly plan and a detailed weekly plan. Overdue work goes first, in week 1."),
        ("Emergency what-if", "what-if", "Press Preview impact with the default defect.",
         "An urgent defect comes in. The planner shows which week to put it in and what it costs the rest."),
        ("Unscheduled work", "unscheduled", "Show the orange action box.",
         "Nothing is dropped silently. Some jobs are longer than any window, so the control office needs to "
         "give longer blocks."),
    ]
    for i, (title, path, show, say) in enumerate(tour, 1):
        a, b = st.columns([5, 1], vertical_alignment="center")
        with a:
            ui.tour_step(i, title, show, say)
        b.page_link(PAGES[path], label="Open", icon="➡️")

    ui.section("Tips for presenting")
    st.markdown("- **Explain mode** (sidebar) adds a short *what you see / what to look at / why it matters* box to "
                "every page. Keep it on for a first look; switch it off for a cleaner screen.\n"
                "- Hover over any **ⓘ** icon or chart bar for details. The sidebar **Glossary** explains terms such as "
                "*block*, *corridor window* and *net value*.\n"
                "- If someone asks *is this real data?*: no. It is synthetic data shaped like Indian Railways systems "
                "(TMS, SMMS, TDMS, COA). The method works the same with real exports.")


def page_overview():
    ui.header("🏠 Overview", f"{len(WEEKS)} weeks, {o.START:%d %b} - {END:%d %b %Y}: {len(T)} maintenance tasks on "
                            f"{len(SECTIONS)} sections, planned by RailOpt and by simulated manual planning.")
    ui.explain("overview")

    def d(a, b, lower_better=True, fmt="{:+.0f}%"):
        p = pct(a, b)
        return fmt.format(p), (p < 0) == lower_better

    rows = []
    for label, a, b, lb in [("Blocks opened (track closures)", mean.blocks, ai["blocks"], True),
                            ("Trains disrupted", mean.trains_disrupted, ai["trains_disrupted"], True)]:
        txt, good = d(b, a, lb)
        rows.append((label, f"{a:,.0f}", f"{b:,}", txt, good))
    rows.append(("Shared blocks (2+ departments)", "0", str(ai["merged_blocks"]),
                 f"{ai['merged_blocks'] / max(ai['blocks'], 1):.0%} of blocks", True))
    rows.append(("Overdue tasks scheduled", ai["overdue_cleared"], ai["overdue_cleared"], "all of them", True))
    rows.append(("Tasks scheduled", f"{mean.tasks_scheduled:.0f}", str(ai["tasks_scheduled"]),
                 f"{ai['tasks_scheduled'] - mean.tasks_scheduled:+.0f}, on purpose", False))
    ui.versus("Manual planning today (simulated)", "RailOpt", rows)

    ui.takeaways([
        f"<b>Fewer closures:</b> departments share {ai['merged_blocks']} blocks, so the line closes "
        f"{mean.blocks - ai['blocks']:.0f} fewer times.",
        f"<b>Fewer trains stopped:</b> work is put in quieter windows, {mean.trains_disrupted - ai['trains_disrupted']:.0f} "
        "fewer train disruptions.",
        f"<b>The trade-off:</b> {mean.tasks_scheduled - ai['tasks_scheduled']:.0f} low-priority jobs wait for next month, "
        "because a closure just for them costs more than the work is worth now.",
    ], title="What this means")

    with st.expander("📊 All 15 measures, and how the comparison works"):
        show = cmp.drop(columns="key").assign(verdict=lambda d: d.verdict.map(ui.VERDICTS).fillna(d.verdict),
                                              kpi=lambda d: d.kpi.str.strip())
        st.dataframe(show, hide_index=True, width="stretch", height=420, column_config={
            "kpi": CC.TextColumn("Measure", width="medium"), "optimised": CC.NumberColumn("RailOpt"),
            "manual_mean": CC.NumberColumn("Manual (mean)", format="%.1f"),
            "manual_best_order": CC.NumberColumn("Manual (best)"),
            "manual_worst_order": CC.NumberColumn("Manual (worst)"),
            "verdict": CC.TextColumn("Verdict", width="large")})
        st.markdown(
            "- **Manual** simulates today's process: each department books its own blocks and never shares them. It "
            "is run in all 6 orders departments could submit in, because the order changes who gets the best windows. "
            "The left column above is the average of the 6.\n"
            "- Both plans use the **same data, the same hard safety rules and the same scoring code**.\n"
            "- The manual results depend on the assumptions in `baseline.py` (M1-M6), above all that departments never "
            "share a block.\n- *Task-days past due* is a lateness proxy, not measured asset downtime.")


def page_calendar():
    ui.header("📅 Block calendar", "When and where each block is opened. One bar = one task; bars of different "
                                   "departments at the same time on the same section share one block.")
    ui.explain("calendar")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1.3, 1.4, 2])
        source = c1.segmented_control("Plan", ["RailOpt", "Manual"], default="RailOpt", key="cal_source",
                                      help=f"Manual shows the best of the 6 simulated orders ({best}).") or "RailOpt"
        week = c2.selectbox("Week", WEEKS + ["All weeks"], format_func=lambda w: w if w == "All weeks" else week_label(w),
                            key="cal_week")
        depts = c3.multiselect("Departments", list(DEPT_COLORS), default=list(DEPT_COLORS), key="cal_dept")
        secs = c4.multiselect("Sections", SECTIONS, default=SECTIONS[:3], key="cal_sec",
                              placeholder="All sections")
    secs = secs or SECTIONS
    if source == "RailOpt":
        p, b = plan, blocks
    else:
        p, b = m.with_week(mplans[best][0]), m.with_week(mplans[best][1])
    p = p[p.dept.isin(depts) & p.section_id.isin(secs)]
    if week != "All weeks":
        p = p[p.week == week]
    b = b[b.window_id.isin(p.window_id) & b.date.isin(p.date)]

    k = st.columns(4)
    k[0].metric("Tasks", p.task_id.nunique())
    k[1].metric("Blocks", len(b))
    k[2].metric("Joint blocks", int(b.merged.sum()) if "merged" in b and len(b) else 0)
    k[3].metric("Trains disrupted", int(b.trains_affected.sum()) if len(b) else 0)
    ui.dept_legend()
    gantt(p, f"{source} plan · {week if week == 'All weeks' else week_label(week)}")
    with st.expander(f"📋 Task list ({len(p)} rows)", expanded=False):
        task_table(p)
        st.download_button("Download this list (CSV)", p.to_csv(index=False), "block_calendar_selection.csv",
                           "text/csv", key="cal_dl")


def page_monthly():
    ui.header("🗓️ Weekly & monthly view", "How the month is split into weeks, and how much work each week, "
                                          "department and section gets.")
    ui.explain("monthly")
    summ = res["summary"]
    cols = st.columns(len(summ))
    for c, r in zip(cols, summ.itertuples()):
        with c.container(border=True):
            st.markdown(f"**{week_label(r.week)}**")
            st.markdown(f"<span style='font-size:1.6rem;color:{ui.PRIMARY};font-weight:700'>{r.scheduled}</span> tasks "
                        f"in <b>{r.blocks}</b> blocks", unsafe_allow_html=True)
            st.caption(f"{r.merged_blocks} joint · {r.trains_disrupted} trains · {r.carried_over} carried over")

    a, b = st.columns(2)
    per = plan.drop_duplicates("task_id").groupby(["week", "dept"]).size().rename("tasks").reset_index()
    fig = px.bar(per, x="week", y="tasks", color="dept", color_discrete_map=DEPT_COLORS, text_auto=True,
                 title="Tasks per week, by department")
    fig.update_layout(xaxis=dict(dtick=1, title="Week"), yaxis_title="Tasks", height=380)
    a.plotly_chart(tidy(fig), width="stretch")
    heat = (plan.drop_duplicates("task_id").groupby(["section_id", "week"]).size().unstack(fill_value=0)
            .reindex(columns=WEEKS, fill_value=0))
    fig = px.imshow(heat, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                    labels=dict(x="Week", y="Section", color="Tasks"), title="Tasks by section and week")
    fig.update_layout(height=380, coloraxis_showscale=False, xaxis=dict(dtick=1))
    b.plotly_chart(tidy(fig), width="stretch")

    if res["realization"] is not None:
        ui.section("Did the weekly plans follow the monthly plan?",
                   "The monthly plan assigns each task to a week; each week is then re-planned in detail.")
        r = res["realization"].status.map(ui.REALIZATION).fillna(res["realization"].status).value_counts()
        cols = st.columns(len(r))
        for c, (label, n) in zip(cols, r.items()):
            c.metric(label, n, f"{n / r.sum():.0%} of tasks", delta_color="off", delta_arrow="off")

    with st.expander("🔧 Detailed weekly solver results"):
        st.dataframe(summ.drop(columns=[c for c in ("status",) if c in summ]), hide_index=True, width="stretch",
                     column_config={"week": "Week", "pool": "Candidates", "backfilled": "Added late",
                                    "scheduled": "Scheduled", "carried_over": "Carried over", "blocks": "Blocks",
                                    "solver_objective": "Solver value", "merged_blocks": "Joint blocks",
                                    "trains_disrupted": "Trains", "net_value": "Net value",
                                    "rel_gap": CC.NumberColumn("Optimality gap", format="%.3f"),
                                    "proven_optimal": "Proven optimal", "violations": "Rule violations",
                                    "wall_time_s": CC.NumberColumn("Solve time (s)", format="%.1f")})
    if res["capacity"] is not None:
        with st.expander("📦 Why is window capacity left over?"):
            cap = res["capacity"]
            st.markdown(f"Free window time by week: **{', '.join(f'{x:.0f}%' for x in cap.free_pct)}**. The backlog is "
                        "small and the 14-day earliest-start rule releases work gradually, so later weeks are limited "
                        "by how much work exists, not by the windows.")
            st.dataframe(cap, hide_index=True, width="stretch", column_config={
                "free_pct": CC.ProgressColumn("Free window time", format="%.0f%%", min_value=0, max_value=100),
                "crew_util_pct": CC.ProgressColumn("Crew utilisation", format="%.0f%%", min_value=0, max_value=100)})


def page_whatif():
    ui.header("⚡ Emergency what-if", "A new urgent defect comes in. Where should it go, and what does it cost "
                                     "the rest of the plan?")
    ui.explain("whatif")
    st.caption("Quick preview: the chosen week is re-planned from scratch, so unrelated tasks may move. "
               "The Rolling re-plan page shows the stability-aware response.")
    with st.form("wi_form", border=True):
        st.markdown("**1 · Describe the defect**")
        f1, f2, f3 = st.columns(3)
        dept = f1.selectbox("Department", list(DEPT_COLORS), format_func=lambda d: f"{d} · {ui.DEPT_NAMES[d]}",
                            key="wi_dept")
        section = f2.selectbox("Section", SECTIONS, key="wi_section")
        km_off = f3.slider("Distance into section (km)", 0.0, 30.0, 5.0, 0.5, key="wi_km",
                           help="Clipped to the section's length.")
        g1, g2 = st.columns(2)
        dur = g1.slider("Work duration (minutes)", 30, 150, 90, 10, key="wi_dur")
        score = g2.slider("Priority score", 0.5, 1.0, 0.95, 0.05, key="wi_score",
                          help="0.95 is typical for a safety-critical emergency.")
        st.markdown("**2 · Choose where to try it**")
        mode = st.radio("Where", ["Compare all weeks"] + [f"Week {w}" for w in WEEKS], horizontal=True,
                        key="wi_mode", label_visibility="collapsed")
        go = st.form_submit_button("Preview impact", type="primary", key="wi_btn")

    if go:
        row = sections[sections.section_id == section].iloc[0]
        km = min(float(row.km_start) + km_off, float(row.km_end))
        emg = wi.make_emergency_task(T, dept, section, km, dur, priority_score=score)
        with st.spinner("Re-optimising..."):
            if mode == "Compare all weeks":
                st.session_state.wi_result = ("scan", wi.scan_weeks(T, windows, plan, blocks, emg, weeks=WEEKS))
            else:
                st.session_state.wi_result = ("week", wi.impact_in_week(T, windows, plan, blocks,
                                                                         int(mode.split()[1]), emg))
    if "wi_result" not in st.session_state:
        st.caption("Set the defect above and press **Preview impact**.")
        return

    kind, r = st.session_state.wi_result
    st.markdown("**3 · Result**")
    if kind == "scan":
        ok = r[r.feasible]
        if ok.empty:
            st.error("No week can take this task: it is longer than every window, or needs a power block that no "
                     "window allows.")
        else:
            best_nv = ok.sort_values(["net_value_change", "tasks_moved"], ascending=[False, True]).iloc[0]
            least = ok.sort_values(["cost_to_rest", "tasks_moved"], ascending=[False, True]).iloc[0]
            st.success(f"**Recommended: week {int(best_nv.week)}** - best overall net value "
                       f"({int(best_nv.net_value_change):+,}), {int(best_nv.tasks_moved)} other tasks moved.")
            if int(least.week) != int(best_nv.week):
                st.info(f"Least disruption to other work: week {int(least.week)} ({int(least.cost_to_rest):,} points). "
                        "But the defect would wait longer, and waiting lowers its own value.")
            fig = px.bar(ok.assign(Week=ok.week.map(lambda w: f"Week {int(w)}")), x="Week", y="net_value_change",
                         text_auto=",.0f", title="Change in plan net value if the defect goes in each week")
            fig.update_traces(marker_color=[ui.ACCENT if w == best_nv.week else ui.MUTED for w in ok.week])
            fig.update_layout(yaxis_title="Net value change", xaxis_title=None, height=320)
            st.plotly_chart(tidy(fig), width="stretch")
        st.dataframe(r, hide_index=True, width="stretch", column_config={
            "week": "Week", "feasible": "Possible?", "net_value_change": "Net value change",
            "cost_to_rest": "Cost to other work", "blocks_added": "Extra blocks", "trains_added": "Extra trains",
            "tasks_displaced": "Tasks bumped out", "tasks_moved": "Tasks moved", "proven": "Proven optimal"})
        return

    if not r["feasible"]:
        st.error(f"Week {r['week']} cannot take this task: {r['reason']}.")
        return
    s = r["summary"]
    k = st.columns(4)
    k[0].metric("Blocks", s["blocks_after"], f"{s['blocks_added']:+d}", delta_color="inverse")
    k[1].metric("Trains disrupted", s["trains_after"], f"{s['trains_after'] - s['trains_before']:+d}",
                delta_color="inverse")
    k[2].metric("Cost to other work", f"{s['cost_to_rest_of_plan']:,}", help="Net-value points lost by the rest "
                                                                                "of the week's plan.")
    k[3].metric("Tasks moved / bumped", f"{s['tasks_moved']} / {s['tasks_displaced']}")
    e = r["emergency"].assign(start=lambda d: d.start_min.map(ui.hhmm), end=lambda d: d.end_min.map(ui.hhmm))
    slot = e.iloc[0]
    st.success(f"**Emergency placed:** {slot.slot.split(' ')[0]} {slot.slot.split(' ')[1]} on {slot.section_id}, "
               f"{slot.start}-{slot.end}.")
    if s["tasks_moved"] > 10:
        st.warning(f"{s['tasks_moved']} other tasks were re-assigned because the whole week was re-optimised. "
                   "Rolling re-plan keeps most of them in place.")
    if len(r["displaced"]):
        st.markdown("**Bumped out of this week**")
        st.dataframe(r["displaced"], hide_index=True, width="stretch")
    if len(r["moved"]):
        with st.expander(f"Moved tasks ({len(r['moved'])})"):
            st.dataframe(r["moved"], hide_index=True, width="stretch",
                         column_config={"task_id": "Task", "tier": "Priority", "dept": "Dept", "was": "Was", "now": "Now"})
    gantt(r["new_plan"], f"Week {r['week']} with the emergency task")


def page_rolling():
    ui.header("🔄 Rolling re-plan", "Defects keep arriving. How much does each strategy disturb a plan that "
                                   "crews and officials have already agreed?")
    ui.explain("rolling")
    desc = {"stable": ("🟦", ui.PRIMARY, "Freezes the next 2 days, re-plans the next 14 and charges a price for every "
                                         "task it moves."),
            "full": ("🟧", ui.ACCENT, "Re-plans everything from scratch. Best score, but moves work crews were already "
                                      "told about."),
            "patch": ("⬜", ui.MUTED, "Squeezes the new task in where it fits. Nothing else moves, but it needs extra "
                                      "blocks.")}
    summ_path = ROOT / "outputs" / "rolling_summary.csv"
    if summ_path.exists():
        rs = pd.read_csv(summ_path).set_index("strategy")
        ui.section("Three ways to respond", "Results on a saved test: 5 synthetic events (3 emergencies, 4 routine "
                                            "defects) over the 4 weeks.")
        c = st.columns(3)
        for col, key in zip(c, ("stable", "full", "patch")):
            icon, color, text = desc[key]
            r = rs.loc[key, ["total_moved", "broken_commitments", "blocks", "trains_disrupted", "net_value"]].astype(int)
            col.markdown(
                f'<div class="ro-card" style="border-top:4px solid {color}"><h4>{icon} {ui.STRATEGIES[key]}</h4>'
                f'<p>{text}</p><p style="margin-top:10px"><span style="font-size:1.6rem;font-weight:700;'
                f'color:{ui.PRIMARY}">{r.total_moved}</span> tasks moved &nbsp;·&nbsp; '
                f'<b>{r.broken_commitments}</b> promises broken<br><span style="color:#8592A0;font-size:.85rem">'
                f'{r.blocks} blocks · {r.trains_disrupted} trains · net value {r.net_value:,}</span></p></div>',
                unsafe_allow_html=True)
        rs = rs.reset_index()
        with st.expander("📋 Full results table"):
            cols = ["strategy", "total_moved", "total_dropped", "total_blocks_changed", "broken_commitments",
                    "mean_emergency_response_days", "tasks_scheduled", "blocks", "trains_disrupted", "net_value"]
            st.dataframe(rs[cols].assign(strategy=rs.strategy.map(ui.STRATEGIES).fillna(rs.strategy)),
                         hide_index=True, width="stretch", column_config={
                             "strategy": "Strategy", "total_moved": "Tasks moved", "total_dropped": "Tasks dropped",
                             "total_blocks_changed": "Blocks changed", "broken_commitments": "Broken commitments",
                             "mean_emergency_response_days": CC.NumberColumn("Emergency response (days)", format="%.2f"),
                             "tasks_scheduled": "Tasks scheduled", "blocks": "Blocks", "trains_disrupted": "Trains",
                             "net_value": CC.NumberColumn("Net value", format="%d")})
        ev_path, sens_path = ROOT / "outputs" / "rolling_events.csv", ROOT / "outputs" / "rolling_sensitivity.csv"
        if ev_path.exists():
            with st.expander("📋 Results per event"):
                st.dataframe(pd.read_csv(ev_path), hide_index=True, width="stretch")
        if sens_path.exists():
            sens = pd.read_csv(sens_path)
            mpen = sens[sens.vary == "move_penalty"]
            with st.expander("📈 How the price per move trades stability for value"):
                a1, a2 = st.columns(2)
                fig = px.line(mpen, x="value", y="total_moved", markers=True, title="Tasks moved")
                fig.update_layout(xaxis_title="Price per move", yaxis_title=None, height=300)
                a1.plotly_chart(tidy(fig), width="stretch")
                fig = px.line(mpen, x="value", y="net_value", markers=True, title="Net value")
                fig.update_traces(line_color=ui.ACCENT)
                fig.update_layout(xaxis_title="Price per move", yaxis_title=None, height=300)
                a2.plotly_chart(tidy(fig), width="stretch")
    else:
        st.info("Run `python Block_planner/replan.py` to produce the saved scenario.")

    with st.expander("🧪 Try your own event: add one defect and compare the three strategies", expanded="rp_result" in st.session_state):
        with st.form("rp_form", border=True):
            r1, r2, r3, r4 = st.columns(4)
            r_kind = r1.segmented_control("Kind", ["Emergency", "Routine"], default="Emergency", key="rp_kind")
            r_dept = r2.selectbox("Department", list(DEPT_COLORS), key="rp_dept",
                                  format_func=lambda d: f"{d} · {ui.DEPT_NAMES[d]}")
            r_sec = r3.selectbox("Section", SECTIONS, key="rp_section")
            now_day = r4.slider("Arrives on day", 0, 20, 3, key="rp_day", help="Day 0 = first day of the plan.")
            q1, q2, q3 = st.columns(3)
            r_dur = q1.slider("Duration (minutes)", 30, 150, 90, 10, key="rp_dur")
            r_lock = q2.slider("Lock days", 0, 4, rp.LOCK_DAYS, key="rp_lock", help=ui.GLOSSARY["Lock days"])
            r_move = q3.slider("Price per move", 0, 300, rp.MOVE_PENALTY, 10, key="rp_move", help=ui.GLOSSARY["Price per move"])
            go = st.form_submit_button("Compare strategies", type="primary", key="rp_btn")

    if go:
        r_kind = r_kind or "Emergency"
        emergency = r_kind == "Emergency"
        srow = sections[sections.section_id == r_sec].iloc[0]
        emg = wi.make_emergency_task(
            T, r_dept, r_sec, float(srow.km_start) + 5.0, r_dur, task_id="EMG-UI" if emergency else "NEW-UI",
            overdue_days=1 - now_day if emergency else -(now_day + 15),
            priority_score=0.95 if emergency else 0.45, tier="CRITICAL" if emergency else "MEDIUM",
            severity=5 if emergency else 3)
        event = [dict(day=now_day, kind="emergency" if emergency else "routine", tasks=emg)]
        rows, plans = [], {}
        with st.spinner("Re-planning with three strategies..."):
            for strat in ("stable", "full", "patch"):
                kw = dict(lock_days=r_lock, move_penalty=r_move) if strat == "stable" else {}
                ev, sm, p, _, _ = rp.simulate(T, windows, plan, event, strat, workers=8, time_limit=15, **kw)
                e = ev.iloc[0]
                rows.append(dict(strategy=ui.STRATEGIES[strat], moved=e.moved, dropped=e.dropped,
                                 blocks_changed=e.blocks_changed, broken_commitments=e.broken_commitments,
                                 new_blocks=e.blocks_opened, trains_change=e.trains_change,
                                 emergency_response_days=e.emergency_response_days, net_value=e.net_value,
                                 valid=(e.violations == 0 and sm["final_violations"] == 0)))
                plans[strat] = p
        st.session_state.rp_result = (pd.DataFrame(rows), plans["stable"], now_day)

    if "rp_result" in st.session_state:
        out, view, day = st.session_state.rp_result
        st.dataframe(out, hide_index=True, width="stretch", column_config={
            "strategy": "Strategy", "moved": "Tasks moved", "dropped": "Tasks dropped",
            "blocks_changed": "Blocks changed", "broken_commitments": "Broken commitments",
            "new_blocks": "New blocks", "trains_change": "Change in trains", "emergency_response_days":
            CC.NumberColumn("Response (days)", format="%.1f"), "net_value": CC.NumberColumn("Net value", format="%d"),
            "valid": CC.CheckboxColumn("All rules met")})
        st.caption("Broken commitments are counted against a 2-day lock.")
        v = view[(view.date >= o.START + pd.Timedelta(days=day))
                 & (view.date < o.START + pd.Timedelta(days=day + rp.LOOKAHEAD_DAYS))]
        v = v[v.section_id.isin(v.section_id.drop_duplicates().head(6))]
        gantt(v, f"Stable plan, days {day}-{day + rp.LOOKAHEAD_DAYS - 1} (first 6 sections)")


def page_unscheduled():
    ui.header("🧩 Unscheduled work", "Tasks left out of the plan, why, and what to do about each one.")
    ui.explain("unscheduled")
    u = res["unscheduled"].copy()
    u["Why"] = u.reason.map(ui.reason_label)
    never = (u.group == "NEVER_PLACEABLE").sum()
    declined = (u.group == "PLACEABLE_NOT_SCHEDULED").sum()
    tl = u[u.reason == "TOO_LONG"]
    k = st.columns(4)
    k[0].metric("Unscheduled tasks", f"{len(u)} / {len(T)}", f"{len(u) / len(T):.0%} of all tasks", delta_color="off", delta_arrow="off")
    k[1].metric("Cannot fit any window", never, help="Needs a longer or different window from the control office.")
    k[2].metric("Could fit, declined", declined, help="Low value now; carried to the next planning cycle.")
    k[3].metric("Critical / high left out", int(u.tier.isin(["CRITICAL", "HIGH"]).sum()),
                delta_color="off", delta_arrow="off", help="These deserve a manual review first.")

    if len(tl):
        short = tl.recommended_action.str.extract(r"\+(\d+)")[0].astype(int)
        st.warning(f"**Action:** {len(tl)} tasks need an extended block ({short.min()}-{short.max()} min longer than "
                   "any window). Ask the control office for longer windows on these sections.", icon="📣")

    a, b = st.columns([1.2, 1])
    with a:
        g = u.groupby(["Why", "tier"]).size().rename("tasks").reset_index()
        fig = px.bar(g, y="Why", x="tasks", color="tier", orientation="h", color_discrete_map=ui.TIER_COLORS,
                     category_orders={"tier": ui.TIER_ORDER}, title="Why tasks were left out", text_auto=True)
        fig.update_layout(yaxis_title=None, xaxis_title="Tasks", height=300)
        st.plotly_chart(tidy(fig), width="stretch")
    with b:
        st.markdown("**What the reasons mean**")
        for code in u.reason.unique():
            name, desc = ui.REASONS.get(code, (code, ""))
            st.markdown(f"- **{name}** - {desc}")

    with st.expander(f"📋 Browse all {len(u)} unscheduled tasks, with filters"):
        f1, f2, f3 = st.columns(3)
        why = f1.multiselect("Reason", sorted(u.Why.unique()), key="un_reason", placeholder="All reasons")
        dept = f2.multiselect("Department", sorted(u.dept.unique()), key="un_dept", placeholder="All departments")
        tier = f3.multiselect("Priority", [t for t in ui.TIER_ORDER if t in set(u.tier)], key="un_tier",
                              placeholder="All priorities")
        v = u[u.Why.isin(why or u.Why.unique()) & u.dept.isin(dept or u.dept.unique()) & u.tier.isin(tier or u.tier.unique())]
        st.dataframe(v[["task_id", "dept", "section_id", "task", "tier", "Why", "due_idx", "recommended_action"]],
                     hide_index=True, width="stretch", column_config={
                         "task_id": "Task ID", "dept": "Dept", "section_id": "Section", "task": "Work", "tier": "Priority",
                         "Why": "Why not scheduled", "due_idx": CC.NumberColumn("Due day", help="Days from plan start; "
                                                                                 "negative = already overdue"),
                         "recommended_action": CC.TextColumn("Recommended action", width="large")})
        st.download_button("Download unscheduled tasks (CSV)", v.to_csv(index=False), "unscheduled_tasks.csv", "text/csv",
                           key="un_dl")
    if res["audit"] is not None:
        with st.expander("🔍 Objective audit (consistency check)"):
            st.caption("The plan's value recomputed independently, compared with what the solver reported.")
            st.dataframe(res["audit"], hide_index=True, width="stretch")


def page_export():
    ui.header("📤 Export block requests", "Download the recommended blocks as requests for review. The layout is "
                                          "illustrative until the real BDMS fields are available.")
    ui.explain("export")
    ew = st.segmented_control("Week", ["All"] + WEEKS, default="All", key="ex_week",
                              format_func=lambda w: "All weeks" if w == "All" else f"Week {w}") or "All"
    wb, wp = (blocks, plan) if ew == "All" else (blocks[blocks.week == ew], plan[plan.week == ew])
    req = bx.to_bdms(wp, wb)
    k = st.columns(4)
    k[0].metric("Block requests", len(req))
    k[1].metric("Joint blocks", int(req.departments.str.contains(r"\+").sum()))
    k[2].metric("Power blocks", int(req.power_block_required.sum()))
    k[3].metric("Work items", int(req.n_work_items.sum()))
    st.dataframe(req, hide_index=True, width="stretch", height=420, column_config={
        "block_id": "Block ID", "section": "Section", "date": "Date", "start": "Start", "end": "End",
        "duration_min": "Minutes", "departments": "Departments", "crews": "Crews",
        "power_block_required": CC.CheckboxColumn("Power off"), "n_work_items": "Items",
        "work_items": CC.TextColumn("Work items", width="large"), "trains_affected": "Trains",
        "goods_trains_affected": "Goods trains", "remarks": "Remarks", "status": "Status"})
    with st.container(border=True):
        st.markdown("**Download**")
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Block requests (CSV)", req.to_csv(index=False), "block_requests.csv", "text/csv",
                           key="dl_csv", width="stretch", type="primary")
        d2.download_button("⬇️ Block requests (JSON)", bx.to_json(wp, wb), "block_requests.json", "application/json",
                           key="dl_json", width="stretch")
        d3.download_button("⬇️ Work items (CSV)", bx.work_items(wp).to_csv(index=False), "work_items.csv", "text/csv",
                           key="dl_items", width="stretch")
    st.warning("Recommendations only - every block request needs review and approval by authorised officials.",
               icon="⚠️")


# ================================================================ navigation + sidebar
PAGES = {
    "start": st.Page(page_start, title="Start here", icon="👋", default=True),
    "overview": st.Page(page_overview, title="Overview", icon="🏠", url_path="overview"),
    "calendar": st.Page(page_calendar, title="Block calendar", icon="📅", url_path="calendar"),
    "monthly": st.Page(page_monthly, title="Weekly & monthly view", icon="🗓️", url_path="monthly"),
    "what-if": st.Page(page_whatif, title="Emergency what-if", icon="⚡", url_path="what-if"),
    "rolling": st.Page(page_rolling, title="Rolling re-plan", icon="🔄", url_path="rolling"),
    "unscheduled": st.Page(page_unscheduled, title="Unscheduled work", icon="🧩", url_path="unscheduled"),
    "export": st.Page(page_export, title="Export requests", icon="📤", url_path="export"),
}
pages = {"Start": [PAGES["start"]],
         "Plan": [PAGES[k] for k in ("overview", "calendar", "monthly")],
         "Scenarios": [PAGES[k] for k in ("what-if", "rolling")],
         "Review": [PAGES[k] for k in ("unscheduled", "export")]}
nav = st.navigation(pages)

st.logo(str(ROOT / "assets" / "logo.svg"), size="large")
with st.sidebar:
    st.toggle("🎓 Explain mode", value=True, key="explain",
              help="Adds a short 'what you see / what to look at / why it matters' box to every page.")
    st.markdown("**Current plan**")
    st.markdown(f"**{o.START:%d %b} - {END:%d %b %Y}**  \n{len(T)} tasks · {len(windows)} corridor windows  \n"
                f"{blocks.shape[0]} blocks planned")
    with st.expander("📖 Glossary"):
        for term, text in ui.GLOSSARY.items():
            st.markdown(f"**{term}** - {text}")
    with st.expander("⚙️ Re-run the planner (~1 min)"):
        bal = st.number_input("Balance strength", 0, 50, h.DEFAULT_BALANCE, key="rerun_balance",
                              help="Penalty for overloading one week. Higher = work spread more evenly across weeks.")
        res_pct = st.slider("Emergency reserve", 0.0, 0.5, h.EMERGENCY_RESERVE, 0.05, key="rerun_reserve",
                            help="Share of window time the monthly plan keeps free for emergencies.")
        early = st.number_input("Earliest start (days before due)", 0, 60, h.EARLY_DAYS, key="rerun_early",
                                help="Work may start at most this many days before its due date.")
        if st.button("Run planner", key="rerun_btn", width="stretch"):
            with st.spinner("Planning 4 weeks..."):
                r = subprocess.run([sys.executable, "horizons.py", "--no-ablation", "--balance", str(bal),
                                    "--reserve", str(res_pct), "--early-days", str(early)],
                                   cwd=ROOT, capture_output=True, text=True, timeout=900)
            st.cache_data.clear()
            if r.returncode == 0:
                st.rerun()
            else:
                st.error(r.stderr[-800:])
        st.caption("What-if and the manual baseline always use the default 14-day rule.")

nav.run()
ui.footer()
