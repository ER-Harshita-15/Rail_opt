"""
Shared look-and-feel for the RailOpt dashboard: colours, CSS, plain-language labels and small
layout helpers. Nothing here changes planning logic - it only affects how results are shown.
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------- colours
PRIMARY = "#0B3C5D"      # deep railway blue
ACCENT = "#F2A541"       # signal amber
MUTED = "#B0B7BF"        # manual / baseline grey
GOOD, BAD, NEUTRAL = "#2E7D32", "#C62828", "#6B7280"

DEPT_COLORS = {"ENGG": "#2A6F97", "S&T": "#E07A1F", "TRD": "#6A994E"}
DEPT_NAMES = {"ENGG": "Engineering (track)", "S&T": "Signal & Telecom", "TRD": "Traction (OHE)"}
TIER_COLORS = {"CRITICAL": "#B71C1C", "HIGH": "#EF6C00", "MEDIUM": "#F9A825", "LOW": "#78909C"}
TIER_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
PLAN_COLORS = [PRIMARY, MUTED]

# ---------------------------------------------------------------- plain-language labels
REASONS = {
    "TOO_LONG": ("Longer than every window", "The work needs more time than any corridor window offers."),
    "NO_POWER_WINDOW": ("No power-block window", "Needs overhead power switched off, but no window allows it."),
    "NO_WINDOW_IN_HORIZON": ("No window on this section", "The control office offered no window on this section."),
    "DECLINED_NET_NEGATIVE": ("Not worth a block yet", "Opening a block would cost more (trains, time) than the work is worth now."),
    "INDIFFERENT_TIE": ("Optional - no net gain", "Scheduling it neither helps nor hurts the plan."),
    "WEEKLY_POOL_EFFECT": ("Left out by weekly pool", "Would add value, but was not in the week's candidate pool."),
    "NOT_YET_ELIGIBLE": ("Too early to start", "Its earliest allowed start date is after this plan."),
    "NO_ELIGIBLE_WINDOW": ("No usable window", "No window is left after its earliest start date."),
    "NOT_EXPLAINED": ("Not analysed", "The planner ran without the explanation step."),
}
GROUPS = {"NEVER_PLACEABLE": "Cannot fit any window", "PLACEABLE_NOT_SCHEDULED": "Could fit, planner declined"}
REALIZATION = {"ON_TIME": "Done in the planned week", "SLIPPED": "Moved to a later week",
               "PULLED_FORWARD": "Done earlier than planned", "NOT_SCHEDULED": "Not scheduled"}
VERDICTS = {"better than all orders": "🟢 Better than every manual order",
            "worse than all orders": "🔴 Worse than every manual order",
            "within the manual range": "🟡 Within the manual range",
            "equal to all orders": "⚪ Same as manual", "info": "ℹ️ For information"}
STRATEGIES = {"stable": "Stable re-plan (RailOpt)", "full": "Full re-optimisation", "patch": "Manual-style patch"}

GLOSSARY = {
    "Block": "A time slot when a section of track is closed to trains so crews can work safely.",
    "Corridor window (COA)": "A slot the Control Office says can be given as a block. Blocks can only be opened inside these.",
    "Joint block": "One block shared by two or more departments, so trains are disrupted once instead of several times.",
    "Section": "A stretch of line between two stations, e.g. BSP-BLSP-J.",
    "Priority tier": "CRITICAL > HIGH > MEDIUM > LOW, from the priority score (criticality, urgency, failure risk, traffic impact).",
    "Net value": "The planner's score: value of work done (higher for urgent / critical tasks, less if delayed) minus the cost of blocks (trains disrupted, block time). Higher is better.",
    "Manual baseline": "A simulation of today's process: each department books its own blocks, never shared. Run in all 6 submission orders.",
    "Trains disrupted": "Sum of trains running through the section during each opened block.",
    "Task-days past due": "For overdue work, how many days late it starts, added up. A lateness proxy, not measured downtime.",
    "Lock days": "During rolling re-planning, blocks in the next N days are frozen because crews and approvals are already arranged.",
    "Price per move": "Penalty the re-planner pays for every task it moves to a different block. Higher = more stable plan.",
}


# Explain mode: for each page, what is on screen, what to look at, and why it matters.
EXPLAIN = {
    "overview": ("The same 4 weeks of maintenance planned two ways: by RailOpt, and by today's manual process "
                 "(each department books its own blocks).",
                 "The two columns. Fewer blocks and fewer trains disrupted means less disruption for the same work.",
                 "This is the one-slide answer to the problem statement: coordinated planning frees up the line."),
    "calendar": ("A timeline of blocks. Each row is one section and department; each bar is one task.",
                 "Bars of different colours at the same time on the same row group are one shared block. "
                 "Switch Plan to Manual to see how many more separate blocks the manual process needs.",
                 "Shared blocks mean trains are stopped once instead of two or three times."),
    "monthly": ("The month split into weeks: how many tasks and blocks each week gets, per department and section.",
                "Week 1 is busiest because overdue work is cleared first; later weeks hold work that is not yet due.",
                "Shows the long-term (monthly) and short-term (weekly) horizons the problem statement asks for."),
    "whatif": ("A test: a new urgent defect arrives. The planner tries it in each week and reports the cost.",
               "The recommended week and how many other tasks had to move.",
               "Shows the plan can absorb emergencies, and what each choice costs the rest of the plan."),
    "rolling": ("What happens when defects keep arriving after the plan is published. Three ways to respond "
                "are compared on the same events.",
                "Tasks moved and broken commitments: a good re-plan changes little that crews were already told.",
                "Real plans change daily; a stable re-plan keeps crews and the control office able to trust it."),
    "unscheduled": ("Tasks the plan left out, each with a reason and a recommended action.",
                    "The orange action box: some work is longer than every available window.",
                    "Nothing is dropped silently. Every left-out task has an explanation an official can act on."),
    "export": ("The recommended blocks as a list of block requests, ready to review.",
               "Joint blocks (several departments) and blocks that need power switched off.",
               "This is what would go into BDMS. Officials still review and approve every request."),
}

# ---------------------------------------------------------------- CSS + plotly theme
CSS = f"""
<style>
.block-container {{padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
[data-testid="stSidebar"] {{background: linear-gradient(180deg, #0B3C5D 0%, #12476b 100%);}}
[data-testid="stSidebar"] * {{color: #E8EEF3;}}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span {{color: #E8EEF3 !important;}}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{background: rgba(242,165,65,.22);}}
[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] {{border-color: rgba(255,255,255,.15);}}
[data-testid="stSidebar"] header span, [data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {{color: #F2A541 !important;}}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * {{color: #0B3C5D !important;}}
[data-testid="stSidebar"] [data-testid="stExpander"] details {{border-color: rgba(255,255,255,.2); background: rgba(255,255,255,.04);}}
[data-testid="stSidebar"] .stButton button {{background: #F2A541; border: none;}}
[data-testid="stSidebar"] .stButton button p {{color: #0B3C5D !important; font-weight: 600;}}
[data-testid="stMetric"] {{background: #FFFFFF; border: 1px solid #E3E8EE; border-radius: 12px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04);}}
[data-testid="stMetricLabel"] p {{font-size: .82rem; color: #5B6B7B; font-weight: 600; text-transform: uppercase; letter-spacing: .03em;}}
[data-testid="stMetricValue"] {{font-size: 1.75rem; color: {PRIMARY};}}
.ro-hero {{background: linear-gradient(120deg, #0B3C5D 0%, #1C5D87 60%, #2A6F97 100%); color: #fff;
    border-radius: 16px; padding: 22px 26px; margin-bottom: 18px;}}
.ro-hero h1 {{color: #fff; font-size: 1.85rem; margin: 0 0 4px 0; padding: 0;}}
.ro-hero p {{color: #D7E3EE; margin: 0; font-size: 1rem;}}
.ro-badge {{display: inline-block; background: {ACCENT}; color: #0B3C5D; font-weight: 700; font-size: .72rem;
    padding: 3px 9px; border-radius: 999px; margin-bottom: 8px; letter-spacing: .04em;}}
.ro-steps {{display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 18px 0;}}
.ro-step {{flex: 1 1 180px; background: #F5F8FB; border: 1px solid #E3E8EE; border-radius: 12px; padding: 12px 14px;}}
.ro-step b {{color: {PRIMARY};}} .ro-step span {{display: block; color: #5B6B7B; font-size: .86rem; margin-top: 2px;}}
.ro-step .n {{display: inline-block; width: 22px; height: 22px; border-radius: 50%; background: {PRIMARY}; color: #fff;
    text-align: center; font-size: .78rem; line-height: 22px; margin-right: 6px;}}
.ro-take {{border-left: 4px solid {ACCENT}; background: #FFF9F0; padding: 12px 16px; border-radius: 8px; margin: 6px 0 14px 0;}}
.ro-take ul {{padding-left: 20px;}} .ro-take li {{margin: 4px 0;}}
.ro-card {{background: #fff; border: 1px solid #E3E8EE; border-radius: 12px; padding: 14px 16px; height: 100%;}}
.ro-card h4 {{margin: 0 0 6px 0; color: {PRIMARY}; font-size: 1rem;}}
.ro-card p {{margin: 0; color: #4B5B6B; font-size: .9rem;}}
.ro-chip {{display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
    margin-right: 6px; color: #fff;}}
.ro-section {{font-size: 1.12rem; font-weight: 700; color: {PRIMARY}; margin: 18px 0 6px 0;}}
.ro-sub {{color: #5B6B7B; margin: -2px 0 10px 0; font-size: .92rem;}}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label > div:first-of-type {{background: rgba(255,255,255,.25) !important;}}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type {{background: {ACCENT} !important;}}
.ro-explain {{display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 2px 0 18px 0;}}
.ro-explain div {{background: #EEF5FB; border: 1px solid #D6E6F3; border-radius: 10px; padding: 10px 14px;
    font-size: .9rem; color: #33475B;}}
.ro-explain b {{display: block; color: {PRIMARY}; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em;
    margin-bottom: 3px;}}
@media (max-width: 900px) {{[data-testid="stSidebar"] [data-testid="stCheckbox"] label > div:first-of-type {{background: rgba(255,255,255,.25) !important;}}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type {{background: {ACCENT} !important;}}
.ro-explain {{grid-template-columns: 1fr;}}}}
.ro-vs {{display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 6px 0 16px 0;}}
.ro-vs > div {{border-radius: 14px; padding: 16px 20px; border: 1px solid #E3E8EE; background: #fff;}}
.ro-vs .us {{border: 2px solid {PRIMARY}; box-shadow: 0 2px 10px rgba(11,60,93,.08);}}
.ro-vs h4 {{margin: 0 0 10px 0; font-size: 1rem; color: #5B6B7B;}} .ro-vs .us h4 {{color: {PRIMARY};}}
.ro-vs table {{width: 100%; border-collapse: collapse; border: none !important; margin: 0;}}
.ro-vs tr {{background: transparent !important; border: none !important;}}
.ro-vs td {{padding: 9px 0 !important; border: none !important; border-bottom: 1px solid #F0F3F6 !important;
    font-size: .95rem; color: #33475B; height: 44px; vertical-align: middle;}}
.ro-vs tr:last-child td {{border-bottom: none;}}
.ro-vs td.v {{text-align: right; font-weight: 700; font-size: 1.15rem; color: #1F2D3A;}}
.ro-vs .us td.v {{color: {PRIMARY};}}
.ro-vs .d {{display: inline-block; margin-left: 8px; padding: 1px 7px; border-radius: 999px; font-size: .72rem;
    font-weight: 700; color: {GOOD}; background: #E8F5E9; vertical-align: middle;}}
.ro-vs .d.bad {{color: #8A5A00; background: #FFF3DC;}}
.ro-tour {{background: #fff; border: 1px solid #E3E8EE; border-radius: 12px; padding: 12px 16px; margin-bottom: 4px;}}
.ro-tour p {{margin: 2px 0; font-size: .92rem; color: #33475B;}}
.ro-tour .say {{color: #5B6B7B; font-style: italic;}}
.ro-foot {{color: #8592A0; font-size: .8rem; margin-top: 28px; border-top: 1px solid #E3E8EE; padding-top: 10px;}}
</style>
"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=13, color="#1F2D3A"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[PRIMARY, ACCENT, "#6A994E", "#8E5572", MUTED],
        title=dict(font=dict(size=15, color=PRIMARY), x=0, xanchor="left"),
        xaxis=dict(gridcolor="#EDF1F5", zerolinecolor="#DDE3EA", linecolor="#DDE3EA"),
        yaxis=dict(gridcolor="#EDF1F5", zerolinecolor="#DDE3EA", linecolor="#DDE3EA"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        margin=dict(l=10, r=10, t=60, b=10), hoverlabel=dict(bgcolor="#fff", font_size=12))
    pio.templates["railopt"] = tpl
    pio.templates.default = "plotly_white+railopt"


# ---------------------------------------------------------------- layout helpers
def hero(title, subtitle, badge="PROTOTYPE · SYNTHETIC DATA"):
    st.markdown(f'<div class="ro-hero"><span class="ro-badge">{badge}</span><h1>{title}</h1>'
                f'<p>{subtitle}</p></div>', unsafe_allow_html=True)


def header(title, subtitle):
    st.markdown(f"## {title}")
    st.markdown(f'<p class="ro-sub">{subtitle}</p>', unsafe_allow_html=True)


def section(title, sub=None):
    st.markdown(f'<div class="ro-section">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<p class="ro-sub">{sub}</p>', unsafe_allow_html=True)


def steps(items):
    html = "".join(f'<div class="ro-step"><span class="n">{i}</span><b>{t}</b><span>{d}</span></div>'
                   for i, (t, d) in enumerate(items, 1))
    st.markdown(f'<div class="ro-steps">{html}</div>', unsafe_allow_html=True)


def takeaways(lines, title="Key takeaways"):
    li = "".join(f"<li>{x}</li>" for x in lines)
    st.markdown(f'<div class="ro-take"><b>{title}</b><ul style="margin:6px 0 0 0">{li}</ul></div>',
                unsafe_allow_html=True)


def card(title, body, color=None):
    bar = f"border-top: 4px solid {color};" if color else ""
    st.markdown(f'<div class="ro-card" style="{bar}"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def dept_legend():
    chips = "".join(f'<span class="ro-chip" style="background:{c}">{k} · {DEPT_NAMES[k]}</span>'
                    for k, c in DEPT_COLORS.items())
    st.markdown(chips, unsafe_allow_html=True)


def explain(page):
    """Three short boxes at the top of a page; hidden when Explain mode is switched off in the sidebar."""
    if not st.session_state.get("explain", True) or page not in EXPLAIN:
        return
    what, look, why = EXPLAIN[page]
    st.markdown(f'<div class="ro-explain"><div><b>👀 What you see</b>{what}</div>'
                f'<div><b>🔍 What to look at</b>{look}</div><div><b>💡 Why it matters</b>{why}</div></div>',
                unsafe_allow_html=True)


def versus(left_title, right_title, rows):
    """Side-by-side 'manual vs RailOpt' card. rows = [(label, manual_value, railopt_value, delta_text, good)]."""
    def side(title, cls, idx):
        trs = ""
        for label, a, b, d, good in rows:
            val = a if idx == 0 else b
            extra = f'<span class="d{"" if good else " bad"}">{d}</span>' if idx == 1 and d else ""
            trs += f'<tr><td>{label}</td><td class="v">{val}{extra}</td></tr>'
        return f'<div class="{cls}"><h4>{title}</h4><table>{trs}</table></div>'
    st.markdown(f'<div class="ro-vs">{side(left_title, "them", 0)}{side(right_title, "us", 1)}</div>',
                unsafe_allow_html=True)


def tour_step(n, title, show, say):
    st.markdown(f'<div class="ro-tour"><p><b>{n}. {title}</b></p><p>👉 {show}</p>'
                f'<p class="say">🗣️ "{say}"</p></div>', unsafe_allow_html=True)


def how_to_read(md):
    with st.expander("ℹ️ How to read this page"):
        st.markdown(md)


def footer():
    st.markdown('<div class="ro-foot">RailOpt is a research prototype on synthetic data. Every schedule is a '
                'recommendation - authorised railway officials review and approve all block requests.</div>',
                unsafe_allow_html=True)


def hhmm(x):
    x = int(x)
    return f"{x // 60:02d}:{x % 60:02d}"


def reason_label(code):
    return REASONS.get(code, (code.replace("_", " ").capitalize(), ""))[0]
