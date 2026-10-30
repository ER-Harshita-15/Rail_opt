
"""
Step 5 - shared KPIs and result loading.

The SAME function scores the optimised plan and the simulated manual baseline,
so the two are compared on identical definitions
(net value uses the Step 3/4 integer objective).

KPI notes
    net_value            sum of assignment values - block costs
                         (optimizer.net_value definition)
    trains_disrupted     sum of trains_affected over opened blocks
                         (COA window data, synthetic)
    days_past_due_total  sum over placeable tasks of
                         max(0, execution day - due day); a task that is
                         never scheduled counts until the end of the horizon.
                         A PROXY for how long assets stay in a degraded state
                         - NOT measured downtime or delay minutes.
    productive_pct       crew work minutes / (work + setup) minutes
                         in opened blocks
    block_occupancy_pct  mean over blocks of
                         (busiest crew minutes / window length)
"""

from pathlib import Path

import pandas as pd

import horizons as h
import optimizer as o


ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
START = o.START


KPI_LABELS = {
    "tasks_scheduled": "Tasks scheduled",
    "blocks": "Blocks opened",
    "merged_blocks": "Multi-department blocks",
    "block_minutes": "Block minutes taken",
    "trains_disrupted": "Trains disrupted (sum over blocks)",
    "goods_trains_disrupted": "  of which goods trains",
    "net_value": "Net value (objective)",
    "tasks_per_block": "Tasks per block",
    "productive_pct": "Productive crew time (%)",
    "block_occupancy_pct": "Block occupancy, busiest crew (%)",
    "overdue_scheduled": "Overdue tasks scheduled",
    "overdue_mean_day": "Overdue tasks: mean start day",
    "crit_high_scheduled": "Critical/high tasks scheduled",
    "crit_high_mean_day": "Critical/high: mean start day",
    "days_past_due_total": "Task-days past due (proxy)",
    "newly_late": "Not-yet-overdue tasks finished late",
    "week1_share_pct": "Share of tasks in week 1 (%)",
    "peak_to_mean_work": "Peak / mean weekly workload",
}


# For each KPI:
# +1 = higher is better
# -1 = lower is better
#  0 = neutral / informational

KPI_DIRECTION = {
    "tasks_scheduled": 1,
    "blocks": -1,
    "merged_blocks": 0,
    "block_minutes": -1,
    "trains_disrupted": -1,
    "goods_trains_disrupted": -1,
    "net_value": 1,
    "tasks_per_block": 1,
    "productive_pct": 1,
    "block_occupancy_pct": 0,
    "overdue_scheduled": 1,
    "crit_high_scheduled": 1,
    "overdue_mean_day": -1,
    "crit_high_mean_day": -1,
    "days_past_due_total": -1,
    "newly_late": -1,
    "week1_share_pct": -1,
    "peak_to_mean_work": -1,
}


def with_week(df, start=START):
    """Add a week number to a plan or block dataframe."""
    if df is None or len(df) == 0 or "week" in df.columns:
        return df

    return df.assign(
        week=(pd.to_datetime(df.date) - start).dt.days // 7 + 1
    )


def kpis(plan, blocks, T, windows, start=START, horizon_days=28):
    """
    Score one plan.

    `plan` and `blocks` use the Step 4 format produced by
    finish_plan_parallel().

    This function also safely handles empty plans.
    """

    # ---------------------------------------------------------
    # 1. Identify tasks that can never be scheduled.
    # ---------------------------------------------------------

    never = set(
        o.find_unschedulable(
            T,
            o.scope(T, windows, horizon_days, start)[0]
        ).task_id
    )

    P = T[~T.task_id.isin(never)].copy()

    # ---------------------------------------------------------
    # 2. Determine whether a valid plan exists.
    # ---------------------------------------------------------

    has = plan is not None and len(plan) > 0

    if has:
        plan = with_week(plan, start)
        blocks = with_week(blocks, start)

        first = (
            plan.groupby("task_id")["date"]
            .min()
            .sub(start)
            .dt.days
        )
    else:
        # An empty numeric Series prevents string dtype
        # propagation when no tasks have been scheduled.
        first = pd.Series(dtype="float64")

    # ---------------------------------------------------------
    # 3. Calculate execution days safely.
    # ---------------------------------------------------------

    P["exec_day"] = pd.to_numeric(
        P.task_id.map(first),
        errors="coerce"
    ).astype("float64")

    # Explicit numeric conversion is important for empty plans
    # and pandas Arrow-backed string columns.
    overdue_days = pd.to_numeric(
        P["overdue_days"],
        errors="coerce"
    ).astype("float64")

    # Due day is expressed relative to the planning start date.
    due = -overdue_days

    # ---------------------------------------------------------
    # 4. Calculate total task-days past due.
    # ---------------------------------------------------------

    # Scheduled tasks use their actual execution day.
    # Unscheduled tasks count until the end of the horizon.
    days_past_due = int(
        (
            P["exec_day"].fillna(float(horizon_days)) - due
        )
        .clip(lower=0)
        .sum()
    )

    # ---------------------------------------------------------
    # 5. Calculate objective value.
    # ---------------------------------------------------------

    if has:
        gain, cost = o.gain_and_cost(
            plan,
            blocks,
            start
        )
    else:
        gain, cost = 0, 0

    # ---------------------------------------------------------
    # 6. Group tasks by overdue and priority status.
    # ---------------------------------------------------------

    ov = P[overdue_days > 0]

    ch = P[
        P.tier.isin(["CRITICAL", "HIGH"])
    ]

    nv = P[
        (overdue_days <= 0)
        & P.exec_day.notna()
    ]

    # Mean execution day for a selected task group.
    def mean(df):
        if not df.exec_day.notna().any():
            return None

        return round(
            float(df.exec_day.mean()),
            2
        )

    # ---------------------------------------------------------
    # 7. Calculate block and task counts.
    # ---------------------------------------------------------

    n_blocks = len(blocks) if has else 0

    n_sched = (
        int(plan.task_id.nunique())
        if has else 0
    )

    # ---------------------------------------------------------
    # 8. Calculate workload distribution.
    # ---------------------------------------------------------

    if has:
        wp, prof = h.workload_profile(
            plan,
            blocks,
            n_weeks=max(1, horizon_days // 7)
        )
    else:
        prof = {
            "week1_share_pct": 0.0,
            "peak_to_mean_work": 0.0
        }

    # ---------------------------------------------------------
    # 9. Calculate newly late tasks.
    # ---------------------------------------------------------

    newly_late = int(
        (
            nv.exec_day
            - (-overdue_days.loc[nv.index])
            > 0
        ).sum()
    )

    # ---------------------------------------------------------
    # 10. Return all KPIs.
    # ---------------------------------------------------------

    return dict(
        tasks_scheduled=n_sched,
        placeable_tasks=len(P),

        blocks=n_blocks,

        merged_blocks=(
            int(blocks.merged.sum())
            if has else 0
        ),

        block_minutes=(
            int(blocks.length_min.sum())
            if has else 0
        ),

        trains_disrupted=(
            int(blocks.trains_affected.sum())
            if has else 0
        ),

        goods_trains_disrupted=(
            int(blocks.goods_trains_affected.sum())
            if has else 0
        ),

        net_value=int(gain - cost),

        tasks_per_block=(
            round(n_sched / n_blocks, 2)
            if n_blocks else 0.0
        ),

        productive_pct=(
            round(
                100
                * float(blocks.work_min.sum())
                / float(blocks.crew_total_min.sum()),
                1
            )
            if has else 0.0
        ),

        block_occupancy_pct=(
            round(
                100 * float(blocks.utilization.mean()),
                1
            )
            if has else 0.0
        ),

        overdue_cleared=(
            f"{int(ov.exec_day.notna().sum())}/{len(ov)}"
        ),

        overdue_mean_day=mean(ov),

        overdue_scheduled=int(
            ov.exec_day.notna().sum()
        ),

        overdue_total=len(ov),

        crit_high_cleared=(
            f"{int(ch.exec_day.notna().sum())}/{len(ch)}"
        ),

        crit_high_mean_day=mean(ch),

        crit_high_scheduled=int(
            ch.exec_day.notna().sum()
        ),

        crit_high_total=len(ch),

        days_past_due_total=days_past_due,

        newly_late=newly_late,

        week1_share_pct=float(
            prof["week1_share_pct"]
        ),

        peak_to_mean_work=float(
            prof["peak_to_mean_work"]
        )
    )


def load_results(out=OUT):
    """Load saved outputs generated by horizons.py."""

    out = Path(out)

    def rd(name, **kw):
        p = out / name

        if p.exists():
            return pd.read_csv(p, **kw)

        return None

    return dict(
        plan=rd(
            "block_plan_4w.csv",
            parse_dates=["date"]
        ),

        blocks=rd(
            "block_summary_4w.csv",
            parse_dates=["date"]
        ),

        summary=rd("horizon_summary.csv"),

        monthly=rd("monthly_plan.csv"),

        realization=rd("monthly_realization.csv"),

        unscheduled=rd("unscheduled_final.csv"),

        capacity=rd("capacity_diagnosis.csv"),

        audit=rd("objective_audit.csv"),

        workload=rd("workload_profile.csv"),

        ablation=rd("merge_ablation_week1.csv")
    )


def load_inputs(early_days=h.EARLY_DAYS):
    """Load and prepare the task and corridor-window input data."""

    raw, windows = o.load()

    return h.prepare_tasks(raw, early_days), windows
