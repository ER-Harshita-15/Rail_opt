"""
Step 5 - BDMS-style export of a block plan.

The real BDMS format is not available, so this is an ILLUSTRATIVE layout: one row per block request
(section, date, time, departments, crews, work items, whether a power block is needed, traffic
affected) plus a work-item table. Replace the column names with the real BDMS fields when known.
The system only RECOMMENDS: an official must review and approve every request.
"""
import json

import pandas as pd


def _hhmm(minutes):
    minutes = int(minutes)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def work_items(plan):
    p = plan.sort_values(["date", "window_id", "gang_type", "start_min"]).copy()
    p["block_id"] = ("BLK-" + p.date.dt.strftime("%Y%m%d") + "-" + p.section_id.str.replace("[^A-Za-z0-9]+", "", regex=True)
                     + "-" + p.window_id)
    return pd.DataFrame(dict(
        block_id=p.block_id, task_id=p.task_id, department=p.dept, crew=p.gang_type, work=p.task,
        km=p.km, start=p.start_min.map(_hhmm), end=p.end_min.map(_hhmm), priority_tier=p.tier,
        overdue_days_at_plan_start=p.overdue_days)).reset_index(drop=True)


def to_bdms(plan, blocks):
    """One row per block request."""
    items = work_items(plan)
    ids = plan.sort_values(["date", "window_id", "gang_type", "start_min"]).copy()
    ids["block_id"] = items.block_id.values
    power = ids.groupby("window_id").block_type.apply(lambda s: bool((s == "POWER").any()))
    desc = ids.groupby("window_id").apply(
        lambda d: "; ".join(f"{r.task_id} {r.task} km {r.km:g} [{r.dept}]" for r in d.itertuples()),
        include_groups=False)
    bid = ids.groupby("window_id").block_id.first()
    crews = ids.groupby("window_id").gang_type.apply(lambda s: ", ".join(sorted(set(s))))
    b = blocks.sort_values(["date", "window_id"]).copy()
    out = pd.DataFrame(dict(
        block_id=b.window_id.map(bid), section=b.section_id, date=b.date.dt.strftime("%Y-%m-%d"),
        start=b.win_start.map(_hhmm), end=b.win_end.map(_hhmm), duration_min=b.length_min,
        departments=b.depts, crews=b.window_id.map(crews), power_block_required=b.window_id.map(power),
        n_work_items=b.n_tasks, work_items=b.window_id.map(desc),
        trains_affected=b.trains_affected, goods_trains_affected=b.goods_trains_affected,
        remarks=b.merged.map({True: "joint block - departments work in parallel", False: "single-department block"}),
        status="RECOMMENDED - awaiting approval"))
    return out.reset_index(drop=True)


def to_json(plan, blocks):
    req = to_bdms(plan, blocks)
    items = work_items(plan)
    out = []
    for r in req.to_dict("records"):
        r["work_item_list"] = items[items.block_id == r["block_id"]].to_dict("records")
        out.append(r)
    return json.dumps(out, indent=2, default=str)
