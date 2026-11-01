"""
Step 6 - synthetic arrival events for the rolling re-planning experiment.

An event is a day on which new defects become known to the planner:
    routine    new maintenance defects with a due date 8-25 days after arrival; scored with the SAME
               Step 2 pipeline (rule-based parts + LightGBM risk model) as the original tasks
    emergency  an urgent defect: severity 5, top tier (CRITICAL), priority score 0.95 (W3 of
               whatif.py), treated as due the day before it arrives

Assumptions (state them in the report)
    E1  Arrival days, departments, sections, locations, durations and severities are random (seeded)
        and carry no real-world meaning; durations are <= 150 min so every task fits a window.
    E2  Routine arrivals are scored with the Step 2 model trained on its synthetic history.
        Scores of the ORIGINAL tasks are unchanged by adding new rows (checked in the tests).
    E3  A new task's overdue_days is measured at the plan origin (day 0) like every other task, so
        a task that is due on day d has overdue_days = -d.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

import horizons as h
import optimizer as o
import whatif as wi

START = o.START
SETUP = {"ENGG": 20, "S&T": 15, "TRD": 25}
GANG = {"ENGG": ("track_gang", 8), "S&T": ("signal_team", 4), "TRD": ("ohe_tower_wagon", 6)}
SOURCE = {"ENGG": "TMS", "S&T": "SMMS", "TRD": "TDMS"}
TEMPLATES = {
    "ENGG": [("Rail fracture", 5, 120), ("Track geometry defect", 3, 150), ("Weld repair", 4, 90),
             ("Fish-plate renewal", 3, 60)],
    "S&T": [("Signal lamp/relay failure", 5, 90), ("Point machine maintenance", 4, 120),
            ("Track circuit fault", 4, 100), ("Axle counter calibration", 3, 80)],
    "TRD": [("OHE wire tension adjustment", 3, 120), ("Insulator replacement", 4, 150),
            ("Sectioning post maintenance", 3, 90), ("Dropper replacement", 2, 100)],
}
DEFAULT_EVENTS = [(2, "emergency", "S&T"), (4, "routine", 2), (6, "emergency", "TRD"),
                  (9, "routine", 2), (12, "emergency", "ENGG")]
TIER_BINS, TIER_LABELS = [-1, .35, .55, .72, 2], ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def tier_of(score):
    return str(pd.cut([score], TIER_BINS, labels=TIER_LABELS)[0])


def _raw_row(tid, dept, section, km, name, severity, duration, due_idx, past_failures):
    return dict(task_id=tid, dept=dept, source_system=SOURCE[dept], section_id=section, km=round(float(km), 2),
                task=name, severity=int(severity), due_date=str((START + timedelta(days=int(due_idx))).date()),
                overdue_days=int(-due_idx), duration_min=int(duration), setup_min=SETUP[dept],
                block_type="POWER" if dept == "TRD" else "LINE", gang_type=GANG[dept][0],
                gang_size=GANG[dept][1], past_failures=int(past_failures))


def score_routine(raw_tasks, new_raw):
    """Step 2 scores for new routine tasks. The original tasks are scored in the same batch so the
    min/max normalisation of traffic impact is unchanged (checked by the tests)."""
    import priority as pr
    sections = pd.read_csv(o.DATA / "sections.csv")
    traffic = pd.read_csv(o.DATA / "traffic_forecast.csv")
    df = pr.add_features(pd.concat([raw_tasks, new_raw], ignore_index=True), sections, traffic)
    model, _, _ = pr.train_risk_model(pr.make_history())
    scored = pr.score_tasks(df, model)
    return scored.set_index("task_id")


def generate_events(T, raw_tasks, sections, spec=None, seed=11, early_days=h.EARLY_DAYS):
    """List of dicts(day, kind, tasks) - `tasks` are rows in the prepared task-table format."""
    rng = np.random.default_rng(seed)
    spec = spec or DEFAULT_EVENTS
    secs = sections.set_index("section_id")
    raw_new, plan = [], []                                   # first build raw rows, then score the routine ones
    n_emg = n_new = 0
    for day, kind, arg in spec:
        if kind == "emergency":
            n_emg += 1
            dept = arg
            name, _, dur = TEMPLATES[dept][int(rng.integers(len(TEMPLATES[dept])))]
            sec = str(rng.choice(sections.section_id))
            km = rng.uniform(secs.km_start[sec], secs.km_end[sec])
            plan.append(dict(day=day, kind=kind, rows=[("EMG-%03d" % n_emg, dept, sec, km, f"EMERGENCY {name}", 5, dur,
                                                         day - 1, 0)]))
        else:
            rows = []
            for _ in range(int(arg)):
                n_new += 1
                dept = str(rng.choice(list(TEMPLATES)))
                name, sev, dur = TEMPLATES[dept][int(rng.integers(len(TEMPLATES[dept])))]
                sec = str(rng.choice(sections.section_id))
                km = rng.uniform(secs.km_start[sec], secs.km_end[sec])
                rows.append(("NEW-%03d" % n_new, dept, sec, km, name, int(rng.integers(2, 5)), dur,
                             day + int(rng.integers(8, 26)), int(rng.poisson(1.5))))
            plan.append(dict(day=day, kind=kind, rows=rows))
    routine_raw = pd.DataFrame([_raw_row(*r) for e in plan if e["kind"] == "routine" for r in e["rows"]])
    scores = score_routine(raw_tasks, routine_raw) if len(routine_raw) else None
    events = []
    for e in plan:
        parts = []
        for (tid, dept, sec, km, name, sev, dur, due_idx, pf) in e["rows"]:
            if e["kind"] == "emergency":
                parts.append(wi.make_emergency_task(T, dept, sec, km, dur, task=name, task_id=tid, severity=sev,
                                                    priority_score=0.95, overdue_days=-due_idx, early_days=early_days,
                                                    tier="CRITICAL", past_failures=pf))
            else:
                sc = round(float(scores.priority_score[tid]), 3)
                parts.append(wi.make_emergency_task(T, dept, sec, km, dur, task=name, task_id=tid, severity=sev,
                                                    priority_score=sc, overdue_days=-due_idx, early_days=early_days,
                                                    tier=tier_of(sc), past_failures=pf))
        tasks = pd.concat(parts, ignore_index=True)
        tasks["due_date"] = [str((START + timedelta(days=int(r[7]))).date()) for r in e["rows"]]
        events.append(dict(day=int(e["day"]), kind=e["kind"], tasks=tasks))
    return sorted(events, key=lambda e: e["day"])
