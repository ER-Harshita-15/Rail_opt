"""
Step 1 - Common schema + synthetic data generator
Automatic Block Planning System (Major Project)

Generates data that mimics TMS (Engineering), SMMS (S&T), TDMS (Traction),
COA (corridor block windows), the train timetable and the goods forecast,
and normalizes the maintenance sources into ONE common task schema.

Run:  python data_gen.py            -> writes CSVs into ./data
"""
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
TODAY = date(2026, 10, 26)          # planning start (a Monday)
OUT = Path(__file__).parent / "data"

# --------------------------------------------------------------------------
# 1. Network: sections (station-to-station block sections on a trunk route)
# --------------------------------------------------------------------------
STATIONS = ["BSP", "BLSP-J", "KRBA", "USLP", "PND", "AKLT", "TLD", "SKR", "DPK", "RIG"]


def make_sections() -> pd.DataFrame:
    rows = []
    for i in range(len(STATIONS) - 1):
        rows.append(dict(
            section_id=f"{STATIONS[i]}-{STATIONS[i+1]}",
            km_start=round(i * 22.0, 1),
            km_end=round((i + 1) * 22.0, 1),
            lines=int(rng.choice([2, 2, 3])),                 # double/triple line
            route_class=str(rng.choice(["A", "B"], p=[0.55, 0.45])),  # A = high density trunk
            electrified=True,
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Timetable + goods forecast  (trains per section per hour, 28 days)
# --------------------------------------------------------------------------
def make_traffic(sections: pd.DataFrame, horizon_days: int = 28) -> pd.DataFrame:
    rows = []
    # relative hourly shape: morning + evening passenger peaks
    shape = np.array([.3, .2, .2, .2, .3, .6, 1.0, 1.2, 1.1, .8, .7, .7,
                      .8, .8, .7, .8, 1.0, 1.3, 1.2, 1.0, .8, .6, .5, .4])
    for _, s in sections.iterrows():
        base_pax = 3.0 if s.route_class == "A" else 1.8
        base_goods = 2.2 if s.route_class == "A" else 1.4
        for d in range(horizon_days):
            day = TODAY + timedelta(days=d)
            goods_surge = 1.35 if day.weekday() in (1, 2) else 1.0   # forecast surge Tue/Wed
            for h in range(24):
                pax = rng.poisson(base_pax * shape[h])
                goods = rng.poisson(base_goods * goods_surge * (1.2 - 0.4 * shape[h]))
                rows.append(dict(section_id=s.section_id, date=day, hour=h,
                                 pax_trains=int(pax), goods_trains=int(goods),
                                 total_trains=int(pax + goods)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. COA: permitted corridor block windows
#    Each section gets daily candidate windows; low-traffic hours are preferred.
# --------------------------------------------------------------------------
def make_corridor_windows(sections: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    rows, wid = [], 0
    candidate_starts = [(1, 4), (10, 13), (13, 16)]   # (start_hour, end_hour) templates
    for _, s in sections.iterrows():
        for d in range(28):
            day = TODAY + timedelta(days=d)
            if day.weekday() == 6 and rng.random() < 0.3:      # some Sundays unavailable
                continue
            for (h0, h1) in candidate_starts:
                if rng.random() < 0.25:                         # COA does not offer every window
                    continue
                t = traffic[(traffic.section_id == s.section_id) & (traffic.date == day)
                            & (traffic.hour >= h0) & (traffic.hour < h1)]
                rows.append(dict(
                    window_id=f"W{wid:05d}", section_id=s.section_id, date=day,
                    start_hour=h0, end_hour=h1, length_min=(h1 - h0) * 60,
                    trains_affected=int(t.total_trains.sum()),
                    goods_trains_affected=int(t.goods_trains.sum()),
                    power_block_possible=bool(rng.random() < 0.7),
                ))
                wid += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Department maintenance systems (raw, each with its OWN field names)
# --------------------------------------------------------------------------
def make_tms(sections, n=60):           # Engineering - Track Management System
    types = [("Rail fracture", 5, 120), ("Track geometry defect", 3, 150),
             ("Ballast deficiency", 2, 180), ("Weld repair", 4, 90), ("Fish-plate renewal", 3, 60)]
    rows = []
    for i in range(n):
        s = sections.sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        t = types[rng.integers(len(types))]
        km = round(rng.uniform(s.km_start, s.km_end), 2)
        due = TODAY + timedelta(days=int(rng.integers(-20, 40)))
        rows.append(dict(defect_id=f"TMS{i:04d}", sec=s.section_id, km=km, defect=t[0],
                         severity=t[1], due=due, est_minutes=t[2],
                         past_defects_12m=int(rng.poisson(1.5))))
    return pd.DataFrame(rows)


def make_smms(sections, n=55):          # S&T - Signalling Maintenance & Management System
    types = [("Signal lamp/relay failure", 5, 90), ("Point machine maintenance", 4, 120),
             ("Track circuit fault", 4, 100), ("Axle counter calibration", 3, 80),
             ("Cable insulation test", 2, 150)]
    rows = []
    for i in range(n):
        s = sections.sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        t = types[rng.integers(len(types))]
        km = round(rng.uniform(s.km_start, s.km_end), 2)
        due = TODAY + timedelta(days=int(rng.integers(-15, 35)))
        rows.append(dict(asset_ref=f"SMMS{i:04d}", block_section=s.section_id, location_km=km,
                         task_desc=t[0], priority_level=t[1], maint_due_date=due,
                         duration_min=t[2], failures_last_year=int(rng.poisson(2.0))))
    return pd.DataFrame(rows)


def make_tdms(sections, n=45):          # Traction - Traction Distribution Mgmt System
    types = [("OHE wire tension adjustment", 3, 120), ("Insulator replacement", 4, 150),
             ("Catenary mast repair", 4, 180), ("Sectioning post maintenance", 3, 90),
             ("Dropper replacement", 2, 100)]
    rows = []
    for i in range(n):
        s = sections.sample(1, random_state=int(rng.integers(1e6))).iloc[0]
        t = types[rng.integers(len(types))]
        km = round(rng.uniform(s.km_start, s.km_end), 2)
        due = TODAY + timedelta(days=int(rng.integers(-18, 38)))
        rows.append(dict(mast_id=f"TDMS{i:04d}", ohe_section=s.section_id, chainage_km=km,
                         work=t[0], criticality=t[1], schedule_due=due, hrs=t[2] / 60,
                         faults_12m=int(rng.poisson(1.2))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. COMMON SCHEMA + adapters
# --------------------------------------------------------------------------
@dataclass
class Task:
    task_id: str
    dept: str                 # ENGG / S&T / TRD
    source_system: str
    section_id: str
    km: float
    task: str
    severity: int             # 1-5 normalized
    due_date: date
    overdue_days: int         # >0 means already overdue
    duration_min: int         # work time only
    setup_min: int            # travel, protection, clearance
    block_type: str           # LINE / POWER / LINE+POWER
    gang_type: str            # resource needed
    gang_size: int
    past_failures: int        # history feature for ML risk model


SETUP = {"ENGG": 20, "S&T": 15, "TRD": 25}
GANG = {"ENGG": ("track_gang", 8), "S&T": ("signal_team", 4), "TRD": ("ohe_tower_wagon", 6)}


def _overdue(due): return (TODAY - due).days


def adapt_tms(df):
    return [Task(r.defect_id, "ENGG", "TMS", r.sec, r.km, r.defect, int(r.severity), r.due,
                 _overdue(r.due), int(r.est_minutes), SETUP["ENGG"], "LINE",
                 *GANG["ENGG"], int(r.past_defects_12m)) for r in df.itertuples()]


def adapt_smms(df):
    return [Task(r.asset_ref, "S&T", "SMMS", r.block_section, r.location_km, r.task_desc,
                 int(r.priority_level), r.maint_due_date, _overdue(r.maint_due_date),
                 int(r.duration_min), SETUP["S&T"], "LINE", *GANG["S&T"],
                 int(r.failures_last_year)) for r in df.itertuples()]


def adapt_tdms(df):
    return [Task(r.mast_id, "TRD", "TDMS", r.ohe_section, r.chainage_km, r.work,
                 int(r.criticality), r.schedule_due, _overdue(r.schedule_due),
                 int(round(r.hrs * 60)), SETUP["TRD"], "POWER", *GANG["TRD"],
                 int(r.faults_12m)) for r in df.itertuples()]


def build_tasks(sections) -> pd.DataFrame:
    tasks = adapt_tms(make_tms(sections)) + adapt_smms(make_smms(sections)) \
        + adapt_tdms(make_tdms(sections))
    return pd.DataFrame([asdict(t) for t in tasks])


# --------------------------------------------------------------------------
if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    sections = make_sections()
    traffic = make_traffic(sections)
    windows = make_corridor_windows(sections, traffic)
    tasks = build_tasks(sections)

    sections.to_csv(OUT / "sections.csv", index=False)
    traffic.to_csv(OUT / "traffic_forecast.csv", index=False)
    windows.to_csv(OUT / "coa_corridor_windows.csv", index=False)
    tasks.to_csv(OUT / "tasks_common_schema.csv", index=False)

    print(f"sections: {len(sections)} | traffic rows: {len(traffic)} | "
          f"windows: {len(windows)} | tasks: {len(tasks)}")
    print("\nTasks by dept:\n", tasks.dept.value_counts().to_string())
    print(f"\nOverdue tasks: {(tasks.overdue_days > 0).sum()} of {len(tasks)}")
    print("\nSample:\n", tasks.head(5).to_string(index=False))
