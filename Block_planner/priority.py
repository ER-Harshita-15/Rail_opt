"""
Step 2 - Priority engine
Score = w1*Criticality + w2*Urgency + w3*FailureRisk(ML) + w4*TrafficImpact

 * Criticality, Urgency, TrafficImpact : rule-based, each normalised to [0, 1]
 * FailureRisk                         : LightGBM probability of failure within 30 days
 * Explanations                        : SHAP values for the risk model + a
                                         per-task breakdown of the final score

IMPORTANT (state this in your report): no real failure history is available, so the
model is trained on a SYNTHETIC history whose labels come from a hidden rule plus
noise. The pipeline is real; the reported accuracy only shows the pipeline works.
With real TMS/SMMS/TDMS history, only `make_history()` needs replacing.

Run:  python priority.py      (needs data/ from data_gen.py)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
SEED = 7

# Expert-set weights (later: tune with AHP or learn from past planner decisions)
WEIGHTS = dict(criticality=0.30, urgency=0.25, risk=0.25, traffic=0.20)

DEPT_CODE = {"ENGG": 0, "S&T": 1, "TRD": 2}
FEATURES = ["severity", "overdue_days", "past_failures", "dept_code",
            "route_class_A", "trains_per_day", "duration_min"]


# --------------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------------
def load_inputs():
    tasks = pd.read_csv(DATA / "tasks_common_schema.csv", parse_dates=["due_date"])
    sections = pd.read_csv(DATA / "sections.csv")
    traffic = pd.read_csv(DATA / "traffic_forecast.csv")
    return tasks, sections, traffic


def add_features(tasks, sections, traffic):
    tpd = (traffic.groupby(["section_id", "date"]).total_trains.sum()
           .groupby("section_id").mean().rename("trains_per_day"))
    df = (tasks.merge(sections[["section_id", "route_class"]], on="section_id")
               .merge(tpd, on="section_id"))
    df["dept_code"] = df.dept.map(DEPT_CODE)
    df["route_class_A"] = (df.route_class == "A").astype(int)
    return df


# --------------------------------------------------------------------------
# Synthetic failure history (stand-in for real defect history)
# --------------------------------------------------------------------------
def make_history(n=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    h = pd.DataFrame({
        "severity": rng.integers(1, 6, n),
        "overdue_days": rng.integers(-30, 40, n),
        "past_failures": rng.poisson(1.6, n),
        "dept_code": rng.integers(0, 3, n),
        "route_class_A": rng.integers(0, 2, n),
        "trains_per_day": rng.uniform(60, 130, n),
        "duration_min": rng.choice([60, 80, 90, 100, 120, 150, 180], n),
    })
    # hidden ground-truth rule: nonlinear, with interaction + noise
    z = (-4.2 + 0.55 * h.severity + 0.45 * h.past_failures
         + 0.045 * h.overdue_days.clip(lower=-10)
         + 0.012 * (h.trains_per_day - 90) + 0.35 * (h.dept_code == 1)
         + 0.04 * np.maximum(h.overdue_days, 0) * (h.severity >= 4)
         + rng.normal(0, 0.6, n))
    h["failed_30d"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return h


def train_risk_model(history):
    X, y = history[FEATURES], history.failed_30d
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED,
                                          stratify=y)
    model = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15,
                               min_child_samples=25, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.9, random_state=SEED, verbose=-1)
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    return model, auc, float(y.mean())


# --------------------------------------------------------------------------
# Rule-based components, each in [0, 1]
# --------------------------------------------------------------------------
def criticality(df):
    # severity of the defect, boosted on high-density (class A) routes
    return (0.7 * df.severity / 5 + 0.3 * np.where(df.route_class == "A", 1.0, 0.6)).clip(0, 1)


def urgency(df):
    # 0 when due in >=14 days, rising to 1 when ~30 days overdue
    return ((df.overdue_days + 14) / 44).clip(0, 1)


def traffic_impact(df):
    lo, hi = df.trains_per_day.min(), df.trains_per_day.max()
    return (df.trains_per_day - lo) / (hi - lo + 1e-9)


# --------------------------------------------------------------------------
def score_tasks(df, model):
    df = df.copy()
    df["risk"] = model.predict_proba(df[FEATURES])[:, 1]
    df["criticality"] = criticality(df)
    df["urgency"] = urgency(df)
    df["traffic"] = traffic_impact(df)
    for k, w in WEIGHTS.items():
        df[f"pts_{k}"] = w * df[k]
    df["priority_score"] = sum(df[f"pts_{k}"] for k in WEIGHTS)
    df["priority_rank"] = df.priority_score.rank(ascending=False, method="first").astype(int)
    # priority tier for the dashboard
    df["tier"] = pd.cut(df.priority_score, [-1, .35, .55, .72, 2],
                        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    return df.sort_values("priority_rank")


def explain(df, model, top_k=3):
    """SHAP: why does the risk model give this task its failure probability?"""
    sv = shap.TreeExplainer(model).shap_values(df[FEATURES])
    sv = sv[1] if isinstance(sv, list) else sv
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    reasons = []
    for i in range(len(df)):
        top = np.argsort(-np.abs(sv[i]))[:top_k]
        reasons.append("; ".join(
            f"{FEATURES[j]}={df.iloc[i][FEATURES[j]]:.4g} ({'+' if sv[i, j] > 0 else '-'}"
            f"{abs(sv[i, j]):.2f})" for j in top))
    return reasons


def score_breakdown(r):
    parts = {k: r[f"pts_{k}"] for k in WEIGHTS}
    main = max(parts, key=parts.get)
    return f"mainly {main} ({parts[main]:.2f} of {r.priority_score:.2f})"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tasks, sections, traffic = load_inputs()
    df = add_features(tasks, sections, traffic)

    history = make_history()
    model, auc, base_rate = train_risk_model(history)
    print(f"Risk model  | hold-out AUC = {auc:.3f} | base failure rate = {base_rate:.1%}"
          f"  (synthetic history)")

    scored = score_tasks(df, model)
    scored["risk_drivers_shap"] = explain(scored, model)
    scored["score_summary"] = scored.apply(score_breakdown, axis=1)

    cols = ["priority_rank", "task_id", "dept", "section_id", "task", "overdue_days",
            "severity", "risk", "criticality", "urgency", "traffic", "priority_score",
            "tier", "score_summary", "risk_drivers_shap"]
    scored[cols].round(3).to_csv(OUT / "prioritized_tasks.csv", index=False)

    print("\nTier counts:\n", scored.tier.value_counts().reindex(
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"]).to_string())
    print("\nTop 8 tasks:")
    print(scored[["priority_rank", "task_id", "section_id", "task", "overdue_days",
                  "risk", "priority_score", "tier"]].head(8).round(2).to_string(index=False))
    print("\nWhy is #1 ranked first?\n ", scored.iloc[0].score_summary)
    print("  risk drivers:", scored.iloc[0].risk_drivers_shap)
