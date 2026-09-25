"""Evaluation harness (docs/07-EVALUATION.md).

Runs the pipeline, joins its output against ground truth, and writes the
metric-suite report from section 8. Every rate here is *computed*, not typed
by hand (section 1.2) — the harness is the only place these numbers may
originate.

Ground truth can come from either source that shares the schema
`(user_id, date, scenario, is_malicious)`:
  - real CERT r4.2 `answers/` (`engine.ingest.ground_truth.load_ground_truth`)
  - the synthetic generator's `answers.json` (`load_ground_truth_json` below),
    used for fast local iteration and for this module's own tests, since the
    real corpus is a multi-gigabyte download not present in every environment

Scope of this version. Built: dataset profile, insider-level recall/TTD/
investigation burden, incident-level precision/volume/compression, user-day
PR-AUC/lift/precision@k/recall@25, confidence calibration (ECE), per-scenario
breakdown, and bootstrap 95% CIs at the user level for the two headline
numbers (section 6). Not yet built: the ablation study (`engine.eval.ablate`,
section 4) and the full per-rule weight-drift table (section 7) — both need a
config-flag harness of their own and are the natural next slice.

Usage:
    python -m engine.eval.harness --raw-dir data/raw/r4.2 \\
        --answers-dir data/raw/answers/answers --out data/artifacts/eval_report.json

    python -m engine.eval.harness --raw-dir data/raw \\
        --answers-json data/raw/answers.json --out data/artifacts/eval_report.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from engine.core.config import Config, load_config
from engine.correlate.incident import Incident
from engine.ingest.ground_truth import load_ground_truth
from engine.run import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
HELD_OUT_SCENARIOS = (3,)
INVESTIGATED_LANES = ("AUTO_FLAG", "ANALYST_REVIEW")
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


# --------------------------------------------------------------------- truth
def load_ground_truth_json(path: Path) -> pd.DataFrame:
    """The synthetic generator's `answers.json` -> the same shape
    `load_ground_truth` returns for real CERT, so downstream code never cares
    which source it got."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict] = []
    for insider in payload["insiders"]:
        for day in insider["malicious_days"]:
            rows.append({
                "user_id": insider["user_id"],
                "date": pd.Timestamp(day),
                "scenario": int(insider["scenario"]),
                "is_malicious": True,
            })
    out = pd.DataFrame(rows, columns=["user_id", "date", "scenario", "is_malicious"])
    return out.sort_values(["user_id", "date"], kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class Insider:
    user_id: str
    scenario: int
    malicious_days: tuple[pd.Timestamp, ...]

    @property
    def first_day(self) -> pd.Timestamp:
        return self.malicious_days[0]

    @property
    def final_day(self) -> pd.Timestamp:
        return self.malicious_days[-1]


def insiders_from_ground_truth(gt: pd.DataFrame) -> list[Insider]:
    out = []
    for user_id, grp in gt.groupby("user_id"):
        days = tuple(sorted(grp["date"]))
        scenario = int(grp["scenario"].iloc[0])
        out.append(Insider(user_id=user_id, scenario=scenario, malicious_days=days))
    return sorted(out, key=lambda i: i.user_id)


# --------------------------------------------------------------- user-days
def dataset_profile(pipeline: Pipeline, gt: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    activity = pipeline.features[["user_id", "date"]].drop_duplicates()
    total_users = int(pipeline.events["user_id"].nunique())
    total_user_days = int(len(activity))

    activity_norm = activity.assign(date=pd.to_datetime(activity["date"]).dt.normalize())
    gt_norm = gt.assign(date=gt["date"].dt.normalize())
    joined = activity_norm.merge(
        gt_norm[["user_id", "date", "is_malicious"]], on=["user_id", "date"], how="left")
    malicious_user_days = int(joined["is_malicious"].fillna(False).sum())

    dates = pd.to_datetime(activity["date"])
    return {
        "config_version": cfg.version,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "total_users": total_users,
        "total_user_days_with_activity": total_user_days,
        "insiders": int(gt["user_id"].nunique()),
        "malicious_user_days": malicious_user_days,
        "base_rate": malicious_user_days / total_user_days if total_user_days else 0.0,
        "date_range": [str(dates.min().date()), str(dates.max().date())],
        "per_scenario_insiders": {
            str(k): int(v) for k, v in gt.groupby("scenario")["user_id"].nunique().items()
        },
    }


def score_user_days(pipeline: Pipeline) -> pd.DataFrame:
    """One row per (user_id, date) the features cover: the day's risk/
    confidence/lane, taken from the incident that covers it when one exists
    (max risk, if more than one does), else 0 for a day with no incident."""
    by_user_date: dict[tuple[str, Any], list[Incident]] = {}
    for inc in pipeline.incidents:
        for d in pd.date_range(inc.window_start.normalize(), inc.window_end.normalize(), freq="D"):
            by_user_date.setdefault((inc.user_id, d.date()), []).append(inc)

    rows = []
    for row in pipeline.features.itertuples(index=False):
        date = pd.Timestamp(row.date)
        key = (row.user_id, date.date())
        covering = by_user_date.get(key, [])
        if covering:
            best = max(covering, key=lambda i: i.risk)
            risk, confidence, lane = best.risk, best.confidence, best.triage_lane
            incident_id = best.incident_id
        else:
            risk, confidence, lane, incident_id = 0.0, 0.0, None, None
        rows.append({
            "user_id": row.user_id, "date": date, "risk": risk,
            "confidence": confidence, "lane": lane, "incident_id": incident_id,
        })
    return pd.DataFrame(rows)


def label_user_days(scored: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    gt_norm = gt.assign(date=gt["date"].dt.normalize())
    out = scored.assign(date=pd.to_datetime(scored["date"]).dt.normalize()).merge(
        gt_norm[["user_id", "date", "is_malicious"]], on=["user_id", "date"], how="left")
    out["is_malicious"] = out["is_malicious"].fillna(False)
    return out


def user_day_metrics(labeled: pd.DataFrame) -> dict[str, Any]:
    labels = labeled["is_malicious"].to_numpy(dtype=bool)
    scores = labeled["risk"].to_numpy(dtype=float)
    base_rate = labels.mean() if len(labels) else 0.0

    pr_auc = float(average_precision_score(labels, scores)) if labels.any() else 0.0
    lift = (pr_auc / base_rate) if base_rate else 0.0

    precision_at_k: dict[str, float] = {}
    recall_at_25 = 0.0
    total_malicious = int(labels.sum())
    for k in (5, 10, 25, 50):
        tp = 0
        predicted = 0
        for _date, grp in labeled.groupby("date"):
            top = grp.nlargest(k, "risk")
            tp += int(top["is_malicious"].sum())
            predicted += len(top)
        precision_at_k[str(k)] = (tp / predicted) if predicted else 0.0
        if k == 25:
            recall_at_25 = (tp / total_malicious) if total_malicious else 0.0

    return {
        "pr_auc": pr_auc, "lift": lift, "base_rate": base_rate,
        "precision_at_k": precision_at_k, "recall_at_25_per_day": recall_at_25,
    }


# --------------------------------------------------------------- attribution
def _is_true_positive(incident: Incident, insiders_by_user: dict[str, Insider]) -> bool:
    """Section 5's attribution rule: an incident is a true positive only if
    its window falls inside that user's labelled malicious window — flagging
    a real insider on an unrelated day is a false positive, not a hit."""
    insider = insiders_by_user.get(incident.user_id)
    if insider is None:
        return False
    start, end = insider.first_day.date(), insider.final_day.date()
    return incident.window_start.date() <= end and incident.window_end.date() >= start


def _caught_before_final_act(incidents: list[Incident], insider: Insider) -> pd.Timestamp | None:
    tps = [inc for inc in incidents
          if inc.user_id == insider.user_id and inc.window_start.date() <= insider.final_day.date()]
    if not tps:
        return None
    return min(inc.window_start for inc in tps)


def insider_metrics(incidents: list[Incident], insiders: list[Insider]) -> dict[str, Any]:
    insiders_by_user = {i.user_id: i for i in insiders}
    tp_incidents = [inc for inc in incidents if _is_true_positive(inc, insiders_by_user)]

    caught_at: dict[str, pd.Timestamp] = {}
    for insider in insiders:
        first_flag = _caught_before_final_act(tp_incidents, insider)
        if first_flag is not None:
            caught_at[insider.user_id] = first_flag

    recall = len(caught_at) / len(insiders) if insiders else 0.0

    ttds = []
    for insider in insiders:
        if insider.user_id in caught_at:
            ttd = (caught_at[insider.user_id].normalize() - insider.first_day.normalize()).days
            ttds.append(max(0, ttd))
    median_ttd = float(np.median(ttds)) if ttds else None

    investigated_incidents = [inc for inc in incidents if inc.triage_lane in INVESTIGATED_LANES]
    non_insiders_investigated = {
        inc.user_id for inc in investigated_incidents if inc.user_id not in insiders_by_user
    }
    n_caught = len(caught_at)
    investigation_burden = (len(non_insiders_investigated) / n_caught) if n_caught else None

    return {
        "recall": recall,
        "n_caught": n_caught,
        "n_insiders": len(insiders),
        "median_time_to_detect_days": median_ttd,
        "investigation_burden": investigation_burden,
        "_caught_at": caught_at,   # internal, consumed by per-scenario / bootstrap below
    }


def incident_metrics(incidents: list[Incident], insiders: list[Insider], cfg: Config,
                     n_users: int, n_days: int) -> dict[str, Any]:
    insiders_by_user = {i.user_id: i for i in insiders}
    if not incidents:
        return {
            "precision": 0.0, "auto_flag_precision": 0.0, "incidents_per_day": 0.0,
            "incidents_per_day_per_1000_users": 0.0, "compression_ratio": 0.0,
            "evidence_precision": None,
        }

    tp_flags = [_is_true_positive(inc, insiders_by_user) for inc in incidents]
    precision = sum(tp_flags) / len(incidents)

    auto_flag = [inc for inc in incidents if inc.triage_lane == "AUTO_FLAG"]
    auto_flag_tp = [_is_true_positive(inc, insiders_by_user) for inc in auto_flag]
    auto_flag_precision = (sum(auto_flag_tp) / len(auto_flag)) if auto_flag else 0.0

    incidents_per_day = len(incidents) / n_days if n_days else 0.0
    per_1000 = incidents_per_day / (n_users / 1000) if n_users else 0.0
    compression_ratio = float(np.mean([inc.signal_count for inc in incidents]))

    # Event-level ground truth isn't available (answers give malicious *days*,
    # not event ids), so evidence precision is approximated at the day level:
    # of a true-positive incident's covered days, the fraction the insider's
    # own labelled malicious days account for.
    day_precisions = []
    for inc, is_tp in zip(incidents, tp_flags):
        if not is_tp:
            continue
        insider = insiders_by_user[inc.user_id]
        mal_days = {d.date() for d in insider.malicious_days}
        covered = {d.date() for d in
                  pd.date_range(inc.window_start.normalize(), inc.window_end.normalize(), freq="D")}
        day_precisions.append(len(covered & mal_days) / len(covered) if covered else 0.0)
    evidence_precision = float(np.mean(day_precisions)) if day_precisions else None

    return {
        "precision": precision, "auto_flag_precision": auto_flag_precision,
        "incidents_per_day": incidents_per_day,
        "incidents_per_day_per_1000_users": per_1000,
        "compression_ratio": compression_ratio,
        "evidence_precision": evidence_precision,
    }


def calibration_metrics(incidents: list[Incident], insiders: list[Insider],
                        n_bins: int = 10) -> dict[str, Any]:
    if not incidents:
        return {"ece": None, "bins": []}
    insiders_by_user = {i.user_id: i for i in insiders}
    conf = np.array([inc.confidence for inc in incidents])
    tp = np.array([_is_true_positive(inc, insiders_by_user) for inc in incidents], dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(incidents)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        n_b = int(mask.sum())
        if n_b == 0:
            bins.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2),
                        "n": 0, "observed_precision": None, "mean_confidence": None})
            continue
        observed = float(tp[mask].mean())
        mean_conf = float(conf[mask].mean())
        ece += (n_b / n) * abs(observed - mean_conf)
        bins.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": n_b,
                    "observed_precision": observed, "mean_confidence": mean_conf})
    return {"ece": ece, "bins": bins}


def per_scenario_metrics(incidents: list[Incident], insiders: list[Insider]) -> dict[str, Any]:
    out = {}
    for scenario in sorted({i.scenario for i in insiders}):
        scen_insiders = [i for i in insiders if i.scenario == scenario]
        result = insider_metrics(incidents, scen_insiders)
        out[str(scenario)] = {
            "insiders": len(scen_insiders),
            "recall": result["recall"],
            "median_ttd": result["median_time_to_detect_days"],
            "held_out": scenario in HELD_OUT_SCENARIOS,
        }
    return out


# --------------------------------------------------------------- bootstrap
def _bootstrap_ci(values: list[float], seed: int = BOOTSTRAP_SEED,
                  n_boot: int = N_BOOTSTRAP) -> tuple[float, float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_insider_recall(incidents: list[Incident], insiders: list[Insider]) -> tuple[float, float] | None:
    """Resample *insiders* with replacement (section 6 — not user-days, since
    a single insider's days are correlated)."""
    insiders_by_user = {i.user_id: i for i in insiders}
    tp_incidents = [inc for inc in incidents if _is_true_positive(inc, insiders_by_user)]
    caught = {insider.user_id for insider in insiders
             if _caught_before_final_act(tp_incidents, insider) is not None}
    flags = [1.0 if i.user_id in caught else 0.0 for i in insiders]
    return _bootstrap_ci(flags)


def bootstrap_incident_precision(incidents: list[Incident], insiders: list[Insider],
                                 all_users: list[str]) -> tuple[float, float] | None:
    """Resample *users* with replacement, recompute precision over the
    incidents belonging to the resampled multiset."""
    if not incidents or not all_users:
        return None
    insiders_by_user = {i.user_id: i for i in insiders}
    tp_flags = {inc.incident_id: _is_true_positive(inc, insiders_by_user) for inc in incidents}
    by_user: dict[str, list[Incident]] = {}
    for inc in incidents:
        by_user.setdefault(inc.user_id, []).append(inc)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    users = np.array(all_users)
    precisions = []
    for _ in range(N_BOOTSTRAP):
        sample = rng.choice(users, size=len(users), replace=True)
        incs = [inc for u in sample for inc in by_user.get(u, [])]
        if not incs:
            continue
        precisions.append(np.mean([tp_flags[inc.incident_id] for inc in incs]))
    if not precisions:
        return None
    return float(np.percentile(precisions, 2.5)), float(np.percentile(precisions, 97.5))


# --------------------------------------------------------------------- report
def run_harness(pipeline: Pipeline, gt: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    insiders = insiders_from_ground_truth(gt)
    profile = dataset_profile(pipeline, gt, cfg)

    scored = score_user_days(pipeline)
    labeled = label_user_days(scored, gt)
    ud_metrics = user_day_metrics(labeled)

    n_users = profile["total_users"]
    dates = pd.to_datetime(pipeline.features["date"])
    n_days = max(1, (dates.max() - dates.min()).days + 1)

    ins_metrics = insider_metrics(pipeline.incidents, insiders)
    recall_ci = bootstrap_insider_recall(pipeline.incidents, insiders)

    inc_metrics = incident_metrics(pipeline.incidents, insiders, cfg, n_users, n_days)
    all_users = sorted(pipeline.features["user_id"].unique())
    precision_ci = bootstrap_incident_precision(pipeline.incidents, insiders, all_users)

    calibration = calibration_metrics(pipeline.incidents, insiders)
    per_scenario = per_scenario_metrics(pipeline.incidents, insiders)

    report = {
        "config_version": cfg.version,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "dataset_profile": profile,
        "insider_level": {
            "recall": ins_metrics["recall"],
            "recall_ci": list(recall_ci) if recall_ci else None,
            "n_caught": ins_metrics["n_caught"],
            "n_insiders": ins_metrics["n_insiders"],
            "median_time_to_detect_days": ins_metrics["median_time_to_detect_days"],
            "investigation_burden": ins_metrics["investigation_burden"],
        },
        "incident_level": {
            "precision": inc_metrics["precision"],
            "precision_ci": list(precision_ci) if precision_ci else None,
            "auto_flag_precision": inc_metrics["auto_flag_precision"],
            "incidents_per_day": inc_metrics["incidents_per_day"],
            "incidents_per_day_per_1000_users": inc_metrics["incidents_per_day_per_1000_users"],
            "compression_ratio": inc_metrics["compression_ratio"],
            "evidence_precision": inc_metrics["evidence_precision"],
        },
        "user_day_level": ud_metrics,
        "calibration": calibration,
        "per_scenario": per_scenario,
        "ablation": None,   # not built yet - see module docstring
        "per_rule": None,   # not built yet - see module docstring
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SentinelTrace evaluation harness")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "r4.2")
    parser.add_argument("--answers-dir", type=Path, default=None,
                       help="real CERT answers/ directory")
    parser.add_argument("--answers-json", type=Path, default=None,
                       help="synthetic generator's answers.json (alternative to --answers-dir)")
    parser.add_argument("--out", type=Path,
                       default=REPO_ROOT / "data" / "artifacts" / "eval_report.json")
    args = parser.parse_args(argv)

    if not args.answers_dir and not args.answers_json:
        parser.error("one of --answers-dir or --answers-json is required")

    cfg = load_config()
    pipeline = Pipeline(cfg).run(args.raw_dir)

    if args.answers_dir:
        gt = load_ground_truth(args.answers_dir)
    else:
        gt = load_ground_truth_json(args.answers_json)

    report = run_harness(pipeline, gt, cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
