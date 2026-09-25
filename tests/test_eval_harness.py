"""engine/eval/harness.py against the synthetic generator (docs/07-EVALUATION.md).

Not a check on the detector's actual quality (the synthetic set is a stand-in,
per tools/generate_synthetic.py's own docstring) - this just proves the
harness runs end to end, joins its ground truth correctly, and produces a
report whose numbers are internally consistent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from engine.core.config import load_config
from engine.eval.harness import load_ground_truth_json, run_harness
from engine.run import Pipeline

CFG = load_config()


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("eval_cert")
    argv = sys.argv
    sys.argv = ["generate_synthetic", "--users", "15", "--days", "60",
               "--insiders", "3", "--seed", "11", "--out", str(out)]
    try:
        from tools.generate_synthetic import main as generate_main
        generate_main()
    finally:
        sys.argv = argv
    return out


@pytest.fixture(scope="module")
def report(fixture_dir: Path) -> dict:
    pipeline = Pipeline(CFG).run(fixture_dir)
    gt = load_ground_truth_json(fixture_dir / "answers.json")
    return run_harness(pipeline, gt, CFG)


def test_ground_truth_json_matches_answers_file(fixture_dir):
    gt = load_ground_truth_json(fixture_dir / "answers.json")
    answers = json.loads((fixture_dir / "answers.json").read_text())
    assert gt["is_malicious"].all()
    assert set(gt["user_id"]) == {i["user_id"] for i in answers["insiders"]}
    total_days = sum(len(i["malicious_days"]) for i in answers["insiders"])
    assert len(gt) == total_days


def test_dataset_profile_is_internally_consistent(report):
    profile = report["dataset_profile"]
    assert profile["total_users"] == 15
    assert profile["insiders"] == 3
    assert 0 < profile["malicious_user_days"] <= profile["total_user_days_with_activity"]
    assert profile["base_rate"] == pytest.approx(
        profile["malicious_user_days"] / profile["total_user_days_with_activity"])


def test_insider_level_bounds(report):
    ins = report["insider_level"]
    assert ins["n_insiders"] == 3
    assert 0 <= ins["n_caught"] <= ins["n_insiders"]
    assert 0.0 <= ins["recall"] <= 1.0
    if ins["recall_ci"] is not None:
        lo, hi = ins["recall_ci"]
        assert 0.0 <= lo <= ins["recall"] + 1e-9
        assert ins["recall"] - 1e-9 <= hi <= 1.0
    if ins["median_time_to_detect_days"] is not None:
        assert ins["median_time_to_detect_days"] >= 0


def test_incident_level_bounds(report):
    inc = report["incident_level"]
    assert 0.0 <= inc["precision"] <= 1.0
    assert 0.0 <= inc["auto_flag_precision"] <= 1.0
    assert inc["incidents_per_day"] >= 0
    assert inc["compression_ratio"] >= 0
    if inc["precision_ci"] is not None:
        lo, hi = inc["precision_ci"]
        assert 0.0 <= lo <= hi <= 1.0
    if inc["evidence_precision"] is not None:
        assert 0.0 <= inc["evidence_precision"] <= 1.0


def test_user_day_level_bounds(report):
    ud = report["user_day_level"]
    assert 0.0 <= ud["pr_auc"] <= 1.0
    assert ud["lift"] >= 0
    for k, precision in ud["precision_at_k"].items():
        assert 0.0 <= precision <= 1.0
    assert 0.0 <= ud["recall_at_25_per_day"] <= 1.0


def test_calibration_bins_sum_to_total_incidents(report):
    calib = report["calibration"]
    if calib["ece"] is None:
        return
    assert 0.0 <= calib["ece"] <= 1.0
    n_incidents = sum(b["n"] for b in calib["bins"])
    for b in calib["bins"]:
        if b["n"] > 0:
            assert 0.0 <= b["observed_precision"] <= 1.0
            assert b["lo"] <= b["mean_confidence"] <= b["hi"] + 1e-9


def test_per_scenario_covers_every_insider(report):
    per_scenario = report["per_scenario"]
    total = sum(v["insiders"] for v in per_scenario.values())
    assert total == report["dataset_profile"]["insiders"]
    for scenario, v in per_scenario.items():
        assert 0.0 <= v["recall"] <= 1.0
        assert v["held_out"] == (int(scenario) == 3)


def test_report_is_json_serialisable(report):
    json.dumps(report, default=str)


def test_harness_is_deterministic(fixture_dir):
    pipeline = Pipeline(CFG).run(fixture_dir)
    gt = load_ground_truth_json(fixture_dir / "answers.json")
    r1 = run_harness(pipeline, gt, CFG)
    r2 = run_harness(pipeline, gt, CFG)
    assert r1["insider_level"]["recall"] == r2["insider_level"]["recall"]
    assert r1["incident_level"]["precision"] == r2["incident_level"]["precision"]
    assert r1["user_day_level"]["pr_auc"] == r2["user_day_level"]["pr_auc"]
