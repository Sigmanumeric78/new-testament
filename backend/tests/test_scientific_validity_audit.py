from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasoning import scientific_validity_audit as sva


def test_ethanol_dose_math_cases_pass() -> None:
    passed, details = sva._dose_math_checks()

    assert passed is True
    assert len(details) == 4
    assert all(item["passes"] for item in details)

    expected = {item["case_id"]: item["expected_ethanol_g"] for item in details}
    observed = {item["case_id"]: item["observed_ethanol_g"] for item in details}

    assert abs(observed["dose_180ml_whisky_40"] - expected["dose_180ml_whisky_40"]) <= 0.25
    assert abs(observed["dose_200ml_vodka_40"] - expected["dose_200ml_vodka_40"]) <= 0.25
    assert abs(observed["dose_500ml_beer_5"] - expected["dose_500ml_beer_5"]) <= 0.25
    assert abs(observed["dose_150ml_wine_12"] - expected["dose_150ml_wine_12"]) <= 0.25


def test_bac_plausibility_and_time_to_sober_checks() -> None:
    bac_pass, bac_details, simulations = sva._bac_plausibility_checks()
    assert bac_pass is True
    assert bac_details["checks"]["case_a_male_fed_180_plausible_band"] is True
    assert bac_details["checks"]["case_b_male_fed_30_below_0_04"] is True
    assert bac_details["checks"]["case_c_female_fasted_higher_than_male_fasted"] is True
    assert bac_details["checks"]["case_d_male_fasted_higher_than_fed"] is True

    time_pass, time_details = sva._time_to_sober_checks(
        simulations,
        "Estimated time until alcohol clears: about 11 hours",
    )
    assert time_pass is True
    assert time_details["checks"]["nonnegative_times"] is True
    assert time_details["checks"]["higher_dose_longer_clearance"] is True
    assert time_details["checks"]["tiny_dose_clears_faster"] is True
    assert time_details["checks"]["user_facing_time_rounded"] is True


def test_input_extraction_truthfulness_passes() -> None:
    passed, details = sva._input_extraction_truthfulness_check()
    assert passed is True
    checks = details["checks"]
    assert all(bool(value) for value in checks.values())


def test_scientific_validity_audit_end_to_end_outputs_report_and_csv(tmp_path: Path) -> None:
    report = sva.run_scientific_validity_audit(timeout_seconds=30)

    assert report["dose_math_pass"] is True
    assert report["bac_plausibility_pass"] is True
    assert report["time_to_sober_pass"] is True
    assert report["input_extraction_pass"] is True
    assert report["assumption_tracking_pass"] is True
    assert report["driving_safety_pass"] is True
    assert report["continue_drinking_safety_pass"] is True
    assert report["emergency_detection_pass"] is True
    assert report["retrieval_relevance_pass"] is True
    assert report["final_answer_truthfulness_pass"] is True
    assert report["safe_for_fastapi_after_scientific_audit"] is True
    assert report["failed_cases"] == 0

    encoded = json.dumps(report, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["total_cases"] == report["total_cases"]

    report_path = tmp_path / "scientific_validity_audit_report.json"
    cases_path = tmp_path / "scientific_validity_cases.csv"
    sva.write_scientific_validity_report(report, report_path)
    sva._write_cases_csv(report["details"]["benchmark_rows"], cases_path)

    assert report_path.exists()
    assert cases_path.exists()
    csv_text = cases_path.read_text(encoding="utf-8")
    assert "case_id,query,expected_behavior" in csv_text
    assert "pass" in csv_text
