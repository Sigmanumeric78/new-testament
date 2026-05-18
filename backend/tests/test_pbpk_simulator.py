from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.pbpk.pbpk_master_simulator import (
    beverage_modifiers_path,
    parameter_library_path,
    population_modifiers_path,
    repo_root,
    run_simulation,
    validation_drink_profile,
    validation_user_profile,
    validate_result,
)


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = repo_root()
    library_df = pd.read_csv(parameter_library_path(root), dtype=str, keep_default_na=False)
    population_df = pd.read_csv(population_modifiers_path(root), dtype=str, keep_default_na=False)
    beverage_df = pd.read_csv(beverage_modifiers_path(root), dtype=str, keep_default_na=False)
    return library_df, population_df, beverage_df


def test_pbpk_validation_scenario_output_contract() -> None:
    library_df, population_df, beverage_df = load_sources()
    result = run_simulation(
        user_payload=validation_user_profile(),
        drink_payload=validation_drink_profile(),
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )
    summary = result["summary"]
    assert "bac_curve" in summary
    assert "acetaldehyde_curve" in summary
    assert "metabolism_rate" in summary
    assert "compound_burden" in summary
    assert len(summary["bac_curve"]) > 50
    assert len(summary["bac_curve"]) == len(summary["acetaldehyde_curve"])
    assert summary["peak_bac_percent"] > 0.0
    assert summary["time_to_peak_h"] >= 0.0
    assert summary["compound_burden"]["ethanol_auc_mg_h_l"] > 0.0
    ok, reasons = validate_result(result)
    assert ok, "; ".join(reasons)


def test_pbpk_validation_scenario_deterministic_repeatability() -> None:
    library_df, population_df, beverage_df = load_sources()
    inputs_user = validation_user_profile()
    inputs_drink = validation_drink_profile()
    result_1 = run_simulation(
        user_payload=inputs_user,
        drink_payload=inputs_drink,
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )
    result_2 = run_simulation(
        user_payload=inputs_user,
        drink_payload=inputs_drink,
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )

    bac_1 = np.array([point["bac_percent"] for point in result_1["summary"]["bac_curve"]], dtype=float)
    bac_2 = np.array([point["bac_percent"] for point in result_2["summary"]["bac_curve"]], dtype=float)
    acet_1 = np.array([point["acetaldehyde_mg_l"] for point in result_1["summary"]["acetaldehyde_curve"]], dtype=float)
    acet_2 = np.array([point["acetaldehyde_mg_l"] for point in result_2["summary"]["acetaldehyde_curve"]], dtype=float)

    assert np.array_equal(bac_1, bac_2)
    assert np.array_equal(acet_1, acet_2)
    assert result_1["summary"]["peak_bac_percent"] == result_2["summary"]["peak_bac_percent"]
    assert result_1["summary"]["time_to_peak_h"] == result_2["summary"]["time_to_peak_h"]
