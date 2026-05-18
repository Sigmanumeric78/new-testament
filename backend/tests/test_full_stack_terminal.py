"""Full-stack terminal integration test for PBPK + Neo4j explainability."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - dependency branch
    GraphDatabase = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.pbpk.pbpk_master_simulator import (
    beverage_modifiers_path,
    parameter_library_path,
    population_modifiers_path,
    repo_root,
    run_simulation,
)
from utils.config import get_neo4j_config

TARGET_BEVERAGE_TOKEN = "whisk"
ABSORPTION_PARAMETERS = ("gastric_emptying_rate", "intestinal_absorption_rate")


def _load_pbpk_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = repo_root()
    library_df = pd.read_csv(parameter_library_path(root), dtype=str, keep_default_na=False)
    population_df = pd.read_csv(population_modifiers_path(root), dtype=str, keep_default_na=False)
    beverage_df = pd.read_csv(beverage_modifiers_path(root), dtype=str, keep_default_na=False)
    return library_df, population_df, beverage_df


def _scenario_user(sex: str, weight: float, fed_or_fasted: str, body_fat_percent: float) -> Dict[str, Any]:
    height = 178.0 if sex == "male" else 165.0
    return {
        "sex": sex,
        "weight": weight,
        "height": height,
        "age": 35,
        "body_fat_percent": body_fat_percent,
        "fed_or_fasted": fed_or_fasted,
        "liver_status": "healthy",
    }


def _scenario_drink(volume_ml: float) -> Dict[str, Any]:
    return {
        "beverage": "whisky",
        "volume_ml": volume_ml,
        "abv": 40.0,
        "serving_time": 0.0,
    }


def _assert(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def _run_pbpk_checks(
    library_df: pd.DataFrame,
    population_df: pd.DataFrame,
    beverage_df: pd.DataFrame,
    errors: List[str],
) -> Dict[str, Any]:
    result = run_simulation(
        user_payload=_scenario_user("male", 75.0, "fed", 20.0),
        drink_payload=_scenario_drink(180.0),
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )
    summary = result["summary"]
    curve = summary["bac_curve"]
    bac = np.array([float(row["bac_percent"]) for row in curve], dtype=float)
    t = np.array([float(row["time_h"]) for row in curve], dtype=float)
    vd_l = float(summary["model_parameters"]["vd_l"])
    dose_g = float(summary["model_parameters"]["dose_g"])
    systemic_mass_g = bac * 10.0 * vd_l

    peak_bac = float(summary["peak_bac_percent"])
    t_peak = float(summary["time_to_peak_h"])
    t_sober = summary["time_to_sober_h"]
    t_sober_v = float(t_sober) if t_sober is not None else None

    _assert(0.08 <= peak_bac <= 0.12, f"PBPK peak BAC out of expected range: {peak_bac}", errors)
    _assert(0.5 <= t_peak <= 4.0, f"PBPK time_to_peak_h unrealistic: {t_peak}", errors)
    _assert(t_sober_v is not None and 3.0 <= t_sober_v <= 18.0, f"PBPK time_to_sober_h unrealistic: {t_sober}", errors)
    _assert(np.all(bac >= -1e-12), "BAC curve contains negative values.", errors)
    _assert(np.all(systemic_mass_g >= -1e-9), "Derived systemic ethanol mass became negative.", errors)
    _assert(np.max(systemic_mass_g) <= dose_g * 1.02, "Mass conservation sanity failed: systemic mass exceeded dose.", errors)
    _assert(np.all(np.diff(t) > 0), "Time grid is non-monotonic.", errors)

    return {
        "peak_bac_percent": peak_bac,
        "time_to_peak_h": t_peak,
        "time_to_sober_h": t_sober_v,
        "dose_g": dose_g,
        "max_systemic_ethanol_g_from_bac": float(np.max(systemic_mass_g)),
    }


def _run_graph_checks(errors: List[str]) -> Dict[str, Any]:
    if GraphDatabase is None:
        raise RuntimeError("neo4j Python driver is not installed.")

    config = get_neo4j_config()
    driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))

    try:
        with driver.session(database=config["database"]) as session:
            q_enzyme = """
            MATCH (b:Beverage)-[:CONTAINS]->(c:Compound)-[:METABOLIZED_BY]->(e:Enzyme)
            WHERE toLower(coalesce(b.name,'')) CONTAINS $bev
               OR toLower(coalesce(b.category,'')) CONTAINS $bev
            RETURN b.name AS beverage,
                   count(*) AS path_count,
                   collect(DISTINCT c.name)[0..20] AS compounds,
                   collect(DISTINCT e.name)[0..20] AS enzymes
            ORDER BY path_count DESC, beverage
            LIMIT 1
            """
            enzyme_row = session.run(q_enzyme, bev=TARGET_BEVERAGE_TOKEN).single()
            _assert(enzyme_row is not None, "No Beverage->Compound->Enzyme path returned.", errors)
            if enzyme_row is None:
                enzyme_summary = {"path_count": 0, "compounds": [], "enzymes": [], "reasoning_chain": ""}
            else:
                compounds = [c for c in enzyme_row["compounds"] if c]
                enzymes = [e for e in enzyme_row["enzymes"] if e]
                _assert(int(enzyme_row["path_count"]) > 0, "Beverage->Compound->Enzyme path_count is zero.", errors)
                _assert(len(compounds) > 0, "Explainability query returned no compounds.", errors)
                _assert(len(enzymes) > 0, "Explainability query returned no enzymes.", errors)
                enzyme_summary = {
                    "beverage": enzyme_row["beverage"],
                    "path_count": int(enzyme_row["path_count"]),
                    "compounds": compounds,
                    "enzymes": enzymes,
                    "reasoning_chain": (
                        f"{enzyme_row['beverage']} contains {', '.join(compounds[:3])}, "
                        f"which are metabolized by {', '.join(enzymes[:3])}."
                    ),
                }

            q_hangover = """
            MATCH (b:Beverage)-[:CONTAINS]->(c:Compound)-[:CONTRIBUTES_TO]->(t:ToxicityRisk)
            WHERE (toLower(coalesce(b.name,'')) CONTAINS $bev
               OR toLower(coalesce(b.category,'')) CONTAINS $bev)
              AND toLower(coalesce(t.risk_type,'')) CONTAINS 'hangover'
            RETURN count(*) AS path_count,
                   collect(DISTINCT c.name)[0..20] AS compounds,
                   collect(DISTINCT t.risk_type)[0..10] AS risk_types
            """
            hangover_row = session.run(q_hangover, bev=TARGET_BEVERAGE_TOKEN).single()
            _assert(hangover_row is not None, "No hangover toxicity query row returned.", errors)
            if hangover_row is None:
                hangover_summary = {"path_count": 0, "compounds": [], "risk_types": []}
            else:
                path_count = int(hangover_row["path_count"])
                compounds = [c for c in hangover_row["compounds"] if c]
                risk_types = [r for r in hangover_row["risk_types"] if r]
                _assert(path_count > 0, "No Beverage->Compound->ToxicityRisk hangover paths found.", errors)
                _assert(len(compounds) > 0, "Hangover query returned no compounds.", errors)
                hangover_summary = {
                    "path_count": path_count,
                    "compounds": compounds,
                    "risk_types": risk_types,
                }

            q_physiology = """
            MATCH (g:PopulationGroup)-[:MODIFIES]->(p:PBPKParameter)-[:AFFECTS]->(bc:BodyCompartment)
            WHERE g.group_name IN ['female', 'fasted']
            RETURN g.group_name AS group_name,
                   count(*) AS path_count,
                   collect(DISTINCT p.parameter_name)[0..10] AS parameters,
                   collect(DISTINCT bc.name)[0..10] AS compartments
            ORDER BY g.group_name
            """
            physiology_rows = list(session.run(q_physiology))
            by_group: Dict[str, Dict[str, Any]] = {}
            for row in physiology_rows:
                by_group[str(row["group_name"])] = {
                    "path_count": int(row["path_count"]),
                    "parameters": [p for p in row["parameters"] if p],
                    "compartments": [c for c in row["compartments"] if c],
                }
            for group in ("female", "fasted"):
                payload = by_group.get(group)
                _assert(payload is not None, f"Missing physiology modifier path for group '{group}'.", errors)
                if payload is None:
                    continue
                _assert(payload["path_count"] > 0, f"Zero physiology paths for group '{group}'.", errors)

            q_fed_fasted_paths = """
            MATCH (g:PopulationGroup)-[:MODIFIES]->(p:PBPKParameter)-[:AFFECTS]->(:BodyCompartment)
            WHERE g.group_name IN ['fed', 'fasted']
              AND p.parameter_name IN ['gastric_emptying_rate', 'intestinal_absorption_rate']
            RETURN g.group_name AS group_name, p.parameter_name AS parameter_name, count(*) AS c
            """
            fed_fasted_rows = list(session.run(q_fed_fasted_paths))
            presence = {(str(row["group_name"]), str(row["parameter_name"])): int(row["c"]) for row in fed_fasted_rows}
            for group in ("fed", "fasted"):
                for parameter in ABSORPTION_PARAMETERS:
                    _assert(
                        presence.get((group, parameter), 0) > 0,
                        f"Missing graph path for ({group})-[:MODIFIES]->({parameter})-[:AFFECTS]->BodyCompartment.",
                        errors,
                    )

    finally:
        driver.close()

    return {
        "explainability_beverage_compound_enzyme": enzyme_summary,
        "toxicity_hangover": hangover_summary,
        "physiology_modifier_paths": by_group,
    }


def _run_pbpk_graph_consistency_checks(
    library_df: pd.DataFrame,
    population_df: pd.DataFrame,
    beverage_df: pd.DataFrame,
    errors: List[str],
) -> Dict[str, Any]:
    fed_sim = run_simulation(
        user_payload=_scenario_user("male", 75.0, "fed", 20.0),
        drink_payload=_scenario_drink(180.0),
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )
    fasted_sim = run_simulation(
        user_payload=_scenario_user("male", 75.0, "fasted", 20.0),
        drink_payload=_scenario_drink(180.0),
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )
    female_fasted_sim = run_simulation(
        user_payload=_scenario_user("female", 60.0, "fasted", 28.0),
        drink_payload=_scenario_drink(180.0),
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )
    low_dose_sim = run_simulation(
        user_payload=_scenario_user("male", 75.0, "fed", 20.0),
        drink_payload=_scenario_drink(30.0),
        library_df=library_df,
        population_df=population_df,
        beverage_df=beverage_df,
    )

    fed_peak = float(fed_sim["summary"]["peak_bac_percent"])
    fed_tpeak = float(fed_sim["summary"]["time_to_peak_h"])
    fasted_peak = float(fasted_sim["summary"]["peak_bac_percent"])
    fasted_tpeak = float(fasted_sim["summary"]["time_to_peak_h"])
    female_peak = float(female_fasted_sim["summary"]["peak_bac_percent"])
    low_dose_peak = float(low_dose_sim["summary"]["peak_bac_percent"])

    _assert(fasted_peak > fed_peak, "PBPK consistency failed: fasted peak BAC is not higher than fed.", errors)
    _assert(fasted_tpeak <= fed_tpeak, "PBPK consistency failed: fasted time_to_peak is not faster/equal vs fed.", errors)
    _assert(female_peak > fasted_peak, "PBPK consistency failed: female fasted peak is not higher than male fasted.", errors)
    _assert(low_dose_peak < 0.04, "PBPK consistency failed: low-dose fed peak BAC should be < 0.04.", errors)

    direction_df = population_df[
        population_df["population_group"].isin(["fed", "fasted"])
        & population_df["parameter_name"].isin(ABSORPTION_PARAMETERS)
    ].copy()
    direction_df["modifier_f"] = direction_df["modifier"].astype(float)
    fed_mean = float(direction_df[direction_df["population_group"] == "fed"]["modifier_f"].mean())
    fasted_mean = float(direction_df[direction_df["population_group"] == "fasted"]["modifier_f"].mean())
    _assert(fed_mean < fasted_mean, "Population modifier direction mismatch: fed mean is not lower than fasted.", errors)

    return {
        "simulator_directionality": {
            "fed_peak_bac_percent": fed_peak,
            "fed_time_to_peak_h": fed_tpeak,
            "fasted_peak_bac_percent": fasted_peak,
            "fasted_time_to_peak_h": fasted_tpeak,
            "female_fasted_peak_bac_percent": female_peak,
            "low_dose_fed_peak_bac_percent": low_dose_peak,
        },
        "population_modifier_directionality": {
            "parameters": list(ABSORPTION_PARAMETERS),
            "fed_mean_modifier": fed_mean,
            "fasted_mean_modifier": fasted_mean,
            "graph_reasoning": "fed/fasted groups are linked to absorption parameters in Neo4j via MODIFIES->AFFECTS paths.",
        },
    }


def run_full_stack_terminal_integration(emit: bool = True) -> Dict[str, Any]:
    errors: List[str] = []
    library_df, population_df, beverage_df = _load_pbpk_sources()

    pbpk_outputs = _run_pbpk_checks(library_df, population_df, beverage_df, errors)
    graph_outputs = _run_graph_checks(errors)
    consistency_outputs = _run_pbpk_graph_consistency_checks(
        library_df,
        population_df,
        beverage_df,
        errors,
    )

    safe_for_weaviate_phase = len(errors) == 0
    status = "PASS" if safe_for_weaviate_phase else "FAIL"

    report = {
        "status": status,
        "safe_for_weaviate_phase": safe_for_weaviate_phase,
        "pbpk_outputs": pbpk_outputs,
        "graph_reasoning_outputs": graph_outputs,
        "pbpk_graph_consistency": consistency_outputs,
        "errors": errors,
    }

    if emit:
        print("=== Full Stack Terminal Integration Test ===")
        print(f"STATUS: {status}")
        print(f"safe_for_weaviate_phase: {safe_for_weaviate_phase}")
        print("\n[PBPK Outputs]")
        print(
            "peak_bac_percent={peak:.6f} time_to_peak_h={tpeak:.2f} time_to_sober_h={tsober:.2f}".format(
                peak=pbpk_outputs["peak_bac_percent"],
                tpeak=pbpk_outputs["time_to_peak_h"],
                tsober=pbpk_outputs["time_to_sober_h"],
            )
        )
        print(
            "dose_g={dose:.4f} max_systemic_ethanol_g_from_bac={sysm:.4f}".format(
                dose=pbpk_outputs["dose_g"],
                sysm=pbpk_outputs["max_systemic_ethanol_g_from_bac"],
            )
        )
        print("\n[Graph Reasoning Outputs]")
        enzyme = graph_outputs["explainability_beverage_compound_enzyme"]
        print(f"why_whisky_hits_harder_path_count={enzyme['path_count']}")
        print(f"reasoning_chain={enzyme['reasoning_chain']}")
        tox = graph_outputs["toxicity_hangover"]
        print(f"hangover_path_count={tox['path_count']} risk_types={tox['risk_types']}")
        print("\n[PBPK <-> Neo4j Consistency]")
        sim_dir = consistency_outputs["simulator_directionality"]
        print(
            "fed_peak={fed:.6f} fasted_peak={fasted:.6f} female_fasted_peak={female:.6f}".format(
                fed=sim_dir["fed_peak_bac_percent"],
                fasted=sim_dir["fasted_peak_bac_percent"],
                female=sim_dir["female_fasted_peak_bac_percent"],
            )
        )
        pop_dir = consistency_outputs["population_modifier_directionality"]
        print(
            "fed_mean_modifier={fed:.4f} fasted_mean_modifier={fasted:.4f}".format(
                fed=pop_dir["fed_mean_modifier"],
                fasted=pop_dir["fasted_mean_modifier"],
            )
        )
        print("\n[System Health Summary]")
        if errors:
            for item in errors:
                print(f"FAIL: {item}")
        else:
            print("PASS: All full-stack checks passed.")

    return report


def test_full_stack_terminal_integration() -> None:
    report = run_full_stack_terminal_integration(emit=True)
    assert report["safe_for_weaviate_phase"], "\n".join(report["errors"])


if __name__ == "__main__":
    result = run_full_stack_terminal_integration(emit=True)
    raise SystemExit(0 if result["safe_for_weaviate_phase"] else 1)
