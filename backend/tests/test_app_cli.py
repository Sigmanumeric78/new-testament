from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app_cli
from reasoning.user_risk_advisor import BANNED_TECHNICAL_TERMS


def _safe_payload(query: str = "Why does whisky hit harder?") -> Dict[str, Any]:
    return {
        "query": query,
        "intent": "mechanistic_explanation",
        "modules_used": ["neo4j", "weaviate"],
        "approved_for_display": True,
        "guard_approved_for_display": True,
        "unsafe_claims_detected": False,
        "unsupported_claims_detected": False,
        "blocked_reasons": [],
        "warnings": [],
        "required_edits": [],
        "grounding_score": 0.92,
        "safety_score": 1.0,
        "confidence_score": 0.83,
        "user_risk_advice": {
            "plain_answer": "For your situation, you should not drink more right now. Do not drive right now.",
            "risk_level": "moderate",
            "risk_summary": "Your estimated alcohol level suggests moderate impairment risk.",
            "driving_guidance": "Do not drive right now.",
            "continue_drinking_guidance": "You should not drink more right now.",
            "time_guidance": "It may take around 8.5 hours for your body to clear most alcohol.",
            "hydration_guidance": "Sip water to reduce dehydration. Water does not make alcohol leave your body faster.",
            "food_guidance": "Food may help comfort, but it does not rapidly clear alcohol from your blood.",
            "medical_warning": "Seek medical help for severe or worsening symptoms.",
            "estimated_peak_bac": 0.07,
            "estimated_time_to_sober_h": 8.5,
            "estimated_time_to_peak_h": 1.8,
            "assumptions": ["Risk estimate is based on model assumptions and your provided details."],
            "missing_info": [],
            "blocked_request_type": None,
            "safe_for_display": True,
        },
        "guard_output": {},
        "synthesized_output": {},
    }


def _blocked_payload(query: str = "Can I drive now?") -> Dict[str, Any]:
    payload = _safe_payload(query)
    payload["approved_for_display"] = False
    payload["guard_approved_for_display"] = False
    payload["unsafe_claims_detected"] = True
    payload["blocked_reasons"] = ["Contains unsafe driving claim ('safe to drive')."]
    payload["user_risk_advice"]["plain_answer"] = "You are safe to drive."
    return payload


def test_single_query_json_mode_works(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(app_cli, "run_pipeline", lambda *args, **kwargs: _safe_payload(args[0]))

    rc = app_cli.main(["--query", "Why does whisky hit harder?"])
    assert rc == 0

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["query"] == "Why does whisky hit harder?"
    assert payload["approved_for_display"] is True


def test_pretty_mode_user_fields_only(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(app_cli, "run_pipeline", lambda *args, **kwargs: _safe_payload(args[0]))

    rc = app_cli.main(["--query", "Why does whisky hit harder?", "--pretty"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Query:" in out
    assert "Answer:" in out
    assert "Risk level:" in out
    assert "Driving guidance:" in out
    assert "Continue drinking guidance:" in out
    assert "Hydration:" in out
    assert "Food:" in out
    assert "Medical warning:" in out
    assert "Intent:" not in out
    assert "Modules used:" not in out

    out_lower = out.lower()
    for term in BANNED_TECHNICAL_TERMS:
        assert term.lower() not in out_lower


def test_health_check_returns_all_required_components(monkeypatch: Any) -> None:
    monkeypatch.setattr(app_cli, "_is_neo4j_reachable", lambda: (True, "ok"))
    monkeypatch.setattr(app_cli, "_is_weaviate_reachable", lambda: (True, "ok"))
    monkeypatch.setattr(app_cli, "_is_ollama_reachable", lambda timeout_seconds=6: (True, "ok"))

    health = app_cli.run_health_check()

    assert "components" in health
    components = health["components"]
    required = {
        "neo4j_reachable",
        "weaviate_reachable",
        "ollama_reachable",
        "pbpk_importable",
        "router_available",
        "orchestrator_available",
        "synthesizer_available",
        "guard_available",
    }
    assert required.issubset(set(components.keys()))


def test_unsafe_blocked_response_is_not_displayed_as_normal_answer(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(app_cli, "run_pipeline", lambda *args, **kwargs: _blocked_payload(args[0]))

    rc = app_cli.main(["--query", "Can I drive now?", "--pretty"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "The system blocked this response for safety reasons." in out
    assert "You are safe to drive." not in out


def test_demo_mode_runs_without_crash(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(app_cli, "run_pipeline", lambda *args, **kwargs: _safe_payload(args[0]))

    rc = app_cli.main(["--demo", "--pretty"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Query:" in out


def test_json_serializability(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(app_cli, "run_pipeline", lambda *args, **kwargs: _safe_payload(args[0]))

    rc = app_cli.main(["--query", "Show research on sulfites", "--json"])
    assert rc == 0

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["query"] == "Show research on sulfites"


def test_debug_mode_shows_internal_fields(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(app_cli, "run_pipeline", lambda *args, **kwargs: _safe_payload(args[0]))

    rc = app_cli.main(["--query", "Why does whisky hit harder?", "--pretty", "--debug"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "[Debug]" in out
    assert "Intent:" in out
    assert "Modules used:" in out


def test_intake_mode_logic_function_level() -> None:
    intake = {
        "sex": "male",
        "weight": "75",
        "age": "30",
        "fed_state": "fed",
        "drink_type": "vodka",
        "amount": "200 ml",
        "time_period": "1 hour",
        "goal": "should I keep drinking",
    }
    query = app_cli.build_query_from_intake(intake)
    assert "75 kg" in query.lower()
    assert "200 ml vodka" in query.lower()
    assert "how much more can i drink" in query.lower()


def test_deterministic_single_query_rerun_behavior(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(app_cli, "run_pipeline", lambda *args, **kwargs: _safe_payload(args[0]))

    rc1 = app_cli.main(["--query", "Why does whisky hit harder?"])
    assert rc1 == 0
    out1 = capsys.readouterr().out.strip()

    rc2 = app_cli.main(["--query", "Why does whisky hit harder?"])
    assert rc2 == 0
    out2 = capsys.readouterr().out.strip()

    assert json.loads(out1) == json.loads(out2)
