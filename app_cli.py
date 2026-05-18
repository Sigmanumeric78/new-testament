"""Phase 08E/08F interactive CLI demo for the full alcohol-risk pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

try:
    import weaviate  # type: ignore
except Exception:  # pragma: no cover
    weaviate = None

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover
    GraphDatabase = None

from reasoning.grounding_safety_guard import GroundingSafetyGuard
from reasoning.hybrid_orchestrator import orchestrate_query
from reasoning.query_router import route_query
from reasoning.response_synthesizer import OLLAMA_MODEL, ResponseSynthesizer
from reasoning.user_risk_advisor import build_user_risk_advice
from simulation.pbpk import pbpk_master_simulator
from utils.config import get_neo4j_config, get_weaviate_config

LOG_PATH = Path("data/interim/reasoning/app_cli_run_log.jsonl")

DEMO_QUERIES: Tuple[str, ...] = (
    "Why does whisky hit harder than beer?",
    "How drunk will I get after 180ml whisky?",
    "Why does wine give me headaches?",
    "Show research on sulfites",
    "I am 60kg female and fasted, how drunk will I get after 180ml whisky?",
)

INTAKE_GOALS: Tuple[str, ...] = (
    "drive check",
    "time to sober",
    "hangover risk",
    "should I keep drinking",
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"none", "null", "nan"}:
        return ""
    return text


def _json_print(payload: Mapping[str, Any], *, compact: bool = False) -> None:
    if compact:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def _is_ollama_reachable(timeout_seconds: int = 6) -> Tuple[bool, str]:
    if shutil.which("ollama") is None:
        return False, "ollama executable not found"
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)

    if completed.returncode != 0:
        return False, _clean_text(completed.stderr) or "ollama list failed"
    return True, "ok"


def _is_neo4j_reachable() -> Tuple[bool, str]:
    if GraphDatabase is None:
        return False, "neo4j driver not installed"

    try:
        config = get_neo4j_config()
    except Exception as exc:
        return False, str(exc)

    driver = None
    try:
        driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))
        with driver.session(database=config["database"]) as session:
            record = session.run("RETURN 1 AS ok").single()
            if record is None or int(record["ok"]) != 1:
                return False, "unexpected Neo4j probe response"
    except Exception as exc:
        return False, str(exc)
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass

    return True, "ok"


def _parse_weaviate_url(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid WEAVIATE_URL: '{url}'. Expected http(s)://host[:port]")
    secure = parsed.scheme.lower() == "https"
    return {
        "host": parsed.hostname,
        "port": int(parsed.port or (443 if secure else 80)),
        "secure": secure,
    }


def _connect_weaviate(config: Mapping[str, str]) -> Any:
    if weaviate is None:
        raise RuntimeError("weaviate-client is not installed")

    url_info = _parse_weaviate_url(config["url"])
    grpc_host = _clean_text(config.get("grpc_host", "")) or "localhost"
    grpc_port = int(_clean_text(config.get("grpc_port", "")) or "50051")
    api_key = _clean_text(config.get("api_key", ""))

    auth_credentials = None
    if api_key:
        try:
            from weaviate.classes.init import Auth  # type: ignore

            auth_credentials = Auth.api_key(api_key)
        except Exception:
            from weaviate.auth import AuthApiKey  # type: ignore

            auth_credentials = AuthApiKey(api_key)

    try:
        return weaviate.connect_to_custom(
            http_host=url_info["host"],
            http_port=url_info["port"],
            http_secure=url_info["secure"],
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=url_info["secure"],
            auth_credentials=auth_credentials,
        )
    except Exception:
        return weaviate.connect_to_local(
            host=url_info["host"],
            port=url_info["port"],
            grpc_port=grpc_port,
            auth_credentials=auth_credentials,
        )


def _is_weaviate_reachable() -> Tuple[bool, str]:
    if weaviate is None:
        return False, "weaviate-client not installed"

    try:
        config = get_weaviate_config()
    except Exception as exc:
        return False, str(exc)

    client = None
    try:
        client = _connect_weaviate(config)
        ready = bool(client.is_ready())
        if not ready:
            return False, "is_ready() returned False"
    except Exception as exc:
        return False, str(exc)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    return True, "ok"


def run_health_check() -> Dict[str, Any]:
    components: Dict[str, Dict[str, Any]] = {}

    neo4j_ok, neo4j_detail = _is_neo4j_reachable()
    components["neo4j_reachable"] = {"ok": neo4j_ok, "detail": neo4j_detail}

    weaviate_ok, weaviate_detail = _is_weaviate_reachable()
    components["weaviate_reachable"] = {"ok": weaviate_ok, "detail": weaviate_detail}

    ollama_ok, ollama_detail = _is_ollama_reachable()
    components["ollama_reachable"] = {"ok": ollama_ok, "detail": ollama_detail}

    try:
        _ = pbpk_master_simulator.run_simulation
        components["pbpk_importable"] = {"ok": True, "detail": "ok"}
    except Exception as exc:
        components["pbpk_importable"] = {"ok": False, "detail": str(exc)}

    try:
        _ = route_query
        components["router_available"] = {"ok": True, "detail": "ok"}
    except Exception as exc:
        components["router_available"] = {"ok": False, "detail": str(exc)}

    try:
        _ = orchestrate_query
        components["orchestrator_available"] = {"ok": True, "detail": "ok"}
    except Exception as exc:
        components["orchestrator_available"] = {"ok": False, "detail": str(exc)}

    try:
        _ = ResponseSynthesizer
        components["synthesizer_available"] = {"ok": True, "detail": "ok"}
    except Exception as exc:
        components["synthesizer_available"] = {"ok": False, "detail": str(exc)}

    try:
        _ = GroundingSafetyGuard
        components["guard_available"] = {"ok": True, "detail": "ok"}
    except Exception as exc:
        components["guard_available"] = {"ok": False, "detail": str(exc)}

    overall_ok = all(bool(payload.get("ok")) for payload in components.values())

    return {
        "status": "ok" if overall_ok else "degraded",
        "health_check_pass": bool(overall_ok),
        "components": components,
    }


def _log_run(
    *,
    query: str,
    intent: str,
    modules: Sequence[str],
    approved_for_display: bool,
    confidence_score: float,
    blocked_reasons: Sequence[str],
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": _clean_text(query),
        "intent": _clean_text(intent),
        "modules": [_clean_text(item) for item in modules if _clean_text(item)],
        "approved_for_display": bool(approved_for_display),
        "confidence_score": round(float(confidence_score), 6),
        "blocked_reasons": [_clean_text(item) for item in blocked_reasons if _clean_text(item)],
    }
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_pipeline(
    query: str,
    *,
    model: str = OLLAMA_MODEL,
    timeout_seconds: int = 30,
    enable_router_llm_fallback: bool = False,
) -> Dict[str, Any]:
    text = _clean_text(query)
    if not text:
        raise ValueError("Query must be non-empty.")

    orchestration = orchestrate_query(text, enable_llm_fallback=bool(enable_router_llm_fallback))

    synthesizer = ResponseSynthesizer(model=model, timeout_seconds=timeout_seconds)
    synthesized = synthesizer.synthesize_response(orchestration)

    guard = GroundingSafetyGuard()
    guarded = guard.validate(synthesized)

    intent = _clean_text(orchestration.get("route", {}).get("intent"))
    modules = list(orchestration.get("route", {}).get("required_modules", []) or [])

    user_advice = build_user_risk_advice(
        query=text,
        guarded_payload=guarded,
        synthesized_payload=synthesized,
        orchestrator_payload=orchestration,
    )

    guard_approved = bool(guarded.get("approved_for_display"))
    unsafe_blocked = bool(guarded.get("unsafe_claims_detected"))

    # Preserve hard safety blocks from the guard. For non-unsafe grounding misses,
    # serve conservative risk-advisor guidance.
    final_approved = bool((guard_approved and user_advice.get("safe_for_display")) or (not unsafe_blocked and user_advice.get("safe_for_display")))

    result = {
        "query": text,
        "approved_for_display": final_approved,
        "guard_approved_for_display": guard_approved,
        "unsafe_claims_detected": unsafe_blocked,
        "unsupported_claims_detected": bool(guarded.get("unsupported_claims_detected")),
        "blocked_reasons": list(guarded.get("blocked_reasons", []) or []),
        "warnings": list(guarded.get("warnings", []) or []),
        "required_edits": list(guarded.get("required_edits", []) or []),
        "grounding_score": guarded.get("grounding_score"),
        "safety_score": guarded.get("safety_score"),
        "intent": intent,
        "modules_used": modules,
        "confidence_score": round(float(synthesized.get("confidence_score") or 0.0), 6),
        "user_risk_advice": user_advice,
        "guard_output": guarded,
        "synthesized_output": synthesized,
    }

    _log_run(
        query=text,
        intent=intent,
        modules=modules,
        approved_for_display=bool(result["approved_for_display"]),
        confidence_score=float(result["confidence_score"]),
        blocked_reasons=list(result.get("blocked_reasons", []) or []),
    )

    return result


def format_pretty_output(payload: Mapping[str, Any], *, debug: bool = False) -> str:
    query = _clean_text(payload.get("query"))
    approved = bool(payload.get("approved_for_display"))

    advice = payload.get("user_risk_advice", {})
    if not isinstance(advice, Mapping):
        advice = {}

    lines: List[str] = []
    lines.append(f"Query: {query}")

    if not approved:
        lines.append("Answer: The system blocked this response for safety reasons.")
        reasons = [
            _clean_text(item)
            for item in list(payload.get("blocked_reasons", []) or [])
            if _clean_text(item)
        ]
        if reasons:
            lines.append(f"Blocked reasons: {'; '.join(reasons)}")
    else:
        lines.append(f"Answer: {_clean_text(advice.get('plain_answer'))}")

    lines.append(f"Risk level: {_clean_text(advice.get('risk_level')) or 'unknown'}")
    peak = advice.get("estimated_peak_bac")
    if peak is None:
        peak_display = "unknown"
    else:
        peak_display = f"about {float(peak):.2f}%"
    lines.append(f"Estimated peak BAC: {peak_display}")
    t_clear = advice.get("estimated_time_to_sober_h")
    if t_clear is None:
        clear_display = "unknown"
    else:
        clear_display = f"about {max(int(round(float(t_clear))), 1)} hours"
    lines.append(f"Estimated time until alcohol clears: {clear_display}")
    lines.append(f"Driving guidance: {_clean_text(advice.get('driving_guidance'))}")
    lines.append(f"Continue drinking guidance: {_clean_text(advice.get('continue_drinking_guidance'))}")
    lines.append(f"Hydration: {_clean_text(advice.get('hydration_guidance'))}")
    lines.append(f"Food: {_clean_text(advice.get('food_guidance'))}")
    lines.append(f"Medical warning: {_clean_text(advice.get('medical_warning'))}")

    assumptions = list(advice.get("assumptions", []) or [])
    assumption_text = "; ".join([_clean_text(item) for item in assumptions if _clean_text(item)])
    lines.append(f"Assumptions: {assumption_text or 'none'}")

    if debug:
        lines.append("")
        lines.append("[Debug]")
        lines.append(f"Intent: {_clean_text(payload.get('intent')) or 'unknown'}")
        modules = [
            _clean_text(item)
            for item in list(payload.get("modules_used", []) or [])
            if _clean_text(item)
        ]
        lines.append(f"Modules used: {', '.join(modules) if modules else 'unknown'}")
        lines.append(f"Guard approved: {str(bool(payload.get('guard_approved_for_display'))).lower()}")
        lines.append(f"Confidence: {payload.get('confidence_score')}")
        lines.append(f"Grounding score: {payload.get('grounding_score')}")
        lines.append(f"Safety score: {payload.get('safety_score')}")

    return "\n".join(lines)


def run_demo(
    *,
    model: str = OLLAMA_MODEL,
    timeout_seconds: int = 30,
    enable_router_llm_fallback: bool = False,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for query in DEMO_QUERIES:
        outputs.append(
            run_pipeline(
                query,
                model=model,
                timeout_seconds=timeout_seconds,
                enable_router_llm_fallback=enable_router_llm_fallback,
            )
        )
    return outputs


def build_query_from_intake(intake_payload: Mapping[str, str]) -> str:
    sex = _clean_text(intake_payload.get("sex"))
    weight = _clean_text(intake_payload.get("weight"))
    age = _clean_text(intake_payload.get("age"))
    fed_state = _clean_text(intake_payload.get("fed_state"))
    drink_type = _clean_text(intake_payload.get("drink_type"))
    amount = _clean_text(intake_payload.get("amount"))
    time_period = _clean_text(intake_payload.get("time_period"))
    goal = _clean_text(intake_payload.get("goal")).lower()

    profile_bits = []
    if sex:
        profile_bits.append(sex)
    if weight:
        profile_bits.append(f"{weight} kg")
    if age:
        profile_bits.append(f"{age} years old")
    if fed_state:
        profile_bits.append(fed_state)

    drinking_bits = []
    if amount and drink_type:
        drinking_bits.append(f"{amount} {drink_type}")
    elif amount:
        drinking_bits.append(amount)
    elif drink_type:
        drinking_bits.append(drink_type)
    if time_period:
        drinking_bits.append(f"over {time_period}")

    if goal == "drive check":
        goal_question = "Can I drive now?"
    elif goal == "time to sober":
        goal_question = "How long until I am sober?"
    elif goal == "hangover risk":
        goal_question = "What is my hangover risk?"
    elif goal == "should i keep drinking":
        goal_question = "How much more can I drink?"
    else:
        goal_question = "What is my current alcohol risk?"

    profile_text = " ".join(profile_bits).strip()
    drinking_text = " ".join(drinking_bits).strip()

    query_parts = ["I am"]
    if profile_text:
        query_parts.append(profile_text)
    if drinking_text:
        query_parts.append(f"and I drank {drinking_text}.")
    else:
        query_parts.append(".")
    query_parts.append(goal_question)
    return " ".join(query_parts).replace(" .", ".").strip()


def run_intake_mode(
    *,
    input_fn: Callable[[str], str] = input,
) -> Dict[str, str]:
    prompts = {
        "sex": "Sex (male/female): ",
        "weight": "Weight in kg (e.g., 75): ",
        "age": "Age (optional): ",
        "fed_state": "Fed or fasted: ",
        "drink_type": "Drink type: ",
        "amount": "Amount (e.g., 180ml, 2 beers): ",
        "time_period": "Time period (e.g., 1 hour, 30 minutes): ",
        "goal": "Goal (drive check/time to sober/hangover risk/should I keep drinking): ",
    }

    payload: Dict[str, str] = {}
    for key in ("sex", "weight", "age", "fed_state", "drink_type", "amount", "time_period", "goal"):
        payload[key] = _clean_text(input_fn(prompts[key]))

    if _clean_text(payload.get("goal")).lower() not in INTAKE_GOALS:
        payload["goal"] = "time to sober"

    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alcohol Intelligence CLI demo")
    parser.add_argument("--query", type=str, default="", help="Single query mode")
    parser.add_argument("--json", action="store_true", help="JSON output mode")
    parser.add_argument("--pretty", action="store_true", help="Human-readable output mode")
    parser.add_argument("--debug", action="store_true", help="Show internal fields in pretty mode")
    parser.add_argument("--health", action="store_true", help="Run health checks")
    parser.add_argument("--demo", action="store_true", help="Run deterministic demo queries")
    parser.add_argument("--intake", action="store_true", help="Run guided intake prompts")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Local Ollama model for synthesis")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="Model timeout seconds")
    parser.add_argument(
        "--enable-router-llm-fallback",
        action="store_true",
        help="Enable optional router fallback in query routing.",
    )
    return parser.parse_args(argv)


def _resolve_output_mode(args: argparse.Namespace) -> str:
    if bool(args.json) and bool(args.pretty):
        raise ValueError("Use either --json or --pretty, not both.")

    if bool(args.json):
        return "json"
    if bool(args.pretty):
        return "pretty"

    if _clean_text(args.query):
        return "json"
    return "pretty"


def _print_help_banner() -> None:
    print("Type a question, or one of: help, exit, quit")


def _run_interactive(
    *,
    model: str,
    timeout_seconds: int,
    enable_router_llm_fallback: bool,
    debug: bool,
) -> int:
    _print_help_banner()

    while True:
        try:
            raw = input("Alcohol Intelligence > ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        text = _clean_text(raw)
        lowered = text.lower()

        if not text:
            continue

        if lowered in {"exit", "quit"}:
            return 0

        if lowered == "help":
            _print_help_banner()
            continue

        payload = run_pipeline(
            text,
            model=model,
            timeout_seconds=timeout_seconds,
            enable_router_llm_fallback=enable_router_llm_fallback,
        )

        print(format_pretty_output(payload, debug=debug))
        print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    output_mode = _resolve_output_mode(args)
    model = _clean_text(args.model) or OLLAMA_MODEL
    timeout_seconds = int(args.timeout_seconds)

    if bool(args.health):
        health = run_health_check()
        _json_print(health, compact=(output_mode == "json"))
        return 0

    if bool(args.demo):
        results = run_demo(
            model=model,
            timeout_seconds=timeout_seconds,
            enable_router_llm_fallback=bool(args.enable_router_llm_fallback),
        )
        if output_mode == "json":
            _json_print({"demo_results": results}, compact=True)
        else:
            for result in results:
                print(format_pretty_output(result, debug=bool(args.debug)))
                print()
        return 0

    if bool(args.intake):
        intake_payload = run_intake_mode()
        query = build_query_from_intake(intake_payload)
        result = run_pipeline(
            query,
            model=model,
            timeout_seconds=timeout_seconds,
            enable_router_llm_fallback=bool(args.enable_router_llm_fallback),
        )
        if output_mode == "json":
            payload = {"intake": intake_payload, "query": query, "result": result}
            _json_print(payload, compact=True)
        else:
            print(format_pretty_output(result, debug=bool(args.debug)))
        return 0

    query = _clean_text(args.query)
    if query:
        result = run_pipeline(
            query,
            model=model,
            timeout_seconds=timeout_seconds,
            enable_router_llm_fallback=bool(args.enable_router_llm_fallback),
        )
        if output_mode == "json":
            _json_print(result, compact=True)
        else:
            print(format_pretty_output(result, debug=bool(args.debug)))
        return 0

    return _run_interactive(
        model=model,
        timeout_seconds=timeout_seconds,
        enable_router_llm_fallback=bool(args.enable_router_llm_fallback),
        debug=bool(args.debug),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
