"""Generate a personalized PBPK user profile from CLI inputs."""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PROFILE = os.path.join(
    BASE_DIR, "data", "processed", "current_user_profile.json"
)

# 5.4 L/min total cardiac output at 70 kg -> 324 L/h total.
CARDIAC_OUTPUT_L_H_KG = 324.0 / 70.0

tissue_volume_fractions_bw = {
    "blood": 0.059,
    "liver": 0.0314,
    "kidneys": 0.0044,
    "gi_tract": 0.034,
    "fat": 0.231,
}

blood_flow_fractions_cardiac_output = {
    "liver": 0.25,
    "gi_tract": 0.21,
    "kidneys": 0.25,
    "fat": 0.05,
}

DEFAULT_PORTAL_FRACTION_OF_LIVER_FLOW = 0.75
BASELINE_ALBUMIN_G_L = 45.0
BASELINE_GFR_L_H = 7.2  # 120 mL/min


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate personalized PBPK physiology profile."
    )
    parser.add_argument("--weight", type=float, help="Weight in kg")
    parser.add_argument("--height", type=float, help="Height in cm")
    parser.add_argument("--sex", type=str, choices=["m", "f"], help="Sex: m or f")
    parser.add_argument("--disease", type=str, default="healthy")
    return parser.parse_args()


def prompt_float(label: str, default: Optional[float] = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{label}{suffix}: ").strip()
        if not raw and default is not None:
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a numeric value.")
            continue
        if value <= 0:
            print("Value must be positive.")
            continue
        return value


def prompt_choice(label: str, allowed: list[str], default: Optional[str] = None) -> str:
    allowed_set = {a.lower() for a in allowed}
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{label}{suffix}: ").strip().lower()
        if not raw and default is not None:
            raw = default.lower()
        if raw in allowed_set:
            return raw
        print(f"Please enter one of: {', '.join(allowed)}.")


def prompt_text(label: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{label}{suffix}: ").strip()
    if not raw and default is not None:
        return default
    return raw


def build_profile(
    weight_kg: float,
    height_cm: float,
    sex_code: str,
    disease: str,
) -> dict:
    sex = "male" if sex_code == "m" else "female"
    bmi = weight_kg / ((height_cm / 100.0) ** 2)
    bsa_m2 = 0.007184 * (height_cm**0.725) * (weight_kg**0.425)

    absolute_cardiac_output_L_h = CARDIAC_OUTPUT_L_H_KG * weight_kg

    absolute_volumes_L = {
        tissue: fraction * weight_kg
        for tissue, fraction in tissue_volume_fractions_bw.items()
    }

    absolute_flows_L_h = {
        tissue: fraction * absolute_cardiac_output_L_h
        for tissue, fraction in blood_flow_fractions_cardiac_output.items()
    }

    portal_flow_L_h = DEFAULT_PORTAL_FRACTION_OF_LIVER_FLOW * absolute_flows_L_h["liver"]
    albumin_g_L = BASELINE_ALBUMIN_G_L
    renal_flow_L_h = absolute_flows_L_h["kidneys"]
    gfr_L_h = BASELINE_GFR_L_H

    disease_scalars = {
        "portal_blood_flow": 1.0,
        "albumin": 1.0,
        "renal_blood_flow": 1.0,
        "gfr": 1.0,
    }
    if disease == "hepatic_severe":
        disease_scalars = {
            "portal_blood_flow": 0.13,
            "albumin": 0.53,
            "renal_blood_flow": 0.48,
            "gfr": 0.55,
        }

    disease_adjusted = {
        "portal_blood_flow_L_h": portal_flow_L_h * disease_scalars["portal_blood_flow"],
        "albumin_g_L": albumin_g_L * disease_scalars["albumin"],
        "renal_blood_flow_L_h": renal_flow_L_h * disease_scalars["renal_blood_flow"],
        "gfr_L_h": gfr_L_h * disease_scalars["gfr"],
    }

    profile = {
        "source": "EPA Exposure Factors Handbook / ICRP Publication 89",
        "sex": sex,
        "disease": disease,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bmi": bmi,
        "bsa_m2": bsa_m2,
        "cardiac_output_L_h_kg": CARDIAC_OUTPUT_L_H_KG,
        "absolute_cardiac_output_L_h": absolute_cardiac_output_L_h,
        "tissue_volume_fractions_bw": tissue_volume_fractions_bw,
        "absolute_volumes_L": absolute_volumes_L,
        "blood_flow_fractions_cardiac_output": blood_flow_fractions_cardiac_output,
        "absolute_flows_L_h": absolute_flows_L_h,
        "clinical_baseline": {
            "portal_blood_flow_L_h": portal_flow_L_h,
            "albumin_g_L": albumin_g_L,
            "renal_blood_flow_L_h": renal_flow_L_h,
            "gfr_L_h": gfr_L_h,
        },
        "disease_scalars_applied": disease_scalars,
        "disease_adjusted_parameters": disease_adjusted,
    }
    return profile


def save_profile(profile: dict) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PROFILE), exist_ok=True)
    with open(OUTPUT_PROFILE, "w") as f:
        json.dump(profile, f, indent=4)


def print_profile_summary(profile: dict) -> None:
    absolute_volumes_L = profile["absolute_volumes_L"]
    absolute_flows_L_h = profile["absolute_flows_L_h"]
    disease_adjusted = profile["disease_adjusted_parameters"]

    print("PBPK Profile Generated")
    print(f"Saved: {OUTPUT_PROFILE}")
    print(
        f"User -> weight: {profile['weight_kg']:.2f} kg | "
        f"height: {profile['height_cm']:.2f} cm | "
        f"sex: {profile['sex']} | disease: {profile['disease']}"
    )
    print(f"BMI: {profile['bmi']:.4f}")
    print(f"BSA: {profile['bsa_m2']:.4f} m^2")
    print(f"Liver volume: {absolute_volumes_L['liver']:.4f} L")
    print(f"Liver flow: {absolute_flows_L_h['liver']:.4f} L/h")
    print("Disease-adjusted:")
    print(f"  Portal flow: {disease_adjusted['portal_blood_flow_L_h']:.4f} L/h")
    print(f"  Albumin: {disease_adjusted['albumin_g_L']:.4f} g/L")
    print(f"  Renal flow: {disease_adjusted['renal_blood_flow_L_h']:.4f} L/h")
    print(f"  GFR: {disease_adjusted['gfr_L_h']:.4f} L/h")


def generate_profile_interactive(args: Optional[argparse.Namespace] = None) -> dict:
    if args is None:
        args = argparse.Namespace(weight=None, height=None, sex=None, disease="healthy")

    print("Enter user profile values:")
    weight_kg = prompt_float("Weight (kg)", args.weight)
    height_cm = prompt_float("Height (cm)", args.height)
    sex_code = prompt_choice("Sex (m/f)", ["m", "f"], args.sex)
    disease = prompt_text("Disease", args.disease).strip().lower()
    if not disease:
        disease = "healthy"

    profile = build_profile(weight_kg, height_cm, sex_code, disease)
    save_profile(profile)
    print_profile_summary(profile)
    return profile


def main() -> None:
    args = parse_args()
    generate_profile_interactive(args)


if __name__ == "__main__":
    main()
