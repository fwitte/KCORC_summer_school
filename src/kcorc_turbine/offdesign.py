from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .correlations import (
    disk_friction_loss_W,
    incidence_corrected_velocity_coefficient,
    partial_admission_losses_W,
    rotor_velocity_coefficient,
    sector_filling_coefficient,
    stator_velocity_coefficient,
)
from .properties import CoolPropBackend, PropertyBackend


@dataclass(frozen=True)
class FixedGeometry:
    fluid: str
    p_in_design_Pa: float
    T_in_design_K: float
    p_out_design_Pa: float
    pressure_ratio_design: float
    n_design_rpm: float
    m_dot_design_kg_s: float
    eta_design: float
    P_mech_design_W: float
    D_mid_m: float
    alpha_stator_deg: float
    beta_rotor_in_deg: float
    beta_rotor_out_deg: float
    partial_admission_eff: float
    rotor_chord_m: float
    rotor_pitch_m: float
    h_rotor_m: float
    A_throat_m2: float
    A_outlet_m2: float
    superheat_design_K: float | None

    @classmethod
    def from_design(cls, result, backend: PropertyBackend | None = None):
        backend = backend or CoolPropBackend()
        superheat: float | None
        try:
            T_dew = backend.dew_temperature(result.inputs.p_in_Pa, result.inputs.fluid)
            superheat = result.inputs.T_in_K - T_dew
        except Exception:
            superheat = None
        return cls(
            fluid=result.inputs.fluid,
            p_in_design_Pa=result.inputs.p_in_Pa,
            T_in_design_K=result.inputs.T_in_K,
            p_out_design_Pa=result.inputs.p_out_Pa,
            pressure_ratio_design=float(result.performance["pressure_ratio"]),
            n_design_rpm=result.inputs.n_rpm,
            m_dot_design_kg_s=result.inputs.m_dot_kg_s,
            eta_design=float(result.performance["eta_turb"]),
            P_mech_design_W=float(result.performance["P_mech_W"]),
            D_mid_m=result.inputs.D_mid_m,
            alpha_stator_deg=result.inputs.alpha_stator_deg,
            beta_rotor_in_deg=result.inputs.beta_rotor_in_deg,
            beta_rotor_out_deg=result.inputs.beta_rotor_out_deg,
            partial_admission_eff=float(result.geometry["partial_admission_eff"]),
            rotor_chord_m=result.inputs.rotor_chord_m,
            rotor_pitch_m=float(result.geometry["rotor_pitch_m"]),
            h_rotor_m=float(result.geometry["h_rotor_m"]),
            A_throat_m2=float(result.geometry["A_throat_m2"]),
            A_outlet_m2=float(result.geometry["A_outlet_m2"]),
            superheat_design_K=superheat,
        )


def inlet_temperature_for_pressure(
    geometry: FixedGeometry,
    p_in_Pa: float,
    backend: PropertyBackend | None = None,
) -> float:
    """Preserve the design superheat where a dew state is available."""
    backend = backend or CoolPropBackend()
    if geometry.superheat_design_K is None:
        return geometry.T_in_design_K
    try:
        return backend.dew_temperature(p_in_Pa, geometry.fluid) + geometry.superheat_design_K
    except Exception:
        return geometry.T_in_design_K


def evaluate_stator_offdesign(
    geometry: FixedGeometry,
    p_in_Pa: float,
    T_in_K: float,
    p_out_Pa: float,
    backend: PropertyBackend | None = None,
    n_steps: int = 110,
) -> dict[str, float | bool]:
    backend = backend or CoolPropBackend()
    state_in = backend.state_PT(p_in_Pa, T_in_K, geometry.fluid)
    p_values = np.linspace(p_in_Pa, p_out_Pa, int(n_steps))
    states = [backend.state_PS(float(p), state_in["s_J_kgK"], geometry.fluid) for p in p_values]
    h_is = np.array([s["h_J_kg"] for s in states])
    rho_is = np.array([s["rho_kg_m3"] for s in states])
    a_is = np.array([s["a_m_s"] for s in states])
    c_is = np.sqrt(np.maximum(2.0 * (state_in["h_J_kg"] - h_is), 0.0))
    Ma_is = c_is / a_is
    crossings = np.flatnonzero(Ma_is >= 1.0)
    choked = bool(crossings.size)
    throat_index = int(crossings[0] if choked else np.argmax(Ma_is))
    phi_s = stator_velocity_coefficient(float(Ma_is[-1]))

    if choked:
        mdot = geometry.A_throat_m2 * rho_is[throat_index] * c_is[throat_index]
    else:
        mdot = np.nan  # filled after the actual outlet state is known

    h_actual_out = state_in["h_J_kg"] - phi_s**2 * (state_in["h_J_kg"] - h_is[-1])
    state_out = backend.state_PH(p_out_Pa, float(h_actual_out), geometry.fluid)
    c_out = float(np.sqrt(max(2.0 * (state_in["h_J_kg"] - h_actual_out), 0.0)))
    if not choked:
        mdot = geometry.A_outlet_m2 * state_out["rho_kg_m3"] * c_out
    A_required = mdot / max(state_out["rho_kg_m3"] * c_out, 1e-12)

    alpha = np.radians(geometry.alpha_stator_deg)
    return {
        "valid_stator": bool(np.isfinite(mdot) and mdot > 0.0 and c_out > 0.0),
        "choked": choked,
        "p_in_Pa": float(p_in_Pa),
        "T_in_K": float(T_in_K),
        "p_out_Pa": float(p_out_Pa),
        "pressure_ratio": float(p_in_Pa / p_out_Pa),
        "h_in_J_kg": float(state_in["h_J_kg"]),
        "s_in_J_kgK": float(state_in["s_J_kgK"]),
        "h_out_is_J_kg": float(h_is[-1]),
        "dh_is_J_kg": float(state_in["h_J_kg"] - h_is[-1]),
        "m_dot_kg_s": float(mdot),
        "p_throat_Pa": float(p_values[throat_index]),
        "Ma_out_is": float(Ma_is[-1]),
        "phi_stator": float(phi_s),
        "h_nozzle_out_J_kg": float(h_actual_out),
        "T_nozzle_out_K": float(state_out["T_K"]),
        "rho_nozzle_out_kg_m3": float(state_out["rho_kg_m3"]),
        "a_nozzle_out_m_s": float(state_out["a_m_s"]),
        "c_nozzle_out_m_s": c_out,
        "Ma_nozzle_out": float(c_out / state_out["a_m_s"]),
        "c1a_m_s": float(c_out * np.sin(alpha)),
        "c1u_m_s": float(c_out * np.cos(alpha)),
        "A_required_outlet_m2": float(A_required),
        "area_mismatch_rel": float(A_required / geometry.A_outlet_m2 - 1.0),
    }


def evaluate_rotor_offdesign(
    geometry: FixedGeometry,
    stator: dict[str, float | bool],
    n_rpm: float,
    backend: PropertyBackend | None = None,
) -> dict[str, float | bool]:
    backend = backend or CoolPropBackend()
    if not bool(stator["valid_stator"]):
        return {"valid": False, "n_rpm": float(n_rpm)}

    U = np.pi * geometry.D_mid_m * n_rpm / 60.0
    c1a = float(stator["c1a_m_s"])
    c1u = float(stator["c1u_m_s"])
    w1a = c1a
    w1u = c1u - U
    w1 = float(np.hypot(w1a, w1u))

    beta1_flow_deg = float(np.degrees(np.arctan2(w1a, w1u)))
    incidence_deg = beta1_flow_deg - geometry.beta_rotor_in_deg

    Ma1r = w1 / float(stator["a_nozzle_out_m_s"])

    # Keep the original design-point correlation tied to geometric blade
    # turning, then apply inlet incidence as a separate off-design penalty.
    beta2_geometry = 180.0 - geometry.beta_rotor_out_deg
    theta = beta2_geometry - geometry.beta_rotor_in_deg
    _, phi_r_zero_incidence = rotor_velocity_coefficient(
        theta,
        Ma1r,
        geometry.h_rotor_m,
        geometry.rotor_chord_m,
    )
    phi_r = incidence_corrected_velocity_coefficient(
        phi_r_zero_incidence,
        incidence_deg,
    )
    active_arc = geometry.partial_admission_eff * np.pi * geometry.D_mid_m
    Ks = sector_filling_coefficient(geometry.rotor_pitch_m, active_arc)
    w2_signed = -w1 * phi_r * Ks
    beta2 = np.radians(beta2_geometry)
    w2a = w2_signed * (-np.sin(beta2))
    w2u = w2_signed * (-np.cos(beta2))
    c2a = float(w2a)
    c2u = float(w2u + U)
    c2 = float(np.hypot(c2a, c2u))

    h2 = float(stator["h_nozzle_out_J_kg"]) + 0.5 * (w1**2 - w2_signed**2)
    state2 = backend.state_PH(float(stator["p_out_Pa"]), h2, geometry.fluid)
    mdot = float(stator["m_dot_kg_s"])
    P_aero = mdot * U * (c1u - c2u)
    P_fric = disk_friction_loss_W(n_rpm, geometry.D_mid_m, state2["rho_kg_m3"])
    pa = partial_admission_losses_W(
        geometry.partial_admission_eff,
        state2["rho_kg_m3"],
        n_rpm,
        geometry.D_mid_m,
        geometry.h_rotor_m,
        U,
    )
    P_mech = P_aero - P_fric - pa["selected_W"]
    dh_is = float(stator["dh_is_J_kg"])
    eta = P_mech / max(mdot * dh_is, 1e-12)
    h_out = float(stator["h_in_J_kg"]) - eta * dh_is
    state_out = backend.state_PH(float(stator["p_out_Pa"]), h_out, geometry.fluid)
    area_ok = abs(float(stator["area_mismatch_rel"])) <= 0.40
    valid = bool(
        np.isfinite(eta)
        and 0.0 < eta < 1.0
        and P_mech > 0.0
        and area_ok
    )
    return {
        **stator,
        "valid": valid,
        "n_rpm": float(n_rpm),
        "U_m_s": float(U),
        "U_over_cis": float(U / np.sqrt(2.0 * dh_is)),
        "Ma1_rel": float(Ma1r),
        "beta1_flow_deg": float(beta1_flow_deg),
        "beta1_metal_deg": float(geometry.beta_rotor_in_deg),
        "incidence_deg": float(incidence_deg),
        "phi_rotor_zero_incidence": float(phi_r_zero_incidence),
        "phi_rotor": float(phi_r),
        "sector_coefficient": float(Ks),
        "c2a_m_s": c2a,
        "c2u_m_s": c2u,
        "c2_m_s": c2,
        "P_aero_W": float(P_aero),
        "P_fric_W": float(P_fric),
        "P_partial_admission_W": float(pa["selected_W"]),
        "P_mech_W": float(P_mech),
        "eta_turb": float(eta),
        "T_out_K": float(state_out["T_K"]),
        "m_dot_rel": float(mdot / geometry.m_dot_design_kg_s),
        "P_mech_rel": float(P_mech / geometry.P_mech_design_W),
        "PR_rel": float(stator["pressure_ratio"] / geometry.pressure_ratio_design),
        "n_rel": float(n_rpm / geometry.n_design_rpm),
    }


def records_to_frame(records: list[dict[str, float | bool]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)
