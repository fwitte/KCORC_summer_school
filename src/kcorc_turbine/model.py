from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

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
from .properties import CoolPropBackend, PropertyBackend, normalize_fluid_name


@dataclass(frozen=True)
class DesignInputs:
    fluid: str = "R1233zd(E)"
    p_in_Pa: float = 2.500e6
    T_in_K: float = 150.0 + 273.15
    p_out_Pa: float = 475e3
    m_dot_kg_s: float = 5.025
    n_rpm: float = 11000.0
    D_mid_m: float = 0.2
    alpha_stator_deg: float = 11.5
    beta_rotor_in_deg: float = 23.0
    beta_rotor_out_deg: float = 25.0
    partial_admission: float = 1.0
    rotor_solidity: float = 1.0
    rotor_chord_m: float = 0.025
    stator_solidity: float = 1.2
    rotor_height_clearance_m: float = 0.0003


    def validate(self) -> None:
        if not (self.p_in_Pa > self.p_out_Pa > 0.0):
            raise ValueError("Require p_in > p_out > 0.")
        if self.T_in_K <= 0.0 or self.m_dot_kg_s <= 0.0:
            raise ValueError("Temperature and mass flow must be positive.")
        if self.n_rpm <= 0.0 or self.D_mid_m <= 0.0:
            raise ValueError("Speed and diameter must be positive.")
        if not (0.0 < self.partial_admission <= 1.0):
            raise ValueError("Partial admission must lie in (0, 1].")
        if self.rotor_chord_m <= 0.0 or self.rotor_solidity <= 0.0:
            raise ValueError("Chord and solidity must be positive.")


@dataclass
class TurbineDesignResult:
    inputs: DesignInputs
    states: dict[str, dict[str, float]]
    path: pd.DataFrame
    performance: dict[str, float | bool]
    geometry: dict[str, float | int]
    velocities: dict[str, float]
    losses: dict[str, float]

    def summary(self) -> pd.DataFrame:
        rows = [
            ("Fluid", self.inputs.fluid, "-"),
            ("Pressure ratio", self.performance["pressure_ratio"], "-"),
            ("Isentropic enthalpy drop", self.performance["dh_is_J_kg"] / 1e3, "kJ/kg"),
            ("Isentropic velocity", self.performance["c_is_m_s"], "m/s"),
            ("Ideal nozzle exit Mach", self.performance["Ma_out_is"], "-"),
            ("Loss-adjusted nozzle exit Mach", self.performance["Ma_nozzle_out"], "-"),
            ("Blade-speed ratio U/c_is", self.performance["U_over_cis"], "-"),
            ("Aerodynamic power", self.performance["P_aero_W"] / 1e3, "kW"),
            ("Mechanical power", self.performance["P_mech_W"] / 1e3, "kW"),
            ("Turbine efficiency", self.performance["eta_turb"], "-"),
            ("Throat pressure", self.performance["p_throat_Pa"] / 1e3, "kPa"),
            ("Throat area (total)", self.geometry["A_throat_m2"] * 1e6, "mm²"),
            ("Nozzle exit area (total)", self.geometry["A_outlet_m2"] * 1e6, "mm²"),
            ("Area ratio A_out/A*", self.geometry["area_ratio"], "-"),
            ("Effective partial admission", self.geometry["partial_admission_eff"], "-"),
            ("Nozzle count", self.geometry["no_nozzles"], "-"),
            ("Rotor blade count", self.geometry["no_rotor_blades"], "-"),
            ("Nozzle height", self.geometry["h_nozzle_m"] * 1e3, "mm"),
            ("Rotor height", self.geometry["h_rotor_m"] * 1e3, "mm"),
            ("Rotor hub diameter", self.geometry["D_hub_m"] * 1e3, "mm"),
            ("Rotor tip diameter", self.geometry["D_tip_m"] * 1e3, "mm"),
            ("Rotor exit swirl c2u", self.velocities["c2u_m_s"], "m/s"),
        ]
        return pd.DataFrame(rows, columns=["Quantity", "Value", "Unit"])

    def to_json_dict(self) -> dict:
        return {
            "model_version": "kcorc-student-draft-0.2",
            "inputs": asdict(self.inputs),
            "states": self.states,
            "performance": self.performance,
            "geometry": self.geometry,
            "velocities": self.velocities,
            "losses": self.losses,
        }

    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")
        return path


def _isentropic_path(
    inputs: DesignInputs,
    backend: PropertyBackend,
    n_steps: int,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    state_in = backend.state_PT(inputs.p_in_Pa, inputs.T_in_K, inputs.fluid)
    state_out_is = backend.state_PS(inputs.p_out_Pa, state_in["s_J_kgK"], inputs.fluid)
    p_values = np.linspace(inputs.p_in_Pa, inputs.p_out_Pa, int(n_steps))
    records: list[dict[str, float]] = []
    for p in p_values:
        st = backend.state_PS(float(p), state_in["s_J_kgK"], inputs.fluid)
        dh = max(state_in["h_J_kg"] - st["h_J_kg"], 0.0)
        c = float(np.sqrt(2.0 * dh))
        Ma = c / max(st["a_m_s"], 1e-12)
        records.append(
            {
                "p_Pa": float(p),
                "pressure_ratio_local": float(p / inputs.p_out_Pa),
                "T_is_K": st["T_K"],
                "h_is_J_kg": st["h_J_kg"],
                "rho_is_kg_m3": st["rho_kg_m3"],
                "a_is_m_s": st["a_m_s"],
                "c_is_m_s": c,
                "Ma_is": Ma,
            }
        )
    return state_in, state_out_is, pd.DataFrame.from_records(records)


def design_stage(
    inputs: DesignInputs,
    backend: PropertyBackend | None = None,
    n_steps: int = 320,
) -> TurbineDesignResult:
    """Complete reference implementation used by widgets and checkpoints.

    The student notebook separately reconstructs the key equations in small
    TODO cells. This function exists to provide a robust interactive reference.
    """
    inputs.validate()
    backend = backend or CoolPropBackend()
    fluid = normalize_fluid_name(inputs.fluid)
    if fluid != inputs.fluid:
        inputs = DesignInputs(**{**asdict(inputs), "fluid": fluid})

    state_in, state_out_is, path = _isentropic_path(inputs, backend, n_steps)
    dh_is = state_in["h_J_kg"] - state_out_is["h_J_kg"]
    if dh_is <= 0.0:
        raise ValueError("The selected states do not provide a positive turbine enthalpy drop.")
    c_is = float(np.sqrt(2.0 * dh_is))
    U = float(np.pi * inputs.D_mid_m * inputs.n_rpm / 60.0)
    U_over_cis = U / c_is

    Ma_is = path["Ma_is"].to_numpy()
    crossings = np.flatnonzero(Ma_is >= 1.0)
    choked = bool(crossings.size)
    throat_index = int(crossings[0] if choked else np.argmax(Ma_is))
    phi_stator = stator_velocity_coefficient(float(path.iloc[-1]["Ma_is"]))

    h_actual = path["h_is_J_kg"].to_numpy(copy=True)
    if choked:
        h_throat = float(path.iloc[throat_index]["h_is_J_kg"])
        for i in range(throat_index, len(path)):
            h_actual[i] = h_throat - phi_stator**2 * (
                h_throat - float(path.iloc[i]["h_is_J_kg"])
            )
    else:
        h_actual = state_in["h_J_kg"] - phi_stator**2 * (
            state_in["h_J_kg"] - path["h_is_J_kg"].to_numpy()
        )

    actual_records = []
    for p, h in zip(path["p_Pa"], h_actual):
        st = backend.state_PH(float(p), float(h), inputs.fluid)
        c = float(np.sqrt(max(2.0 * (state_in["h_J_kg"] - h), 0.0)))
        actual_records.append((st["T_K"], st["rho_kg_m3"], st["a_m_s"], c, c / st["a_m_s"]))
    actual = np.asarray(actual_records)
    path["h_actual_J_kg"] = h_actual
    path["T_actual_K"] = actual[:, 0]
    path["rho_actual_kg_m3"] = actual[:, 1]
    path["a_actual_m_s"] = actual[:, 2]
    path["c_actual_m_s"] = actual[:, 3]
    path["Ma_actual"] = actual[:, 4]

    # The source meanline model applies the stator velocity coefficient to
    # the complete isentropic nozzle enthalpy drop for the outlet state.
    h_nozzle_out = state_in["h_J_kg"] - phi_stator**2 * (
        state_in["h_J_kg"] - state_out_is["h_J_kg"]
    )
    state_nozzle_out = backend.state_PH(
        inputs.p_out_Pa, float(h_nozzle_out), inputs.fluid
    )
    c_nozzle_out = float(phi_stator * c_is)
    Ma_nozzle_out = c_nozzle_out / state_nozzle_out["a_m_s"]

    throat = path.iloc[throat_index]
    A_throat = inputs.m_dot_kg_s / max(
        throat["rho_is_kg_m3"] * throat["c_is_m_s"], 1e-12
    )
    A_outlet = inputs.m_dot_kg_s / max(
        state_nozzle_out["rho_kg_m3"] * c_nozzle_out, 1e-12
    )
    area_ratio = A_outlet / A_throat

    alpha = np.radians(inputs.alpha_stator_deg)
    c1a = c_nozzle_out * np.sin(alpha)
    c1u = c_nozzle_out * np.cos(alpha)

    chord_s_ax = 1.25 * inputs.rotor_chord_m
    pitch_s_nom = chord_s_ax / inputs.stator_solidity
    circumference = np.pi * inputs.D_mid_m
    no_nozzles = max(1, int(round(inputs.partial_admission * circumference / pitch_s_nom)))
    active_arc = no_nozzles * pitch_s_nom
    partial_eff = float(np.clip(active_arc / circumference, 1e-6, 1.0))
    h_nozzle = inputs.m_dot_kg_s / max(
        state_nozzle_out["rho_kg_m3"] * c1a * partial_eff * circumference,
        1e-12,
    )
    h_rotor = h_nozzle + inputs.rotor_height_clearance_m
    b_throat_total = A_throat / h_nozzle
    b_outlet_total = A_outlet / h_nozzle

    w1a = c1a
    w1u = c1u - U
    w1 = float(np.hypot(w1a, w1u))
    beta1_flow = float(np.degrees(np.arctan2(w1a, w1u)))
    incidence_deg = beta1_flow - inputs.beta_rotor_in_deg
    Ma1_rel = w1 / state_nozzle_out["a_m_s"]

    # The empirical rotor correlation is a design-point / zero-incidence
    # correlation, so theta remains a geometric blade turning angle.
    beta2_geometry = 180.0 - inputs.beta_rotor_out_deg
    theta = beta2_geometry - inputs.beta_rotor_in_deg
    phi_rotor_base, phi_rotor_zero_incidence = rotor_velocity_coefficient(
        theta, Ma1_rel, h_rotor, inputs.rotor_chord_m
    )

    # Off-design inlet incidence is added as a separate loss mechanism.
    phi_rotor = incidence_corrected_velocity_coefficient(
        phi_rotor_zero_incidence,
        incidence_deg,
    )

    pitch_r_nom = inputs.rotor_chord_m / inputs.rotor_solidity
    no_rotor_blades = max(3, int(round(circumference / pitch_r_nom)))
    pitch_r = circumference / no_rotor_blades
    sigma_r_actual = inputs.rotor_chord_m / pitch_r
    active_arc_r = partial_eff * circumference
    Ks = sector_filling_coefficient(pitch_r, active_arc_r)

    w2_signed = -w1 * phi_rotor * Ks
    beta2_rad = np.radians(beta2_geometry)
    w2a = w2_signed * (-np.sin(beta2_rad))
    w2u = w2_signed * (-np.cos(beta2_rad))
    c2a = float(w2a)
    c2u = float(w2u + U)
    c2 = float(np.hypot(c2a, c2u))

    dh_rotor_static = 0.5 * (w1**2 - w2_signed**2)
    h2 = state_nozzle_out["h_J_kg"] + dh_rotor_static
    state_rotor_out = backend.state_PH(inputs.p_out_Pa, h2, inputs.fluid)
    DOR_proxy = dh_rotor_static / dh_is

    P_aero = inputs.m_dot_kg_s * U * (c1u - c2u)
    P_fric = disk_friction_loss_W(inputs.n_rpm, inputs.D_mid_m, state_rotor_out["rho_kg_m3"])
    pa = partial_admission_losses_W(
        partial_eff,
        state_rotor_out["rho_kg_m3"],
        inputs.n_rpm,
        inputs.D_mid_m,
        h_rotor,
        U,
    )
    P_mech = P_aero - P_fric - pa["selected_W"]
    eta_turb = P_mech / (inputs.m_dot_kg_s * dh_is)
    h_out_eta = state_in["h_J_kg"] - eta_turb * dh_is
    state_out_eta = backend.state_PH(inputs.p_out_Pa, h_out_eta, inputs.fluid)

    geometry = {
        "A_throat_m2": float(A_throat),
        "A_outlet_m2": float(A_outlet),
        "area_ratio": float(area_ratio),
        "h_nozzle_m": float(h_nozzle),
        "h_rotor_m": float(h_rotor),
        "D_hub_m": float(inputs.D_mid_m - h_rotor),
        "D_tip_m": float(inputs.D_mid_m + h_rotor),
        "partial_admission_eff": partial_eff,
        "active_arc_m": float(active_arc),
        "stator_pitch_m": float(pitch_s_nom),
        "no_nozzles": no_nozzles,
        "b_nozzle_throat_m": float(b_throat_total / no_nozzles),
        "b_nozzle_out_m": float(b_outlet_total / no_nozzles),
        "rotor_pitch_m": float(pitch_r),
        "rotor_solidity_actual": float(sigma_r_actual),
        "no_rotor_blades": no_rotor_blades,
        "rotor_chord_m": float(inputs.rotor_chord_m),
    }
    velocities = {
        "U_m_s": U,
        "c1_m_s": float(c_nozzle_out),
        "c1a_m_s": float(c1a),
        "c1u_m_s": float(c1u),
        "w1_m_s": w1,
        "w1a_m_s": float(w1a),
        "w1u_m_s": float(w1u),
        "beta1_flow_deg": beta1_flow,
        "beta1_metal_deg": float(inputs.beta_rotor_in_deg),
        "incidence_deg": float(incidence_deg),
        "Ma1_rel": float(Ma1_rel),
        "w2_m_s": float(abs(w2_signed)),
        "w2a_m_s": float(w2a),
        "w2u_m_s": float(w2u),
        "c2_m_s": c2,
        "c2a_m_s": c2a,
        "c2u_m_s": c2u,
        "beta2_geometry_deg": float(beta2_geometry),
    }
    performance = {
        "pressure_ratio": float(inputs.p_in_Pa / inputs.p_out_Pa),
        "dh_is_J_kg": float(dh_is),
        "c_is_m_s": c_is,
        "U_over_cis": float(U_over_cis),
        "Ma_out_is": float(path.iloc[-1]["Ma_is"]),
        "Ma_nozzle_out": float(Ma_nozzle_out),
        "phi_stator": float(phi_stator),
        "phi_rotor_base": float(phi_rotor_base),
        "phi_rotor_zero_incidence": float(phi_rotor_zero_incidence),
        "phi_rotor": float(phi_rotor),
        "incidence_deg": float(incidence_deg),
        "sector_coefficient": float(Ks),
        "p_throat_Pa": float(throat["p_Pa"]),
        "T_throat_K": float(throat["T_is_K"]),
        "choked": choked,
        "P_aero_W": float(P_aero),
        "P_mech_W": float(P_mech),
        "eta_turb": float(eta_turb),
        "DOR_proxy": float(DOR_proxy),
        "h_out_eta_J_kg": float(h_out_eta),
        "T_out_eta_K": float(state_out_eta["T_K"]),
    }
    losses = {
        "P_fric_W": float(P_fric),
        "P_partial_admission_W": float(pa["selected_W"]),
        "P_partial_admission_legacy_W": float(pa["legacy_W"]),
        "P_partial_admission_pumping_W": float(pa["pumping_W"]),
    }
    states = {
        "inlet": state_in,
        "outlet_isentropic": state_out_is,
        "nozzle_outlet": state_nozzle_out,
        "rotor_outlet_static": state_rotor_out,
        "outlet_from_efficiency": state_out_eta,
    }
    return TurbineDesignResult(inputs, states, path, performance, geometry, velocities, losses)


def freeze_geometry(
    result: TurbineDesignResult,
    backend: PropertyBackend | None = None,
):
    from .offdesign import FixedGeometry

    return FixedGeometry.from_design(result, backend=backend)
