from __future__ import annotations

import numpy as np


def stator_velocity_coefficient(Ma_out_is: float) -> float:
    """Velocity coefficient used in the source meanline model."""
    loss_fraction = (
        0.0029 * Ma_out_is**3
        - 0.0502 * Ma_out_is**2
        + 0.2241 * Ma_out_is
        - 0.0877
    )
    return float(np.sqrt(np.clip(1.0 - loss_fraction, 1e-6, 1.0)))


def rotor_velocity_coefficient(
    theta_deg: float,
    Ma1_rel: float,
    blade_height_m: float,
    chord_m: float,
) -> tuple[float, float]:
    """Return base and height-corrected rotor relative-velocity coefficients."""
    theta = float(theta_deg)
    M = float(Ma1_rel)
    phi_base = (
        0.957
        - 0.000362 * theta
        - 0.0258 * M
        + 0.00000639 * theta**2
        + 0.0674 * M**2
        - 0.0000000753 * theta**3
        - 0.043 * M**3
        - 0.000238 * theta * M
        + 0.00000145 * theta**2 * M
        + 0.0000425 * theta * M**2
    )
    phi_base = float(np.clip(phi_base, 0.05, 0.999))
    aspect = max(float(blade_height_m) / max(float(chord_m), 1e-9), 1e-6)
    k1, k2 = 2.0, 0.65
    radicand = 1.0 - ((1.0 - phi_base**2) / k1) * (
        1.0 + (k1 - 1.0) * aspect ** (-k2)
    )
    return phi_base, float(np.sqrt(np.clip(radicand, 1e-6, 1.0)))



def incidence_corrected_velocity_coefficient(
    phi_zero_incidence: float,
    incidence_deg: float,
    K_inc: float = 1.0,
) -> float:
    """Apply a simple off-design inlet-incidence penalty.

    The existing rotor velocity-coefficient correlation is treated as a
    zero-incidence/design-point correlation. Off-design incidence adds an
    additional kinetic-energy loss proportional to sin(i)^2:

        phi_eff^2 = phi_0^2 - K_inc * sin(i)^2

    where i = beta1_flow - beta1_metal.

    This is deliberately a low-order teaching correction rather than a
    high-fidelity universal incidence-loss correlation.
    """
    phi0 = float(phi_zero_incidence)
    i_rad = np.radians(float(incidence_deg))
    phi_sq = phi0**2 - float(K_inc) * np.sin(i_rad) ** 2
    return float(np.sqrt(np.clip(phi_sq, 1e-6, 0.999**2)))

def sector_filling_coefficient(rotor_pitch_m: float, active_arc_m: float) -> float:
    """Reduced filling/emptying coefficient retained from the research model."""
    return float(np.clip(1.0 - rotor_pitch_m / max(active_arc_m, 1e-12), 0.0, 1.0))


def disk_friction_loss_W(n_rpm: float, D_mid_m: float, rho2_kg_m3: float) -> float:
    return float(0.01 * (n_rpm / 60.0) ** 3 * D_mid_m**5 * rho2_kg_m3)


def partial_admission_losses_W(
    partial_admission: float,
    rho2_kg_m3: float,
    n_rpm: float,
    D_mid_m: float,
    h_rotor_m: float,
    U_m_s: float,
    enclosure_factor: float = 0.4,
) -> dict[str, float]:
    """Two reduced empirical partial-admission estimates used in the source model.

    These correlations are deliberately exposed as assumptions in the tutorial;
    they should not be interpreted as universal high-fidelity loss laws.
    """
    eps = float(np.clip(partial_admission, 1e-6, 1.0))
    legacy = (
        (1.0 - eps)
        * rho2_kg_m3
        * (n_rpm / 60.0) ** 3
        * D_mid_m**4
        * 3.8
        * h_rotor_m
    )
    Kp = 3.63 * float(enclosure_factor)
    pumping = (
        Kp
        * rho2_kg_m3
        * U_m_s**3
        * h_rotor_m
        * D_mid_m**4
        * (1.0 - eps)
    )
    return {
        "legacy_W": float(legacy),
        "pumping_W": float(pumping),
        "selected_W": float(max(legacy, pumping)),
    }
