from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_nozzle_expansion(path: pd.DataFrame, throat_index: int | None = None):
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    ax.plot(path["pressure_ratio_local"], path["Ma_is"], label="Isentropic")
    if "Ma_actual" in path:
        ax.plot(path["pressure_ratio_local"], path["Ma_actual"], "--", label="Loss-adjusted")
    if throat_index is None:
        throat_index = int(np.argmin(np.abs(path["Ma_is"].to_numpy() - 1.0)))
    row = path.iloc[int(throat_index)]
    ax.scatter([row["pressure_ratio_local"]], [row["Ma_is"]], marker="x", s=70, label="Sonic throat")
    ax.axhline(1.0, linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"Local pressure ratio $p/p_{out}$ [-]")
    ax.set_ylabel("Mach number [-]")
    ax.invert_xaxis()
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_velocity_triangles(velocities: dict[str, float]):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    U = velocities["U_m_s"]
    vectors = [
        ("$c_1$", 0.0, 0.0, velocities["c1u_m_s"], velocities["c1a_m_s"]),
        ("$w_1$", 0.0, 0.0, velocities["w1u_m_s"], velocities["w1a_m_s"]),
        ("$U$ inlet", velocities["w1u_m_s"], velocities["w1a_m_s"], U, 0.0),
        ("$c_2$", 0.0, 0.0, velocities["c2u_m_s"], velocities["c2a_m_s"]),
        ("$w_2$", 0.0, 0.0, velocities["w2u_m_s"], velocities["w2a_m_s"]),
        ("$U$ outlet", velocities["w2u_m_s"], velocities["w2a_m_s"], U, 0.0),
    ]
    for label, x, y, dx, dy in vectors:
        ax.arrow(x, y, dx, dy, length_includes_head=True, head_width=4.0, head_length=5.0, alpha=0.85)
        ax.text(x + dx, y + dy, " " + label, fontsize=10)
    ax.axhline(0.0, linewidth=0.7)
    ax.axvline(0.0, linewidth=0.7)
    ax.set_xlabel("Tangential velocity [m/s]")
    ax.set_ylabel("Axial velocity [m/s]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax


def plot_loss_breakdown(result):
    labels = ["Aerodynamic", "Disk friction", "Partial admission", "Mechanical"]
    values = [
        result.performance["P_aero_W"] / 1e3,
        result.losses["P_fric_W"] / 1e3,
        result.losses["P_partial_admission_W"] / 1e3,
        result.performance["P_mech_W"] / 1e3,
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(labels, values)
    ax.set_ylabel("Power [kW]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig, ax


def plot_offdesign_map(
    map_df: pd.DataFrame,
    design_PR: float | None = None,
    design_n_rpm: float | None = None,
):
    required = {"pressure_ratio", "n_rpm", "eta_turb"}
    missing = required.difference(map_df.columns)
    if missing:
        raise ValueError(f"Missing map columns: {sorted(missing)}")
    pivot = map_df.pivot(index="pressure_ratio", columns="n_rpm", values="eta_turb")
    n_grid, pr_grid = np.meshgrid(pivot.columns.to_numpy(), pivot.index.to_numpy())
    eta = pivot.to_numpy(dtype=float)
    if "valid" in map_df:
        valid = map_df.pivot(index="pressure_ratio", columns="n_rpm", values="valid").to_numpy(dtype=bool)
        eta = np.where(valid, eta, np.nan)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    filled = ax.contourf(n_grid, pr_grid, eta, levels=14)
    contours = ax.contour(n_grid, pr_grid, eta, levels=8, linewidths=1.0)
    ax.clabel(contours, inline=True, fontsize=8)
    fig.colorbar(filled, ax=ax, label="Turbine efficiency [-]")
    if design_PR is not None and design_n_rpm is not None:
        ax.scatter([design_n_rpm], [design_PR], marker="*", s=120, label="Design point")
        ax.legend()
    ax.set_xlabel("Rotational speed [rpm]")
    ax.set_ylabel("Pressure ratio [-]")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig, ax
