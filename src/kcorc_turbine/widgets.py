from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import matplotlib.pyplot as plt

from .model import DesignInputs, design_stage
from .plotting import plot_nozzle_expansion, plot_velocity_triangles
from .properties import CoolPropBackend
import numpy as np


def _widgets():
    try:
        import ipywidgets as widgets
        from IPython.display import HTML, clear_output, display
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ipywidgets is required. Run `uv sync`.") from exc
    return widgets, HTML, clear_output, display


def design_explorer(default: DesignInputs | None = None):
    """Create a compact live design-point explorer for the lecture/demo."""
    widgets, _, _, _ = _widgets()
    default = default or DesignInputs()
    backend = CoolPropBackend()

    fluid = widgets.Dropdown(
        options=["R1233zd(E)", "R245fa", "Cyclopentane"],
        value=(default.fluid if default.fluid in {"R1233zd(E)", "R245fa", "Cyclopentane"} else "R1233zd(E)"),
        description="Fluid",
    )
    p_in = widgets.FloatSlider(value=default.p_in_Pa / 1e3, min=900, max=3400, step=50, description="p in [kPa]", continuous_update=False)
    p_out = widgets.FloatSlider(value=default.p_out_Pa / 1e3, min=150, max=900, step=25, description="p out [kPa]", continuous_update=False)
    T_in = widgets.FloatSlider(value=default.T_in_K - 273.15, min=90, max=210, step=2, description="T in [°C]", continuous_update=False)
    n = widgets.FloatSlider(value=default.n_rpm, min=2000, max=18000, step=250, description="n [rpm]", continuous_update=False)
    D = widgets.FloatSlider(value=default.D_mid_m, min=0.20, max=0.60, step=0.01, description="D mid [m]", continuous_update=False)
    e = widgets.FloatSlider(value=default.partial_admission, min=0.15, max=1.0, step=0.025, description="admission [-]", continuous_update=False)
    alpha = widgets.FloatSlider(value=default.alpha_stator_deg, min=8, max=22, step=0.5, description="alpha 1 [deg]", continuous_update=False)
    beta1 = widgets.FloatSlider(value=default.beta_rotor_in_deg, min=15, max=40, step=1, description="beta 1 [deg]", continuous_update=False)
    beta2 = widgets.FloatSlider(value=default.beta_rotor_out_deg, min=15, max=40, step=1, description="beta 2 [deg]", continuous_update=False)

    controls_list = [fluid, p_in, p_out, T_in, n, D, e, alpha, beta1, beta2]

    summary = widgets.HTML()
    nozzle_image = widgets.Image(format="png", layout=widgets.Layout(max_width="800px"))
    triangle_image = widgets.Image(format="png", layout=widgets.Layout(max_width="800px"))
    error_message = widgets.HTML()

    def figure_to_png(fig):
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return buffer.getvalue()

    def update(*_):
        error_message.value = ""
        try:
            inp = replace(
                default,
                fluid=fluid.value,
                p_in_Pa=p_in.value * 1e3,
                p_out_Pa=p_out.value * 1e3,
                T_in_K=T_in.value + 273.15,
                n_rpm=n.value,
                D_mid_m=D.value,
                partial_admission=e.value,
                alpha_stator_deg=alpha.value,
                beta_rotor_in_deg=beta1.value,
                beta_rotor_out_deg=beta2.value,
            )

            result = design_stage(inp, backend=backend, n_steps=180)
            p = result.performance
            g = result.geometry
            v = result.velocities

            summary.value = f"""
            <div style='display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:8px;margin:6px 0 12px 0'>
              <div style='padding:9px;background:#eef4ff;border-radius:6px'><b>η</b><br>{p['eta_turb']:.3f}</div>
              <div style='padding:9px;background:#eef4ff;border-radius:6px'><b>P mech</b><br>{p['P_mech_W']/1e3:.1f} kW</div>
              <div style='padding:9px;background:#eef4ff;border-radius:6px'><b>Ma nozzle</b><br>{p['Ma_nozzle_out']:.2f}</div>
              <div style='padding:9px;background:#eef4ff;border-radius:6px'><b>U/c is</b><br>{p['U_over_cis']:.3f}</div>

              <div style='padding:9px;background:#f5f5f5;border-radius:6px'><b>Blade height</b><br>{g['h_rotor_m']*1e3:.1f} mm</div>
              <div style='padding:9px;background:#f5f5f5;border-radius:6px'><b>Nozzles</b><br>{g['no_nozzles']}</div>
              <div style='padding:9px;background:#f5f5f5;border-radius:6px'><b>Rotor blades</b><br>{g['no_rotor_blades']}</div>
              <div style='padding:9px;background:#f5f5f5;border-radius:6px'><b>c2u</b><br>{v['c2u_m_s']:.1f} m/s</div>

              <div style='padding:9px;background:#fff8e1;border-radius:6px'><b>β1 flow</b><br>{v['beta1_flow_deg']:.1f}°</div>
              <div style='padding:9px;background:#fff8e1;border-radius:6px'><b>β1 metal</b><br>{v['beta1_metal_deg']:.1f}°</div>
              <div style='padding:9px;background:#fff8e1;border-radius:6px'><b>Incidence i</b><br>{v['incidence_deg']:+.1f}°</div>
              <div style='padding:9px;background:#fff8e1;border-radius:6px'><b>φ rotor eff.</b><br>{p['phi_rotor']:.3f}</div>

              <div style='padding:9px;background:#f5f5f5;border-radius:6px'><b>β2 metal</b><br>{inp.beta_rotor_out_deg:.1f}°</div>
              <div style='padding:9px;background:#f5f5f5;border-radius:6px'><b>Ma1 rel</b><br>{v['Ma1_rel']:.2f}</div>
              <div style='padding:9px;background:#f5f5f5;border-radius:6px'><b>φ rotor, i=0</b><br>{p['phi_rotor_zero_incidence']:.3f}</div>
            </div>
            """

            if result.performance["choked"]:
                crossings = np.flatnonzero(result.path["Ma_is"].to_numpy() >= 1.0)
                throat_index = int(crossings[0]) if len(crossings) else None
            else:
                throat_index = None

            with plt.ioff():
                fig1, _ = plot_nozzle_expansion(result.path, throat_index)
                fig2, _ = plot_velocity_triangles(result.velocities)

            nozzle_image.value = figure_to_png(fig1)
            triangle_image.value = figure_to_png(fig2)

        except Exception as exc:
            plt.close("all")
            summary.value = ""
            nozzle_image.value = b""
            triangle_image.value = b""
            error_message.value = f"""
            <div style='padding:10px;background:#fff1e8;border-left:5px solid #c45100'>
                <b>Invalid design point:</b> {exc}
            </div>
            """

    controls = widgets.VBox(controls_list, layout=widgets.Layout(width="390px"))
    results = widgets.VBox([summary, nozzle_image, triangle_image, error_message], layout=widgets.Layout(width="850px"))
    ui = widgets.HBox([controls, results], layout=widgets.Layout(align_items="flex-start"))

    for control in controls_list:
        control.observe(update, names="value")

    update()
    return ui


def offdesign_explorer(map_df, design_PR: float, design_n_rpm: float):
    """Interactive interpolation of an already generated off-design map."""
    widgets, HTML, clear_output, display = _widgets()
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator

    valid = map_df.loc[map_df["valid"] & map_df["eta_turb"].notna()].copy()
    points = valid[["pressure_ratio", "n_rpm"]].to_numpy()
    eta_interp = LinearNDInterpolator(points, valid["eta_turb"].to_numpy())
    power_interp = LinearNDInterpolator(points, valid["P_mech_W"].to_numpy())
    mdot_interp = LinearNDInterpolator(points, valid["m_dot_kg_s"].to_numpy())

    pr = widgets.FloatSlider(value=design_PR, min=float(valid["pressure_ratio"].min()), max=float(valid["pressure_ratio"].max()), step=0.05, description="PR [-]", continuous_update=False)
    n = widgets.FloatSlider(value=design_n_rpm, min=float(valid["n_rpm"].min()), max=float(valid["n_rpm"].max()), step=250, description="n [rpm]", continuous_update=False)
    output = widgets.Output()

    def update(*_):
        with output:
            clear_output(wait=True)
            eta = float(np.asarray(eta_interp(pr.value, n.value)))
            power = float(np.asarray(power_interp(pr.value, n.value)))
            mdot = float(np.asarray(mdot_interp(pr.value, n.value)))
            if not np.isfinite(eta):
                display(HTML("<div style='padding:10px;background:#fff1e8'>The selected point lies outside the valid interpolated region.</div>"))
                return
            display(HTML(
                f"<div style='display:flex;gap:10px'>"
                f"<div style='padding:10px;background:#eef4ff;border-radius:6px'><b>η</b><br>{eta:.3f}</div>"
                f"<div style='padding:10px;background:#eef4ff;border-radius:6px'><b>P</b><br>{power/1e3:.1f} kW</div>"
                f"<div style='padding:10px;background:#eef4ff;border-radius:6px'><b>m dot</b><br>{mdot:.2f} kg/s</div>"
                f"</div>"
            ))

    pr.observe(update, names="value")
    n.observe(update, names="value")
    update()
    return widgets.VBox([widgets.HBox([pr, n]), output])
