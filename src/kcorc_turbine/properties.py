from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


_FLUID_ALIASES = {
    "R1233ZDE": "R1233zd(E)",
    "R1233ZD(E)": "R1233zd(E)",
    "R1233zdE": "R1233zd(E)",
}


def normalize_fluid_name(fluid: str) -> str:
    """Normalize common aliases used in the research and teaching scripts."""
    key = fluid.strip()
    return _FLUID_ALIASES.get(key, _FLUID_ALIASES.get(key.upper(), key))


@runtime_checkable
class PropertyBackend(Protocol):
    """Small property interface used by the teaching model.

    Keeping the interface narrow makes the equations independent of the
    property library and allows unit testing without REFPROP/CoolProp.
    """

    def state_PT(self, p_Pa: float, T_K: float, fluid: str) -> dict[str, float]: ...
    def state_PS(self, p_Pa: float, s_J_kgK: float, fluid: str) -> dict[str, float]: ...
    def state_PH(self, p_Pa: float, h_J_kg: float, fluid: str) -> dict[str, float]: ...
    def dew_temperature(self, p_Pa: float, fluid: str) -> float: ...
    def critical_pressure(self, fluid: str) -> float: ...


@dataclass(frozen=True)
class CoolPropBackend:
    """Real-fluid property backend based on CoolProp."""

    def _cp(self):
        try:
            import CoolProp.CoolProp as CP
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "CoolProp is required. Run `uv sync` in the repository root "
                "and select the resulting .venv as the notebook kernel."
            ) from exc
        return CP

    @staticmethod
    def _valid(value: float, name: str) -> float:
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"CoolProp returned an invalid value for {name}.")
        return value

    def state_PT(self, p_Pa: float, T_K: float, fluid: str) -> dict[str, float]:
        CP = self._cp()
        f = normalize_fluid_name(fluid)
        try:
            return {
                "p_Pa": float(p_Pa),
                "T_K": float(T_K),
                "h_J_kg": self._valid(CP.PropsSI("H", "P", p_Pa, "T", T_K, f), "h"),
                "s_J_kgK": self._valid(CP.PropsSI("S", "P", p_Pa, "T", T_K, f), "s"),
                "rho_kg_m3": self._valid(CP.PropsSI("D", "P", p_Pa, "T", T_K, f), "rho"),
                "a_m_s": self._valid(CP.PropsSI("A", "P", p_Pa, "T", T_K, f), "a"),
            }
        except Exception as exc:
            raise ValueError(
                f"Invalid PT state for {f}: p={p_Pa:.4g} Pa, T={T_K:.3f} K."
            ) from exc

    def state_PS(self, p_Pa: float, s_J_kgK: float, fluid: str) -> dict[str, float]:
        CP = self._cp()
        f = normalize_fluid_name(fluid)
        try:
            return {
                "p_Pa": float(p_Pa),
                "T_K": self._valid(CP.PropsSI("T", "P", p_Pa, "S", s_J_kgK, f), "T"),
                "h_J_kg": self._valid(CP.PropsSI("H", "P", p_Pa, "S", s_J_kgK, f), "h"),
                "s_J_kgK": float(s_J_kgK),
                "rho_kg_m3": self._valid(CP.PropsSI("D", "P", p_Pa, "S", s_J_kgK, f), "rho"),
                "a_m_s": self._valid(CP.PropsSI("A", "P", p_Pa, "S", s_J_kgK, f), "a"),
            }
        except Exception as exc:
            raise ValueError(
                f"Invalid PS state for {f}: p={p_Pa:.4g} Pa, s={s_J_kgK:.4g} J/(kg K)."
            ) from exc

    def state_PH(self, p_Pa: float, h_J_kg: float, fluid: str) -> dict[str, float]:
        CP = self._cp()
        f = normalize_fluid_name(fluid)
        try:
            return {
                "p_Pa": float(p_Pa),
                "T_K": self._valid(CP.PropsSI("T", "P", p_Pa, "H", h_J_kg, f), "T"),
                "h_J_kg": float(h_J_kg),
                "s_J_kgK": self._valid(CP.PropsSI("S", "P", p_Pa, "H", h_J_kg, f), "s"),
                "rho_kg_m3": self._valid(CP.PropsSI("D", "P", p_Pa, "H", h_J_kg, f), "rho"),
                "a_m_s": self._valid(CP.PropsSI("A", "P", p_Pa, "H", h_J_kg, f), "a"),
            }
        except Exception as exc:
            raise ValueError(
                f"Invalid PH state for {f}: p={p_Pa:.4g} Pa, h={h_J_kg:.4g} J/kg."
            ) from exc

    def dew_temperature(self, p_Pa: float, fluid: str) -> float:
        CP = self._cp()
        f = normalize_fluid_name(fluid)
        return self._valid(CP.PropsSI("T", "P", p_Pa, "Q", 1, f), "T_dew")

    def critical_pressure(self, fluid: str) -> float:
        CP = self._cp()
        f = normalize_fluid_name(fluid)
        return self._valid(CP.PropsSI("PCRIT", f), "p_critical")
