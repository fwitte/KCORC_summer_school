"""Teaching backend for the KCORC supersonic ORC turbine tutorial."""

from .model import DesignInputs, TurbineDesignResult, design_stage, freeze_geometry
from .offdesign import (
    FixedGeometry,
    evaluate_stator_offdesign,
    evaluate_rotor_offdesign,
    inlet_temperature_for_pressure,
)
from .properties import CoolPropBackend, PropertyBackend, normalize_fluid_name

__all__ = [
    "CoolPropBackend",
    "DesignInputs",
    "FixedGeometry",
    "PropertyBackend",
    "TurbineDesignResult",
    "design_stage",
    "evaluate_rotor_offdesign",
    "evaluate_stator_offdesign",
    "freeze_geometry",
    "inlet_temperature_for_pressure",
    "normalize_fluid_name",
]
