# TESPy Workshop at the KCORC Summer School 2026

This is a hands-on modelling workshop using
[TESPy](https://tespy.readthedocs.io) as part of the KCORC Summer School 2026.
The workshop is organized in six groups, three of which on an
**Organic Rankine Cycle** case study and three on a **Heat Pump** case study.
Each group has its own folder with the task description.

## Group overview

| Group | Topic | Materials |
| --- | --- | --- |
| ORC 1 | ORC design | [ORC_design/](ORC_design/) |
| ORC 2 | ORC off-design | [ORC_offdesign/](ORC_offdesign/) |
| ORC 3 | ORC with zeotropic mixtures | [ORC_zeotropic/](ORC_zeotropic/) |
| HP 1 | Heat pump design | [HP_design/](HP_design/) |
| HP 2 | Heat pump off-design | [HP_offdesign/](HP_offdesign/) |
| HP 3 | Heat pump with zeotropic mixtures | [HP_zeotropic/](HP_zeotropic/) |

## ORC case study: 5.5 MWe double-stage ORC Kirchstockach

The ORC groups work on the Kirchstockach geothermal power plant, a two-stage
ORC consisting of a High-Temperature (HT) and a Low-Temperature (LT) cycle
operating in series on the same geothermal brine ([flowsheet](orc.svg),
reference:
[Heberle et al., 2015](https://worldgeothermal.org/pdf/IGAstandard/WGC/2015/26002.pdf)).

## Heat pump case study

## Getting started

1. Set up the Python environment from the repository root:
   - Windows: run `START_KCORC.bat` (installs [uv](https://docs.astral.sh/uv/)
     locally and creates the environment), or
   - download uv manually and run `uv sync`
2. Start JupyterLab (`uv run jupyter lab`) or open the notebooks in VS Code
3. Open your group's notebook, which provides the task description.

## Further materials

- [slides/](slides/): workshop slides
- [orc.svg](orc.svg): flowsheet of the Kirchstockach double-stage ORC

## Authors

- Amalia Stainchaouer
- Francesco Witte
- Jannik von Zabienski
- Christopher Schifflechner
