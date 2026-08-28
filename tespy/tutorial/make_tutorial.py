"""Derive the participant notebook and the checkpoint notebooks from
the reference notebook.

The reference notebook (:code:`reference/tutorial_reference.ipynb`)
is the single source of truth: it is complete, it runs, and every code
cell carries tags that say what to do with it.

    python make_tutorial.py              # regenerate everything
    python make_tutorial.py --execute    # run the reference first, then regenerate

Participants get :code:`tutorial.ipynb` with every cell they are
supposed to type left empty. Each checkpoint is the same notebook with
the steps up to that point already filled in - fall behind, open the
checkpoint, run the filled part and carry on in it.

Tags
----
:code:`step-N`
    Which step of the walkthrough the cell belongs to. Checkpoints are
    cut at step boundaries, so every code cell needs exactly one.
:code:`type-along`
    The participants type this cell themselves - it is empty in the
    participant notebook and in every checkpoint before its step.
:code:`given`
    Shipped filled in everywhere: imports, plotting, result printing.
    Nobody should type a matplotlib call under time pressure.
:code:`raises-exception`
    Fails on purpose. Kept everywhere, and tolerated by
    :code:`--execute`.

Adding a step means tagging its cells :code:`step-N` and, if it
should be catchable, adding it to :code:`CHECKPOINTS` below.
"""
import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
REFERENCE = HERE / "reference" / "tutorial_reference.ipynb"
PARTICIPANT = HERE / "tutorial.ipynb"
CHECKPOINT_DIR = HERE / "checkpoints"

#: name of the checkpoint -> last step it has filled in
CHECKPOINTS = [
    ("checkpoint_1_components", 2),
    ("checkpoint_2_connections", 4),
    ("checkpoint_3_parametrized", 5),
    ("checkpoint_4_debugging", 7),
]

BANNER = """<div class="alert alert-block alert-info">

**Catch-up copy - steps 1 to {step} are filled in.**

Run the filled cells down to the end of step {step} (in JupyterLab:
select the last one and use *Run > Run All Above Selected Cell*, then
run it), and continue with step {next_step} as if you had typed them
yourself.

Generated from `{source}`.

</div>"""


def tags(cell):
    return set(cell.get("metadata", {}).get("tags", []))


def step_of(cell):
    for tag in tags(cell):
        match = re.fullmatch(r"step-(\d+)", tag)
        if match:
            return int(match.group(1))
    return None


def markdown(text):
    lines = text.split("\n")
    return {"cell_type": "markdown", "metadata": {"tags": ["generated"]},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def check(notebook):
    """Every code cell needs a step and a role - a cell that carries
    neither is almost always a tag that was forgotten."""
    problems = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        if step_of(cell) is None:
            problems.append(f"cell {index}: no step-N tag")
        if not tags(cell) & {"type-along", "given"}:
            problems.append(f"cell {index}: neither type-along nor given")
    return problems


def derive(notebook, filled_through=0):
    """A copy of the notebook with outputs cleared and every
    ``type-along`` cell above ``filled_through`` emptied."""
    out = copy.deepcopy(notebook)
    emptied = 0
    for cell in out["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
        if "type-along" in tags(cell) and step_of(cell) > filled_through:
            cell["source"] = []
            emptied += 1
    return out, emptied


def write(path, notebook):
    path.write_text(json.dumps(notebook, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true",
        help="execute the reference notebook before deriving from it"
    )
    args = parser.parse_args()

    if args.execute:
        print(f"executing {REFERENCE.name} ...")
        subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "notebook",
             "--execute", "--inplace", str(REFERENCE)],
            check=True, cwd=HERE,
        )

    notebook = json.loads(REFERENCE.read_text())

    problems = check(notebook)
    if problems:
        print("untagged cells in the reference notebook:")
        for problem in problems:
            print(" ", problem)
        return 1

    scaffold, emptied = derive(notebook)
    write(PARTICIPANT, scaffold)
    print(f"{PARTICIPANT.name}: {emptied} cells left to type")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    for name, last_step in CHECKPOINTS:
        catchup, remaining = derive(notebook, filled_through=last_step)
        catchup["cells"].insert(1, markdown(BANNER.format(
            step=last_step, next_step=last_step + 1,
            source=REFERENCE.name,
        )))
        write(CHECKPOINT_DIR / f"{name}.ipynb", catchup)
        print(f"checkpoints/{name}.ipynb: steps 1-{last_step} filled in, "
              f"{remaining} cells left to type")

    return 0


if __name__ == "__main__":
    sys.exit(main())
