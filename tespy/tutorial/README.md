# TESPy tutorial

This is a small geothermal ORC tutorial on the first steps using TESPy:
[tutorial.ipynb](tutorial.ipynb).

It is the same model as the slides show in the solver part.

## Checkpoints

We develop the code together, so you type the code yourself to learn how to
use the software. In case you fall behind or need a clean start, open the last
checkpoint, run it and carry on there.

| notebook | filled in |
| --- | --- |
| `checkpoints/checkpoint_1_components.ipynb` | up to step 2 |
| `checkpoints/checkpoint_2_connections.ipynb` | up to step 4 |
| `checkpoints/checkpoint_3_parametrized.ipynb` | up to step 5 |
| `checkpoints/checkpoint_4_debugging.ipynb` | up to step 7 |

---

## For the lecturer

[reference/tutorial_reference.ipynb](reference/tutorial_reference.ipynb)
is the **single source of truth**. It is the complete notebook with all
contents available. The template tutorial.ipynb and the checkpoints are
automatically generated from it with the following command:

```bash
python make_tutorial.py              # regenerate notebooks
python make_tutorial.py --execute    # execute the reference and regenerate
```

The cell tags control which parts are kept and which ones are removed:

| tag | effect |
| --- | --- |
| `step-N` | which step the cell belongs to |
| `type-along` | empty in the participant notebook and in every checkpoint before its step |
| `given` | already filled in |
| `raises-exception` | code fails on purpose but execution should continue |

Every code cell needs a `step-N` tag and either `type-along` or `given`.
