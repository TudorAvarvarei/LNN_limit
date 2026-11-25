# Cart pole size and training budget study

This project trains CfC, LTC, GRU, LSTM, MLP, and RNN controllers with dense and NCP-style wiring, then compares control performance, representation interpretability, size, and training budget. It implements [the research brief](AGENT_IMPLEMENTATION_PROMPT.md).

The pipeline is implemented and a small end-to-end smoke run is saved in `results/smoke`. Those runs validate the software; their tiny training budgets and single seed do not support model rankings.

Local validation: **44 tests passed**, all 24 model/task combinations completed, and a separate bounded MLP run learned balancing control. Details and actual measurements are in [VALIDATION.md](docs/VALIDATION.md).

## Run it

In the existing workspace environment:

```bash
/tmp/recurrent-control-venv/bin/python -m lnn_study plan --config configs/full.yaml
/tmp/recurrent-control-venv/bin/python -m lnn_study run --config configs/smoke.yaml
```

For a new environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m lnn_study run --config configs/smoke.yaml
python -m pytest -q
```

The installed `lnn-study` command is equivalent to `python -m lnn_study`. The tested CPU dependency versions are in [requirements-tested.txt](requirements-tested.txt).

## Train the study

Inspect the grid before launching it:

```bash
lnn-study plan --config configs/pilot.yaml
lnn-study plan --config configs/full.yaml --output results/full_grid_manifest.json
```

The full preset contains **1,440 training trajectories**, ten seeds per cell, five budget checkpoints, and approximately **7.2 billion training transitions**. The pilot uses six seeds, two sizes, and three budgets; it still requires 288 million transitions. Neither runs automatically.

Start with a selected trajectory, or omit the filters to train the entire chosen preset:

```bash
lnn-study train --config configs/pilot.yaml --family gru --wiring dense --task balance --size 12 --seed 0
```

Use `--task swingup` for swing-up and `--wiring ncp` for the structured version. All six families use the same commands. Family names are `cfc`, `ltc`, `gru`, `lstm`, `mlp`, and `rnn`.

Training automatically resumes from the latest checkpoint and skips completed work. `--stop-after-updates 5` provides a bounded interruption check. Each seed is trained once, saving successive budgets along the same trajectory. Add later budgets to continue training; past budgets that were never saved cannot be reconstructed.

## Evaluate and analyze

After training, use the same configuration:

```bash
lnn-study evaluate --config configs/pilot.yaml
lnn-study probe --config configs/pilot.yaml
lnn-study interpret --config configs/pilot.yaml
lnn-study analyze --config configs/pilot.yaml
```

`run` performs all these stages in order. `pareto` reruns the analysis and frontier export. `--run-dir PATH` changes the output directory, and the model/task/size/seed filters also select evaluation and interpretation work. Use `--force` to regenerate their artifacts.

The main outputs are under `results/<preset>/analysis/`:

| File | Contents |
| --- | --- |
| `summary.csv` | Seed means, confidence intervals, completeness and feasibility frequency |
| `per_seed.csv` | Measurements for each training seed and budget |
| `pareto.csv` | Feasible frontiers, separately for steps/wall time and MIG/modularity/tree size |
| `performance_matches.csv` | Comparable-performance pairs and matching support |
| `minimum_sizes.csv` | Minimum sizes meeting declared performance criteria |
| `budget_size_interactions.csv` | Whether smaller networks catch up as budget increases |
| `run_statuses.json` | Completed, missing, interrupted or failed runs |
| `*.png` | Heatmaps, interpretability comparisons and tradeoff plots |

Each run also saves its resolved settings, architecture/mask audit, training log, checkpoints, paired clean/disturbed episodes, shared/on-policy neuron activations, and fitted trees. `analysis/REPORT.md` states how to interpret the outputs.

## Methods and limits

Both tasks use continuous force, full state observations, a 20 ms period, and standard cart-pole physics. The actor uses PPO; a separate critic prevents value gradients from changing its representation. All training settings are explicit in [config.py](src/lnn_study/config.py), with preset overrides in `configs/`.

NCPS provides native CfC/LTC NCP models. The other structured versions are labeled research extensions, and the MLP drops recurrent edges. Dense and structured models differ in depth as well as connectivity. Their difference cannot be attributed solely to sparsity.

Interpretability uses held-out, whole-trajectory splits and common observation histories. MIG/modularity use canonical physical factors; they do **not** implement the papers' calibrated learned tree-leaf vocabulary. See [methods.md](docs/methods.md) for exact metrics, thresholds, seed accounting, and source mappings. Optimizer settings still need a balanced development pilot before scientific architecture conclusions.
