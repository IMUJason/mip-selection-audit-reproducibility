# mip-selection-audit-reproducibility

Reproducibility package for the paper *"The Hidden Cost of Smart Solver
Decisions: An Auditable Cross-Component Study of Selection Policies in
Mixed-Integer Programming"* (Jin Xin Cao).

This repository contains the **input data and core algorithm scripts only**
(the audited branch-and-bound harness, its CPLEX backend adapter, the
experiment drivers, the learned-selector training pipeline, and the result
analysis scripts). Manuscript sources, figures, and figure-generation code
are not part of this package.

## Environment

- Python 3.10, CPLEX 22.1.1 (Python API). The experiments were run with the
  CPLEX educational distribution in a conda environment (`env310`); any
  CPLEX >= 22.1 with Python 3.10 should behave identically apart from
  wall-clock effects documented in the paper.
- Unit tests: `PYTHONPATH=src python -m pytest tests/ -v`

## Layout

- `src/plan4/` — the audited B&B engine and policies
  (`branch_and_bound.py`), the CPLEX backend adapter with documented
  Gurobi-to-CPLEX migration semantics (`cplex_adapter.py`), selectors,
  metrics, provenance.
- `scripts/`
  - `run_grid.py` — main policy grid (16 instances x 8 policies x 4 budgets x 3 seeds)
  - `run_grid_ext.py` — 24-instance replication cohort grid
  - `run_grid_phase2.py` / `run_learned_v2.py` — learned-selector runs
  - `run_cplex_native.py` — solver-native reference baselines
  - `train_learned_v2.py` — full-action-space selector training (LOOCV depth selection)
  - `derive_portfolios.py` — deterministic wrapper-portfolio derivation
  - `analyze_grid.py` / `analyze_anchors.py` — result aggregation, sanity
    invariants, and anchor-mechanism analysis
- `data/` — locked manifests (`dataset_manifest_round1.json`,
  `extension_manifest.json`) with per-instance SHA-256 hashes and official
  optima pointers, and the two frozen selector models (round-27 original and
  learned-v2).
- `tests/` — 19 unit tests covering the backend migration.

## Instances

Benchmark and replication instances are standard MIPLIB 2017 files and are
NOT bundled (size). Both manifests list each instance's file name, SHA-256,
and official optimum; place the `.mps.gz` files under `data/instances/`
(download from [miplib.zib.de](https://miplib.zib.de) or your local MIPLIB
2017 mirror) and verify hashes against the manifests before running.

## Reproducing the experiments

```bash
python scripts/run_grid.py --workers 6          # phase 1 (grid)
python scripts/run_cplex_native.py --workers 4  # native references
python scripts/run_grid_ext.py --workers 6      # replication cohort
python scripts/train_learned_v2.py              # retrains + exports learned-v2
python scripts/run_learned_v2.py --workers 6
python scripts/derive_portfolios.py
python scripts/analyze_grid.py                  # tables-ready CSVs + invariant report
```

All runs write per-run manifests, summaries, and per-step traces under
`results/`; every number in the paper regenerates from these artifacts via
`analyze_grid.py` / `analyze_anchors.py`.

## License

MIT.
