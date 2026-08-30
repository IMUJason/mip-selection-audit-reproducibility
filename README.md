# mip-selection-audit-reproducibility

Reproducibility package for the paper *"Auditing Selection Policies Inside MIP
Solvers: Locked Replication, Charged Accounting, and Solver-Native Baselines
for Node Selection and Root Cut Selection"* (Jin Xin Cao).

This repository contains the complete audit chain referenced in the paper:
the audited branch-and-bound harness with its CPLEX backend adapter, all
experiment drivers, the locked instance manifests, the two frozen selector
models, the per-run result artifacts from which every table and figure
regenerates, the Study-2 (root cut selection) merged run logs with their
manifests and route-audit files, and the verification script that asserts
every paired-comparison number in the paper's Study-2 table.

## Environment

- Python 3.10, CPLEX 22.1.1 (Python API). The experiments were run with the
  CPLEX educational distribution in a conda environment (`env310`); any
  CPLEX >= 22.1 with Python 3.10 should behave identically apart from
  wall-clock effects documented in the paper.
- Study-2 raw logs were produced with SCIP 10.0.1 via PySCIPOpt 6.1.0 (the
  Study-2 harness itself is not part of this package; its released outputs
  and the downstream audit/verification code are).
- Unit tests: `PYTHONPATH=src python -m pytest tests/ -v` (19 tests covering
  the backend migration).

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
  - `make_paper_tables.py` — all Study-1 tables from `results/analysis/*.csv`
  - `make_extension_table.py` — replication-cohort table from the ledger and manifest
  - `make_figures.py` — the three paper figures (budget curves, anchor
    mechanism, replication scatter)
  - `verify_study2_external.py` — asserts every row of the Study-2 paired
    table (all ten comparisons, both estimands) against the released merged
    logs; exits non-zero on any mismatch
- `data/` — locked manifests (`dataset_manifest_round1.json`,
  `extension_manifest.json`) with per-instance SHA-256 hashes and official
  optima pointers, and the two frozen selector models (round-27 original and
  learned-v2).
- `results/`
  - `grid/`, `grid_ext/` — run-level ledgers plus per-run summary JSONs for
    the main grid, phase-2, learned-v2, and replication cohort
  - `cplex_native/` — per-run JSONs of the solver-native reference runs
  - `analysis/` — tables-ready CSVs produced by `analyze_grid.py` /
    `analyze_anchors.py` (plus the sanity-invariant report)
  - `study2/` — Study-2 (root cut selection in SCIP): per-split merged result
    logs (`results_merged.jsonl`), per-round logs (`rounds_merged.jsonl`),
    run registries, shard manifests, locked instance manifests, and the
    route-audit files (`route_audit/`), covering the benchmark hold-out
    (h140), the external confirmatory split (c140), the development split
    (d120, two run directories), and the untouched external hold-out (h120)
- `tests/` — 19 unit tests covering the backend migration.

## Instances

Benchmark and replication instances are standard MIPLIB 2017 files and are
NOT bundled (size). Both Study-1 manifests list each instance's file name,
SHA-256, and official optimum; place the `.mps.gz` files under
`data/instances/` (download from [miplib.zib.de](https://miplib.zib.de) or
your local MIPLIB 2017 mirror) and verify hashes against the manifests before
running.

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

To regenerate the paper's tables and figures from the released artifacts
without re-running the grids:

```bash
python scripts/analyze_grid.py        # only if you want to rebuild results/analysis/
python scripts/make_paper_tables.py   # Study-1 tables -> paper/tables/
python scripts/make_extension_table.py
python scripts/make_figures.py        # figures -> paper/figures/
python scripts/verify_study2_external.py   # asserts the Study-2 table, all rows
```

Note on scope: per-step decision traces (the multi-hundred-MB
`*_trace.jsonl` files) are regenerated by re-running the grids and are not
bundled; the run-level ledgers and per-run summaries shipped here are the
inputs every paper number derives from.

## License

MIT.
