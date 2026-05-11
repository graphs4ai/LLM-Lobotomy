# AGENT_PROGRESS.md

## Project

LLM-Lobotomy automation for K / optimization-trial sweeps using the existing Hydra-based experiment setup.

This file is the persistent progress ledger for the coding agent. Update it after every meaningful change. Do not rely on chat history as the source of truth.

---

## High-Level Objective

Automate the LLM-Lobotomy experiment pipeline so that controlled sweeps over feature count (`top_k`) and optimization budget (`optimization.n_trials`) can be launched, resumed, summarized, and audited without manually copying W&B artifact names between stages.

The immediate scientific question is whether the inconclusive results under the new three-way split are caused by:

1. insufficient optimization budget;
2. too many optimized intervention features;
3. lack of generalization across the new split;
4. mismatch between the continuous soft IPI objective and the final discrete Likert/IPI metric.

---

## Current Starting Point

The project already uses Hydra for experiment configuration. Do not replace Hydra.

Known pipeline stages:

1. `src/extract_activations.py`
   - extracts residual-stream activations;
   - produces activation artifacts.

2. `src/train_eval_svc.py`
   - performs feature selection / ranking;
   - consumes activation artifacts;
   - produces feature-ranking artifacts.

3. `src/optimize_intervention.py`
   - optimizes activation multipliers;
   - consumes feature-ranking artifacts;
   - produces multiplier / intervention artifacts.

4. `src/likert_scale_test.py`
   - evaluates baseline and intervened IPI / Likert outputs;
   - consumes multiplier artifacts for intervened runs.

5. `src/poeta_evaluator.py`
   - evaluates capability preservation;
   - out of scope for the first automation pass.

Current bottleneck:

```text
activation artifact -> feature selection
feature ranking artifact -> optimization
multiplier artifact -> intervened Likert evaluation
```

These artifact dependencies are currently too manual. The automation should focus on deterministic naming, metadata-based resolution, local manifests, sweep expansion, and summary generation.

---

## Non-Negotiable Constraints

- Do not remove Hydra.
- Do not introduce Snakemake, Luigi, Airflow, or another workflow engine unless explicitly requested.
- Do not rewrite the core experiment logic unless necessary.
- Preserve existing script entrypoints whenever possible.
- Preserve existing behavior when `top_k=80` and `optimization.n_trials=3000`.
- Do not hardcode model names, artifact names, or split paths inside Python code when they can come from Hydra.
- Do not use the final IPI test split to select `top_k`, `n_trials`, bounds, feature-selection settings, or any other hyperparameter.
- Do not automate PoETa first.
- Do not run expensive full experiments during implementation.
- Use dry-runs and tiny smoke tests before launching real sweeps.
- Do not commit W&B API keys, secrets, local cache paths, or machine-specific absolute paths.
- Every patch should be small enough to review.

---

## Target Experiment Matrix

### Phase 1: Budget test for current K

Purpose: determine whether `K=80` failed because `n_trials=500` was too small.

```yaml
models:
  - gemma-3-4b

directions:
  - minimize

feature_counts:
  - 80

trial_grid:
  "80": [500, 1500, 3000]
```

### Phase 2: Smaller-K sweep

Purpose: determine whether fewer intervention dimensions improve optimization and/or generalization.

```yaml
models:
  - gemma-3-4b

directions:
  - minimize

feature_counts:
  - 4
  - 8
  - 16
  - 32

trial_grid:
  "4": [500]
  "8": [500, 1000]
  "16": [1000, 2000]
  "32": [2000, 3000]
```

### Phase 3: Expansion

Only after Phases 1 and 2 work:

```text
models = all target models
directions = minimize, maximize
```

---

## Required Hydra Fields

Add or normalize these fields in the Hydra config tree.

```yaml
data:
  split_id: null
  feature_selection_dataset: null
  optimization_dataset: null
  validation_dataset: null
  ipi_test_dataset: null

feature_selection:
  method: anova_mrmr
  ranking_top_n: 256
  seed: 42

optimization:
  direction: minimize
  top_k: 80
  n_trials: 500
  seed: 42
  objective_mode: soft_ipi
  validation_fraction: null
  split_seed: 42

likert:
  condition: baseline
  multiplier_artifact_name: null
  prompt_template_version: default
  parser_version: default
  temperature: 0
  decoding_strategy: greedy

artifacts:
  activations_name: null
  feature_ranking_name: null
  multiplier_name: null
  likert_baseline_name: null
  likert_intervened_name: null

pipeline:
  dry_run: true
  resume: true
  force: false
  skip_existing: true
```

Default `pipeline.dry_run` should be `true` until execution mode is tested.

---

## Progress Checklist

### Stage 0 — Repository inspection

- [x] Inspect `config/config.yaml`.
- [x] Inspect `config/model/*.yaml`.
- [x] Inspect `src/extract_activations.py`.
- [x] Inspect `src/train_eval_svc.py`.
- [x] Inspect `src/optimize_intervention.py`.
- [x] Inspect `src/likert_scale_test.py`.
- [x] Inspect `src/poeta_evaluator.py`.
- [x] Document current Hydra fields.
- [x] Document current W&B artifact dependencies.
- [x] Document minimal files requiring changes.

Status notes:

```text
Completed on 2026-05-08.

Current Hydra field inventory (high-level):
- Base config: data.feature_selection_statements, data.optimization_statements, training.*, feature_selection.*, ipi_eval.*, wandb.*, random_state.
- Model overlays: model.*, extraction.*, optimization.* (including feature_artifact_name, target_neuron_count, n_trials, direction), ipi_eval.multiplier_artifact_name, poeta.*.
- PoETa entrypoint also consumes likert.* in current code path.

Current W&B artifact dependencies:
- extract_activations.py: optional dataset artifact input (cfg.data.dataset_artifact_name) and logs activations artifact.
- train_eval_svc.py: consumes activations artifact (cfg.data.activations_artifact_name) and logs svm feature ranking artifact CSV.
- optimize_intervention.py: consumes feature ranking artifact (cfg.optimization.feature_artifact_name) and logs multipliers artifact.
- likert_scale_test.py: consumes multipliers artifact (cfg.ipi_eval.multiplier_artifact_name) and logs baseline/comparison evaluation artifacts.
- poeta_evaluator.py: may consume likert.multiplier_artifact_name and logs PoETa evaluation artifacts.

Minimal files requiring immediate Stage 1 changes:
- src/optimize_intervention.py
- config/config.yaml
```

---

### Stage 1 — Make `top_k` and `n_trials` configurable

Goal: `src/optimize_intervention.py` must read optimization dimensionality and trial budget from Hydra.

Tasks:

- [x] Add or normalize `optimization.top_k`.
- [x] Add or normalize `optimization.n_trials`.
- [x] Add or normalize `optimization.direction`.
- [x] Add or normalize `optimization.seed`.
- [x] Replace any hardcoded `80` used as final intervention feature count.
- [x] Replace any hardcoded `3000` used as optimization trial count.
- [x] Validate `top_k > 0`.
- [x] Validate `n_trials > 0`.
- [x] Validate `direction in {"minimize", "maximize"}`.
- [x] Preserve behavior for `top_k=80`, `n_trials=3000`.

Acceptance command:

```bash
python src/optimize_intervention.py --config-name gemma-3-4b \
  optimization.direction=minimize \
  optimization.top_k=16 \
  optimization.n_trials=1000
```

Tiny smoke-test command, if safe:

```bash
python src/optimize_intervention.py --config-name gemma-3-4b \
  optimization.direction=minimize \
  optimization.top_k=4 \
  optimization.n_trials=5
```

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- optimize_intervention now reads optimization.top_k (fallback to optimization.target_neuron_count for backward compatibility).
- optimize_intervention now validates top_k, n_trials, and direction with explicit ValueError messages.
- optimize_intervention now reads optimization.seed (fallback to random_state) and uses it for Optuna sampler seeding.
- Added normalized defaults in config/config.yaml for optimization.top_k, optimization.n_trials, optimization.direction, optimization.seed.
- Existing behavior preserved through default/fallback path: top_k=80 and n_trials=3000 remain valid when model overlays provide these values.
```

---

### Stage 2 — Deterministic experiment IDs and artifact names

Goal: add pure utility functions for deterministic run IDs and artifact names.

Create:

```text
src/utils/experiment_ids.py
```

Required functions:

```python
def make_run_id(
    model_name: str,
    split_id: str,
    direction: str | None,
    top_k: int | None,
    n_trials: int | None,
    seed: int,
    condition: str | None = None,
) -> str:
    ...


def make_activation_artifact_name(model_name: str, split_id: str, layers: str) -> str:
    ...


def make_feature_ranking_artifact_name(model_name: str, split_id: str, ranking_top_n: int) -> str:
    ...


def make_multiplier_artifact_name(
    model_name: str,
    split_id: str,
    direction: str,
    top_k: int,
    n_trials: int,
    seed: int,
) -> str:
    ...


def make_likert_artifact_name(
    model_name: str,
    split_id: str,
    condition: str,
    seed: int,
    direction: str | None = None,
    top_k: int | None = None,
    n_trials: int | None = None,
) -> str:
    ...
```

Recommended formats:

```text
{model}__{split_id}__{direction}__k{top_k}__trials{n_trials}__seed{seed}
activations-{model}-{split_id}-{layers}
feature-ranking-{model}-{split_id}-top{ranking_top_n}
multipliers-{model}-{split_id}-{direction}-k{top_k}-trials{n_trials}-seed{seed}
likert-baseline-{model}-{split_id}-seed{seed}
likert-intervened-{model}-{split_id}-{direction}-k{top_k}-trials{n_trials}-seed{seed}
```

Constraints:

- [x] No W&B calls in this file.
- [x] No Hydra dependency in this file.
- [x] Pure deterministic string functions only.
- [x] Add simple unit tests or a `__main__` smoke test if no test framework exists.

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- Added `src/utils/experiment_ids.py` with pure deterministic naming helpers:
  - make_run_id
  - make_activation_artifact_name
  - make_feature_ranking_artifact_name
  - make_multiplier_artifact_name
  - make_likert_artifact_name
- Added input normalization helper for stable names (`/` and spaces converted to `-`).
- Added `__main__` smoke test that prints canonical sample outputs.
- Added `src/utils/__init__.py` to establish utils package.
```

---

### Stage 3 — W&B artifact helper functions

Goal: centralize artifact existence checks, resolution, and logging with metadata.

Create:

```text
src/utils/wandb_artifacts.py
```

Required functions:

```python
def artifact_exists(
    artifact_name: str,
    artifact_type: str | None = None,
) -> bool:
    ...


def resolve_artifact(
    artifact_name: str,
    artifact_type: str | None = None,
    required_metadata: dict | None = None,
) -> str:
    ...


def log_artifact_with_metadata(
    artifact_name: str,
    artifact_type: str,
    files: list[str],
    metadata: dict,
    aliases: list[str] | None = None,
) -> str:
    ...
```

Metadata required for optimization artifacts:

```json
{
  "stage": "optimization",
  "model_name": "gemma-3-4b",
  "split_id": "three_way_split_v1",
  "direction": "minimize",
  "top_k": 80,
  "n_trials": 3000,
  "seed": 42,
  "objective_mode": "soft_ipi",
  "feature_artifact_name": "feature-ranking-gemma-3-4b-three_way_split_v1-top256:latest",
  "optimization_dataset": "data/splits/optimization.csv",
  "validation_dataset": "data/splits/validation.csv"
}
```

Constraints:

- [x] Do not require manual artifact-name copying.
- [x] Validate metadata before reusing artifacts.
- [x] Fail with clear error messages.
- [x] Keep W&B network calls isolated in this file where practical.

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- Added `src/utils/wandb_artifacts.py` with:
  - `artifact_exists(artifact_name, artifact_type=None) -> bool`
  - `resolve_artifact(artifact_name, artifact_type=None, required_metadata=None) -> str`
  - `log_artifact_with_metadata(artifact_name, artifact_type, files, metadata, aliases=None) -> str`
- Added optimization metadata contract validation for stage="optimization" artifacts.
- Added explicit metadata mismatch checks during artifact resolution.
- Added clear, fail-fast error handling for unresolved artifacts, missing files, missing metadata, and uninitialized wandb runs.
- W&B API/network interactions are centralized in this helper module.
```

---

### Stage 4 — Ranked feature artifact independent of final K

Goal: `src/train_eval_svc.py` must emit a ranked feature artifact with at least `feature_selection.ranking_top_n` entries.

Tasks:

- [x] Add or normalize `feature_selection.ranking_top_n`, default `256`.
- [x] Preserve current top-80 downstream behavior for compatibility.
- [x] Emit `ranked_features` in the feature artifact.
- [x] Include `rank`, `layer`, `feature`, `score` if available, and `selection_frequency` if available.
- [x] Include metadata: `model_name`, `split_id`, `feature_selection_dataset`, `method`, `ranking_top_n`, `seed`.
- [x] Ensure existing consumers do not break.

Expected artifact structure:

```json
{
  "model_name": "...",
  "split_id": "...",
  "feature_selection_dataset": "...",
  "method": "anova_mrmr",
  "ranking_top_n": 256,
  "ranked_features": [
    {
      "rank": 1,
      "layer": 12,
      "feature": 2381,
      "score": 0.123,
      "selection_frequency": 3
    }
  ]
}
```

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- Updated `train_eval_svc.py` to produce a full, stable feature ranking over all candidate features (including zero-frequency entries), then slice by `feature_selection.ranking_top_n`.
- Added artifact payload JSON (`feature_ranking.json`) containing:
  - model_name
  - split_id
  - feature_selection_dataset
  - method
  - ranking_top_n
  - seed
  - ranked_features[] with rank, layer, feature, score (nullable), selection_frequency, selection_count, feature_name
- Kept existing `feature_ranking.csv` artifact output to avoid breaking current downstream consumers.
- Added enhanced artifact metadata fields required by the stage while preserving prior metadata.
- Added `feature_selection.method: anova_mrmr` normalization in `config/config.yaml`.
```

---

### Stage 5 — Optimization slices `top_k` from ranked features

Goal: `src/optimize_intervention.py` should select features by slicing the ranked feature artifact.

Required behavior:

```python
selected_features = ranked_features[:cfg.optimization.top_k]
```

Tasks:

- [x] Load `ranked_features` from the feature artifact.
- [x] Validate `cfg.optimization.top_k <= len(ranked_features)`.
- [x] Preserve compatibility with older feature artifacts if feasible.
- [x] Record `top_k`, `n_trials`, `direction`, `seed`, `split_id`, and dataset paths in W&B metadata.
- [x] Do not change the soft IPI objective.
- [x] Do not change the intervention formula.

Acceptance commands:

```bash
python src/optimize_intervention.py --config-name gemma-3-4b \
  optimization.direction=minimize \
  optimization.top_k=4 \
  optimization.n_trials=5

python src/optimize_intervention.py --config-name gemma-3-4b \
  optimization.direction=minimize \
  optimization.top_k=80 \
  optimization.n_trials=5
```

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- Updated `optimize_intervention.py` to consume `feature_ranking.json` and load `ranked_features`.
- Implemented explicit deterministic slicing:
  `selected_features = ranked_features[:top_k]`
- Added hard validation:
  - fails when `feature_ranking.json` is missing
  - fails when `ranked_features` is malformed
  - fails when `top_k > len(ranked_features)`
- Mapped selected ranked features to neuron keys using `feature_name` when present, else `layer`+`feature`.
- Kept objective and intervention logic unchanged.
- Extended multipliers artifact metadata and W&B summary with:
  - top_k, n_trials, direction, seed, split_id
  - feature_artifact_name
  - optimization_dataset (resolved path)
  - validation_dataset (resolved path)

Compatibility note:
- Based on project decision (no legacy compatibility requirement), optimization now requires new Stage-4 artifact payload (`feature_ranking.json`) and intentionally does not fallback to old CSV-only artifacts.
```

---

### Stage 6 — Hydra experiment configs for sweeps

Goal: add experiment configs for K/trial sweeps.

Create:

```text
config/experiment/k80_trials.yaml
config/experiment/small_k_trials.yaml
```

`config/experiment/k80_trials.yaml`:

```yaml
name: k80_trials
split_id: three_way_split_v1
seed: 42

models:
  - gemma-3-4b

directions:
  - minimize

feature_counts:
  - 80

trial_grid:
  "80": [500, 1500, 3000]

stages:
  extract_activations: true
  feature_selection: true
  optimization: true
  likert_baseline: true
  likert_intervened: true
  poeta: false
```

`config/experiment/small_k_trials.yaml`:

```yaml
name: small_k_trials
split_id: three_way_split_v1
seed: 42

models:
  - gemma-3-4b

directions:
  - minimize

feature_counts:
  - 4
  - 8
  - 16
  - 32

trial_grid:
  "4": [500]
  "8": [500, 1000]
  "16": [1000, 2000]
  "32": [2000, 3000]

stages:
  extract_activations: false
  feature_selection: false
  optimization: true
  likert_baseline: true
  likert_intervened: true
  poeta: false
```

Tasks:

- [x] Add `experiment` config group only if compatible with current Hydra structure.
- [x] Do not break existing `--config-name gemma-3-4b` usage.
- [x] Preserve the current model-selection convention unless refactoring is necessary.

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- Added optional Hydra config group entry in `config/config.yaml`:
  - `defaults: - experiment: null`
- Added sweep experiment configs:
  - `config/experiment/k80_trials.yaml`
  - `config/experiment/small_k_trials.yaml`
- Kept configurations aligned with project decision to use `model=<name>` convention in composed runs.
- Verified compatibility and composition using dry config rendering (no execution):
  - `python src/extract_activations.py experiment=k80_trials model=gemma-3-4b --cfg job`
  - `python src/extract_activations.py experiment=small_k_trials model=gemma-3-4b --cfg job`
- Both commands resolved successfully and included merged `experiment.*` subtree without running expensive jobs.
```

---

### Stage 7 — Dry-run pipeline runner

Goal: create a Hydra-driven orchestrator that initially only prints planned commands and writes planned manifests.

Create:

```text
src/run_pipeline.py
```

Required behavior:

- [x] Load one Hydra experiment config.
- [x] Expand `model x direction x top_k x n_trials` into concrete jobs.
- [x] Generate deterministic run IDs.
- [x] Generate deterministic artifact names.
- [x] Print planned commands.
- [x] Write `runs/pipeline/{run_id}/manifest.json` with `status = planned`.
- [x] Do not execute commands in the first implementation pass.
- [x] Do not call W&B in the first implementation pass unless needed for validation.

Expected command:

```bash
python src/run_pipeline.py experiment=k80_trials pipeline.dry_run=true
```

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- Added `src/run_pipeline.py` (Hydra-driven planner, dry-run behavior only).
- Implemented experiment matrix expansion:
  model x direction x top_k x n_trials.
- Integrated deterministic naming using `src/utils/experiment_ids.py`.
- Prints planned stage commands and writes per-job manifests under:
  `runs/pipeline/{run_id}/manifest.json`.
- Manifests include: run metadata, commands, artifacts, metrics placeholders, and `status="planned"`.
- No subprocess execution and no W&B usage in this stage.

Validation command:
- `python src/run_pipeline.py experiment=k80_trials pipeline.dry_run=true`
  - Planned 3 jobs and wrote manifests as expected.
```

---

### Stage 8 — Manifest schema and resumability

Goal: use local manifests as the pipeline source of truth for planned, running, completed, skipped, and failed jobs.

Manifest path:

```text
runs/pipeline/{run_id}/manifest.json
```

Required manifest shape:

```json
{
  "run_id": "gemma-3-4b__three_way_split_v1__minimize__k80__trials3000__seed42",
  "status": "planned",
  "model_name": "gemma-3-4b",
  "split_id": "three_way_split_v1",
  "direction": "minimize",
  "top_k": 80,
  "n_trials": 3000,
  "seed": 42,
  "commands": [],
  "artifacts": {
    "activations": "activations-gemma-3-4b-three_way_split_v1-all:latest",
    "feature_ranking": "feature-ranking-gemma-3-4b-three_way_split_v1-top256:latest",
    "multipliers": "multipliers-gemma-3-4b-three_way_split_v1-minimize-k80-trials3000-seed42:latest",
    "likert_baseline": "likert-baseline-gemma-3-4b-three_way_split_v1-seed42:latest",
    "likert_intervened": "likert-intervened-gemma-3-4b-three_way_split_v1-minimize-k80-trials3000-seed42:latest"
  },
  "metrics": {
    "soft_ipi_optimization_baseline": null,
    "soft_ipi_optimization_intervened": null,
    "delta_soft_ipi_optimization": null,
    "soft_ipi_validation_baseline": null,
    "soft_ipi_validation_intervened": null,
    "delta_soft_ipi_validation": null,
    "discrete_ipi_test_baseline": null,
    "discrete_ipi_test_intervened": null,
    "delta_discrete_ipi_test": null,
    "wilcoxon_p_value": null
  },
  "error": null
}
```

Statuses:

```text
planned
running
completed
skipped
failed
```

Resume rules:

- [x] If `pipeline.resume=true` and manifest status is `completed`, skip the job.
- [x] If `pipeline.force=true`, rerun regardless of manifest status.
- [x] If a command fails, write `status = failed` and store the error message.

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- Extended `src/run_pipeline.py` to use manifests as planning source-of-truth:
  - reads existing `runs/pipeline/{run_id}/manifest.json` when present;
  - if `resume=true` and existing status is `completed` (and not `force`), job is skipped;
  - if `force=true`, job is replanned regardless of prior status.
- Added fail-safe per-job planning guard:
  - on planning exception, writes manifest with `status="failed"` and error message.
- Added plan summary counters:
  - planned jobs
  - skipped jobs
  - failed jobs

Validation commands:
- `python src/run_pipeline.py experiment=k80_trials pipeline.dry_run=true`
- `python src/run_pipeline.py experiment=k80_trials pipeline.dry_run=true pipeline.resume=true pipeline.force=true`
- Resume skip rule smoke test:
  - temporarily set one manifest status to `completed`
  - reran with `pipeline.resume=true pipeline.force=false`
  - observed expected skip
  - restored test manifest status afterward
```

---

### Stage 9 — Execution mode

Goal: allow `src/run_pipeline.py` to execute planned commands after dry-run behavior is validated.

Tasks:

- [x] Keep `pipeline.dry_run=true` as the safe default.
- [x] When `pipeline.dry_run=false`, execute commands via `subprocess.run(..., check=True)`.
- [x] Mark manifests as `running` before execution.
- [x] Mark manifests as `completed` after successful execution.
- [x] Mark manifests as `failed` on error.
- [x] Respect `pipeline.resume`, `pipeline.force`, and `pipeline.skip_existing`.

Acceptance command:

```bash
python src/run_pipeline.py experiment=k80_trials pipeline.dry_run=false pipeline.resume=true
```

Do not run this until the dry-run output has been manually inspected.

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- Extended `src/run_pipeline.py` with execution mode:
  - when `pipeline.dry_run=false`, commands execute sequentially via:
    `subprocess.run(command, shell=True, check=True, cwd=<project_root>)`.
- Added manifest status transitions:
  - `planned` -> `running` before command execution;
  - `running` -> `completed` after all commands succeed;
  - `failed` with captured error message if any command raises.
- Added explicit skip logic helper that respects:
  - `pipeline.resume`
  - `pipeline.force`
  - `pipeline.skip_existing`
  for previously completed manifests.
- Preserved `pipeline.dry_run=true` safety default behavior (plan + write manifests, no execution).

Validation performed (safe):
- `python -m py_compile src/run_pipeline.py`
- `python src/run_pipeline.py experiment=k80_trials pipeline.dry_run=true pipeline.resume=true pipeline.force=false pipeline.skip_existing=true`

Execution-mode command (`pipeline.dry_run=false`) intentionally not run here to avoid launching expensive jobs before manual operator approval.
```

---

### Stage 10 — Baseline Likert reuse

Goal: avoid rerunning baseline Likert for every `top_k`, `n_trials`, and `direction`.

A baseline Likert run is reusable for fixed:

```text
model
split_id
ipi_test_dataset
prompt_template_version
parser_version
temperature
decoding_strategy
seed
```

Tasks:

- [x] Generate one baseline Likert command per model/split.
- [x] Generate one intervened Likert command per multiplier artifact.
- [x] Store baseline artifact reference in every job manifest.
- [x] Validate that baseline metadata matches before reuse.

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- Updated `src/run_pipeline.py` to de-duplicate baseline Likert scheduling with a baseline reuse key.
- Baseline reuse key fields:
  - model
  - split_id
  - ipi_test_dataset (fallback to ipi_eval.questions_csv)
  - likert.prompt_template_version
  - likert.parser_version
  - likert.temperature
  - likert.decoding_strategy
  - seed
- Behavior:
  - baseline command scheduled only once per unique baseline key;
  - intervened command remains per concrete multiplier job (direction/top_k/n_trials);
  - every job manifest still includes `artifacts.likert_baseline` reference.
- Validation via dry-run showed exactly one baseline command for `small_k_trials` and reuse messaging on remaining jobs.
```

---

### Stage 11 — Validation metrics in optimization

Goal: distinguish optimization failure, overfitting, and soft/discrete mismatch.

Tasks:

- [x] Add `data.optimization_dataset`.
- [x] Add `data.validation_dataset`.
- [x] Add optional `optimization.validation_fraction`.
- [x] Add optional `optimization.split_seed`.
- [x] Report optimization soft metrics.
- [x] Report validation soft metrics.
- [x] Store these metrics in W&B and/or local manifest.

Required metrics:

```text
soft_ipi_optimization_baseline
soft_ipi_optimization_intervened
delta_soft_ipi_optimization
soft_ipi_validation_baseline
soft_ipi_validation_intervened
delta_soft_ipi_validation
```

Interpretation guide:

```text
soft optimization does not move:
    likely optimization failure, bad bounds, too few trials, or intervention bug.

soft optimization moves but soft validation does not:
    likely overfitting to optimization split.

soft validation moves but discrete IPI does not:
    likely soft/discrete mismatch.

discrete optimization moves but discrete test does not:
    likely lack of held-out generalization.
```

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- Added normalized data fields in `config/config.yaml`:
  - data.split_id
  - data.feature_selection_dataset
  - data.optimization_dataset
  - data.validation_dataset
  - data.ipi_test_dataset
- Updated `src/optimize_intervention.py` to load:
  - optimization questions from `data.optimization_dataset` (fallback legacy field)
  - validation questions from `data.validation_dataset` (fallback legacy path)
- Added soft-score evaluation for both baseline and best-intervened multipliers on:
  - optimization dataset
  - validation dataset
- Computed and recorded required metrics:
  - soft_ipi_optimization_baseline
  - soft_ipi_optimization_intervened
  - delta_soft_ipi_optimization
  - soft_ipi_validation_baseline
  - soft_ipi_validation_intervened
  - delta_soft_ipi_validation
- Persisted metrics to:
  - W&B summary
  - optimization artifact metadata
  - local optimization results JSON (`soft_metrics` section)
- Kept objective/intervention behavior unchanged (metrics are additional reporting).

Validation commands:
- `python -m py_compile src/optimize_intervention.py src/run_pipeline.py src/train_eval_svc.py`
- `python src/optimize_intervention.py model=gemma-3-4b --cfg job`
```

---

### Stage 11.5 — Manifest metric persistence + W&B backfill

Goal: ensure executed pipeline jobs persist their metrics into the local manifest, and provide a standalone backfill path that reconstructs already-completed manifests from W&B without rerunning heavy stages.

Bug:

- `src/run_pipeline.py` wrote manifests with `metrics = _null_metrics()` at plan time and never updated them after `_execute_job_commands` flipped the status to `completed`. All metric keys stayed `null` on disk even though W&B held the correct values.
- `src/likert_scale_test.py` hardcoded `"likert-baseline-results"` and `"likert-comparison-results"` for its W&B artifacts, ignoring `cfg.artifacts.likert_baseline_name` and `cfg.artifacts.likert_intervened_name` set by the pipeline orchestrator. The manifest's `artifacts.likert_*` references therefore did not exist in W&B.

Tasks:

- [x] Add `src/utils/metrics_backfill.py` exposing `NULL_METRICS`, `MetricsBackfillError`, `metrics_are_complete`, and `collect_run_metrics(manifest, project, entity=None)`.
 - Soft optimization/validation metrics are read from the multipliers W&B artifact metadata (deterministic name).
 - Discrete IPI / Wilcoxon metrics are read from the intervened Likert W&B run located via `config.multiplier_artifact_name` / `config.ipi_eval.multiplier_artifact_name`, picking the newest match.
- [x] Update `src/run_pipeline.py` so `_execute_job_commands` calls `collect_run_metrics` after a successful execution and merges the result into the completed manifest. Failures during write-back log a warning and leave nulls in place but do not fail the run.
- [x] Add `src/backfill_manifests.py` standalone CLI (`--project`, `--entity`, `--dry-run`, `--force`, `--run-id <glob>`, `--pipeline-dir`). Walks `runs/pipeline/*/manifest.json`, reuses `collect_run_metrics`, and rewrites manifests in place. By default skips non-completed and already-complete manifests; `--force` overrides.
- [x] Update `src/likert_scale_test.py` to honor `cfg.artifacts.likert_baseline_name` and `cfg.artifacts.likert_intervened_name` (fallback to legacy hardcoded names). Mirror `multiplier_artifact_name` into `wandb.summary` so backfill's run filter has a stable indexable surface.

Validation commands (operator-run):

```bash
python src/backfill_manifests.py --project LLM-Lobotomy --dry-run
python src/backfill_manifests.py --project LLM-Lobotomy
python src/summarize_sweep.py
```

Status notes:

```text
Completed on 2026-05-11.

Implementation notes:
- collect_run_metrics returns a 10-key dict aligned with the manifest schema; None for unresolved values.
- The W&B artifact lookup normalizes bare names to `[entity/]project/name:alias` automatically.
- The pipeline runtime write-back is best-effort: a W&B fetch failure prints a warning and the manifest stays nullable, so the operator can rerun the backfill script later.
- Likert artifact-name fix is backward-compatible: scripts invoked without `artifacts.*` overrides keep the legacy `likert-baseline-results` / `likert-comparison-results` names.
```

---

### Stage 12 — Sweep summary generation

Goal: generate CSV and Markdown summaries from local manifests.

Create:

```text
src/summarize_sweep.py
```

Required behavior:

- [x] Scan `runs/pipeline/*/manifest.json`.
- [x] Write `runs/pipeline/summary.csv`.
- [x] Write `runs/pipeline/summary.md`.
- [x] Do not depend on W&B in the first implementation pass.
- [x] Include failed and skipped jobs in the summary.

Minimum columns:

```text
run_id
model_name
split_id
direction
top_k
n_trials
seed
activation_artifact
feature_ranking_artifact
multiplier_artifact
likert_baseline_artifact
likert_intervened_artifact
soft_ipi_optimization_baseline
soft_ipi_optimization_intervened
delta_soft_ipi_optimization
soft_ipi_validation_baseline
soft_ipi_validation_intervened
delta_soft_ipi_validation
discrete_ipi_test_baseline
discrete_ipi_test_intervened
delta_discrete_ipi_test
wilcoxon_p_value
status
error
```

Questions the summary must answer:

1. Does `K=80` improve when `n_trials` increases from 500 to 1500 to 3000?
2. Do smaller K values produce more stable shifts than `K=80`?
3. Does the soft objective move even when the discrete IPI does not?
4. Is the failure mode optimization failure, overfitting, soft/discrete mismatch, or lack of held-out generalization?

Status notes:

```text
Completed on 2026-05-08.

Implementation notes:
- Added `src/summarize_sweep.py`.
- Implemented manifest scan using local files only:
  - `runs/pipeline/*/manifest.json`
- Added flattening logic from manifest shape to required summary columns.
- Writes:
  - `runs/pipeline/summary.csv`
  - `runs/pipeline/summary.md`
- Includes all discovered manifest statuses (including failed/skipped) in outputs.
- No W&B dependency used in this first-pass summary implementation.

Validation commands:
- `python -m py_compile src/summarize_sweep.py`
- `python src/summarize_sweep.py`
  - Result: scanned 12 manifests and generated both summary files.
```

---

## Fail-Fast Validation Checklist

Before launching expensive jobs, validate:

- [ ] Dataset paths exist.
- [ ] `top_k <= feature_selection.ranking_top_n`.
- [ ] `n_trials > 0`.
- [ ] `direction in {"minimize", "maximize"}`.
- [ ] `data.optimization_dataset != data.ipi_test_dataset`.
- [ ] If strict three-way split is enabled, `data.feature_selection_dataset != data.optimization_dataset`.
- [ ] If strict three-way split is enabled, `data.feature_selection_dataset != data.ipi_test_dataset`.
- [ ] W&B is authenticated when W&B logging/resolution is enabled.
- [ ] Baseline Likert is not scheduled repeatedly for the same model/split/settings.
- [ ] Feature-ranking artifact contains enough ranked features for requested `top_k`.
- [ ] Multiplier artifact metadata matches `model_name`, `split_id`, `direction`, `top_k`, `n_trials`, and `seed` before reuse.

---

## Caching and Reuse Rules

### Activation extraction

Reuse only if metadata match:

```text
model_name
split_id
feature_selection_dataset
layers
tokenizer
hook_location
dtype
```

### Feature selection

Reuse only if metadata match:

```text
activation_artifact_name
feature_selection_method
ranking_top_n
seed
split_id
```

### Optimization

Reuse only if metadata match:

```text
feature_ranking_artifact_name
model_name
split_id
direction
top_k
n_trials
objective_mode
bounds
seed
optimization_dataset
validation_dataset
```

### Likert baseline

Reuse only if metadata match:

```text
model_name
split_id
ipi_test_dataset
condition=baseline
prompt_template_version
parser_version
temperature
decoding_strategy
seed
```

### Likert intervened

Reuse only if metadata match:

```text
model_name
split_id
ipi_test_dataset
condition=intervened
multiplier_artifact_name
direction
top_k
n_trials
prompt_template_version
parser_version
temperature
decoding_strategy
seed
```

---

## Commands to Preserve or Support

Budget test:

```bash
python src/run_pipeline.py experiment=k80_trials pipeline.resume=true
```

Small-K test:

```bash
python src/run_pipeline.py experiment=small_k_trials pipeline.resume=true
```

Dry run:

```bash
python src/run_pipeline.py experiment=k80_trials pipeline.dry_run=true
```

Override models and directions:

```bash
python src/run_pipeline.py experiment=k80_trials \
  experiment.models='[gemma-3-4b,qwen-3-8b]' \
  experiment.directions='[minimize,maximize]'
```

Summary:

```bash
python src/summarize_sweep.py
```

---

## Acceptance Criteria

The implementation is complete when these commands work:

```bash
python src/run_pipeline.py experiment=k80_trials pipeline.resume=true
python src/run_pipeline.py experiment=small_k_trials pipeline.resume=true
python src/summarize_sweep.py
```

Expected output structure:

```text
runs/pipeline/
  gemma-3-4b__three_way_split_v1__minimize__k80__trials500__seed42/
    manifest.json
  gemma-3-4b__three_way_split_v1__minimize__k80__trials1500__seed42/
    manifest.json
  gemma-3-4b__three_way_split_v1__minimize__k80__trials3000__seed42/
    manifest.json
  gemma-3-4b__three_way_split_v1__minimize__k4__trials500__seed42/
    manifest.json
  gemma-3-4b__three_way_split_v1__minimize__k8__trials500__seed42/
    manifest.json
  gemma-3-4b__three_way_split_v1__minimize__k8__trials1000__seed42/
    manifest.json
  gemma-3-4b__three_way_split_v1__minimize__k16__trials1000__seed42/
    manifest.json
  gemma-3-4b__three_way_split_v1__minimize__k16__trials2000__seed42/
    manifest.json
  gemma-3-4b__three_way_split_v1__minimize__k32__trials2000__seed42/
    manifest.json
  gemma-3-4b__three_way_split_v1__minimize__k32__trials3000__seed42/
    manifest.json
  summary.csv
  summary.md
```

---

## Local Agent Operating Protocol

Use the local coding agent iteratively.

For each stage:

1. Read this file.
2. Inspect only the files relevant to the current stage.
3. Make the smallest viable patch.
4. Run the smallest safe validation command.
5. Update this file.
6. Stop and report:
   - files changed;
   - commands run;
   - result;
   - unresolved questions.

Do not attempt to implement all stages in a single pass.

---

## Decision Log

Record implementation decisions here.

| Date | Decision | Rationale | Files affected |
|---|---|---|---|
| 2026-05-11 | Persist manifest metrics by reading W&B after job execution and via a standalone backfill script | Closes the gap where executed jobs left manifest metrics null even though W&B had the values; matches the stated dual-persistence decision (manifest + W&B) | src/run_pipeline.py, src/utils/metrics_backfill.py, src/backfill_manifests.py |
| 2026-05-11 | Use the multipliers W&B artifact metadata as the canonical source for soft optimization/validation metrics during backfill | The multipliers artifact name is deterministic and respected by `optimize_intervention.py`, so it maps 1:1 to a manifest without ambiguity | src/utils/metrics_backfill.py |
| 2026-05-11 | Match intervened Likert W&B runs by `config.multiplier_artifact_name` (with `summary.multiplier_artifact_name` as a mirrored fallback) | Likert artifacts are generic/hardcoded historically; matching on the multipliers reference is the only deterministic per-job key available across past runs | src/utils/metrics_backfill.py, src/likert_scale_test.py |
| 2026-05-11 | Make runtime metric write-back best-effort (warn on failure, never mark the run failed) | The heavy work is already done — a transient W&B query failure should not flip an otherwise valid run, and the standalone backfill script can recover later | src/run_pipeline.py |
| 2026-05-11 | Honor `artifacts.likert_baseline_name` and `artifacts.likert_intervened_name` in likert_scale_test.py, with backward-compatible fallback | Makes the orchestrator's deterministic identity thread end-to-end for Likert outputs while preserving ad-hoc usage | src/likert_scale_test.py |
| 2026-05-08 | Keep summary generation fully local in first pass (manifest-only) | Avoids extra network coupling and makes sweep recap reproducible from workspace state alone | src/summarize_sweep.py |
| 2026-05-08 | Report soft metrics on both optimization and validation splits for baseline/intervened | Distinguishes optimization failure from overfitting and supports soft/discrete mismatch diagnosis | src/optimize_intervention.py, config/config.yaml |
| 2026-05-08 | Baseline Likert scheduling is keyed by model/split/eval settings | Prevents redundant baseline evaluations while preserving explicit per-job baseline artifact references | src/run_pipeline.py |
| 2026-05-08 | Execute pipeline commands from project root using subprocess in run_pipeline | Keeps relative script paths stable under Hydra and ensures manifest status transitions are atomic per job | src/run_pipeline.py |
| 2026-05-08 | Completed manifests are preserved and skipped under resume mode | Prevents redundant reruns while keeping completed state as durable pipeline truth | src/run_pipeline.py |
| 2026-05-08 | Use model config names (e.g. `gemma-3-4b`) as deterministic pipeline identifiers | Keeps run/artifact IDs stable, short, and aligned with `model=<name>` sweep dimension | src/run_pipeline.py |
| 2026-05-08 | Add `experiment: null` as optional default group in base config | Enables experiment matrix composition while keeping existing script entrypoints backward-safe | config/config.yaml, config/experiment/*.yaml |
| 2026-05-08 | Stage 5 enforces ranked_features JSON contract (no CSV-only fallback) | Aligns with explicit decision to drop old artifact compatibility and avoids ambiguous feature reconstruction | src/optimize_intervention.py |
| 2026-05-08 | Keep CSV artifact alongside new ranked_features JSON payload | Preserves existing optimization consumer path while introducing richer stage-4 artifact contract | src/train_eval_svc.py |
| 2026-05-08 | Enforce strict metadata validation for optimization artifact reuse | Prevents silent reuse of mismatched multipliers/features/datasets across sweep jobs | src/utils/wandb_artifacts.py |
| 2026-05-08 | Deterministic IDs use normalized slug fragments (`/` and spaces to `-`) | Prevents unstable W&B/local paths and keeps artifact IDs reproducible across scripts | src/utils/experiment_ids.py |
| 2026-05-08 | Standardize model selection convention to `model=<name>` for pipeline orchestration | Keeps Hydra composition explicit and consistent with planned experiment overrides | AGENTS.md, upcoming run_pipeline/config updates |
| 2026-05-08 | Drop backward compatibility requirement for older W&B artifacts | Reduces complexity and avoids legacy branching while implementing deterministic artifact contracts | AGENTS.md, upcoming artifact helpers and stage integrations |
| 2026-05-08 | Use dual metrics persistence: local manifest + W&B | Improves local auditability/resume logic while preserving experiment tracking and dashboards | AGENTS.md, upcoming optimization/likert/pipeline stages |
| 2026-05-08 | Keep Stage 1 backward compatibility through fallback `top_k -> target_neuron_count` | Existing model configs currently define `target_neuron_count`; fallback avoids breaking current runs while normalizing new field | src/optimize_intervention.py, config/config.yaml |
| TBD | Use Hydra-driven pipeline runner instead of adding a new workflow engine | Existing project already uses Hydra; current bottleneck is artifact dependency automation | TBD |
| TBD | Keep PoETa automation out of the first pass | Immediate debugging question concerns K/trials/generalization, not capability tax | TBD |
| TBD | Use local manifests as source of truth for resume and summary | Avoid repeated W&B queries and improve auditability | TBD |

---

## Open Questions

Track unresolved questions here.

- [x] Does the current repo use `--config-name gemma-3-4b` as the primary model-selection mechanism, or `model=gemma-3-4b`?  
      **Resolved:** standardize new automation paths on `model=<name>`.
- [ ] Where are current artifact names defined: base config, model configs, script defaults, or generated dynamically?
- [ ] Does `train_eval_svc.py` already save enough information to reconstruct ranked features beyond top 80?
- [ ] Does `optimize_intervention.py` currently load a fixed list of features or a ranked list?
- [ ] Is there already a validation split file, or should optimization support `validation_fraction`?
- [x] Where should metrics be written locally: manifest only, W&B only, or both?  
      **Resolved:** both manifest and W&B.
- [x] Are old W&B artifacts required to remain compatible?  
      **Resolved:** no compatibility requirement for older artifacts.

---

## Last Successful Command

```bash
python -m py_compile src/run_pipeline.py src/backfill_manifests.py src/utils/metrics_backfill.py src/likert_scale_test.py
```

---

## Last Failure

```text
python -m py_compile src/run_pipeline.py
failed once due to malformed dict keys while introducing stage-8 manifest logic.
Fixed keys/call signatures and revalidated dry-run/resume/force behavior.
```

---

## Current Status Summary

```text
Status: in progress
Current stage: Stage 11.5 — manifest metric persistence + backfill landed.
Next action: operator runs `python src/backfill_manifests.py --project LLM-Lobotomy --dry-run` against existing runs/pipeline manifests, then drops --dry-run, then reruns src/summarize_sweep.py to confirm populated metric columns.
```
