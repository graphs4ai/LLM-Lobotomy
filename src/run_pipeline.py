from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import Any
import hydra
from omegaconf import DictConfig

"""Hydra-driven pipeline planner (dry-run first pass)."""


from utils.experiment_ids import (
    make_activation_artifact_name,
    make_feature_ranking_artifact_name,
    make_likert_artifact_name,
    make_multiplier_artifact_name,
    make_run_id,
)
from utils.intervention_hooks import DEFAULT_LAST_K, DEFAULT_SCOPE, assert_scope
from utils.metrics_backfill import (
    NULL_METRICS,
    MetricsBackfillError,
    collect_run_metrics,
)


def _null_metrics() -> dict[str, Any]:
    return dict(NULL_METRICS)


def _trial_values_for_k(trial_grid: dict[str, Any], top_k: int) -> list[int]:
    key = str(top_k)
    if key not in trial_grid:
        raise ValueError(f"Missing trial grid entry for top_k={top_k} (key='{key}')")
    return [int(v) for v in trial_grid[key]]


def _build_commands(
    model_cfg_name: str,
    direction: str,
    top_k: int,
    n_trials: int,
    stages: DictConfig,
    include_baseline_likert: bool,
    artifact_names: dict[str, str],
    intervention_scope: str,
    intervention_last_k: int,
) -> list[str]:
    """
    Compose stage commands with explicit Hydra overrides for every artifact
    identity. The orchestrator owns the deterministic name for each stage's
    output and threads it as the input of the next stage, so no script ever
    needs to guess (and stale defaults in config/model/*.yaml cannot leak in).

    `artifact_names` keys: activations, feature_ranking, multipliers,
    likert_baseline, likert_intervened. Values are bare names (no
    entity/project prefix) — wandb resolves them in the active run's project.
    """
    activations_name = artifact_names["activations"]
    feature_ranking_name = artifact_names["feature_ranking"]
    multipliers_name = artifact_names["multipliers"]
    likert_baseline_name = artifact_names["likert_baseline"]
    likert_intervened_name = artifact_names["likert_intervened"]

    activations_ref = f"{activations_name}:latest"
    feature_ranking_ref = f"{feature_ranking_name}:latest"
    multipliers_ref = f"{multipliers_name}:latest"

    cmds: list[str] = []
    if stages.get("extract_activations", False):
        cmds.append(
            "python src/extract_activations.py "
            f"model={model_cfg_name} "
            f"artifacts.activations_name={activations_name}"
        )
    if stages.get("feature_selection", False):
        cmds.append(
            "python src/train_eval_svc.py "
            f"model={model_cfg_name} "
            f"data.activations_artifact_name={activations_ref} "
            f"artifacts.feature_ranking_name={feature_ranking_name}"
        )
    if stages.get("optimization", False):
        cmds.append(
            "python src/optimize_intervention.py "
            f"model={model_cfg_name} "
            f"optimization.direction={direction} "
            f"optimization.top_k={top_k} "
            f"optimization.n_trials={n_trials} "
            f"optimization.intervention_scope={intervention_scope} "
            f"optimization.intervention_last_k={intervention_last_k} "
            f"optimization.feature_artifact_name={feature_ranking_ref} "
            f"artifacts.multiplier_name={multipliers_name}"
        )
    if stages.get("likert_baseline", False) and include_baseline_likert:
        # Baseline must NOT load a multiplier artifact; null wins over any
        # leftover model-config default. Scope is irrelevant for baseline
        # generation (no hooks) but we still pass it for log consistency.
        cmds.append(
            "python src/likert_scale_test.py "
            f"model={model_cfg_name} likert.condition=baseline "
            "ipi_eval.multiplier_artifact_name=null "
            f"likert.intervention_scope={intervention_scope} "
            f"likert.intervention_last_k={intervention_last_k} "
            f"artifacts.likert_baseline_name={likert_baseline_name}"
        )
    if stages.get("likert_intervened", False):
        cmds.append(
            "python src/likert_scale_test.py "
            f"model={model_cfg_name} likert.condition=intervened "
            f"optimization.direction={direction} optimization.top_k={top_k} "
            f"optimization.n_trials={n_trials} "
            f"likert.intervention_scope={intervention_scope} "
            f"likert.intervention_last_k={intervention_last_k} "
            f"ipi_eval.multiplier_artifact_name={multipliers_ref} "
            f"artifacts.likert_intervened_name={likert_intervened_name}"
        )
    if stages.get("poeta", False):
        cmds.append(f"python src/poeta_evaluator.py model={model_cfg_name}")
    return cmds


def _write_manifest(manifest_path: Path, payload: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _should_skip_existing(previous_status: str | None, resume: bool, force: bool, skip_existing: bool) -> bool:
    """
    Decide whether a previously planned job should be skipped.

    Rules:
    - `force=True` always reruns (never skips).
    - Only jobs with previous status `completed` are eligible for skipping.
    - For completed jobs, skip when either `resume` or `skip_existing` is enabled.
    """
    if force:
        return False
    if previous_status != "completed":
        return False
    return resume or skip_existing


def _execute_job_commands(
    commands: list[str],
    working_directory: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    wandb_project: str | None,
    wandb_entity: str | None,
) -> None:
    running_manifest = dict(manifest)
    running_manifest["status"] = "running"
    running_manifest["error"] = None
    _write_manifest(manifest_path, running_manifest)

    for command in commands:
        subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=str(working_directory),
        )

    completed_manifest = dict(running_manifest)
    completed_manifest["status"] = "completed"
    completed_manifest["error"] = None

    completed_manifest["metrics"] = _resolve_completion_metrics(
        manifest=running_manifest,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        existing_metrics=running_manifest.get("metrics"),
    )

    _write_manifest(manifest_path, completed_manifest)


def _resolve_completion_metrics(
    manifest: dict[str, Any],
    wandb_project: str | None,
    wandb_entity: str | None,
    existing_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Best-effort metric write-back after a successful execution.

    Failures here must not mark the run as failed: the heavy work already
    succeeded and the metric values can be reconstructed later via
    `src/backfill_manifests.py`. We log a warning, fall back to the existing
    metrics block (or nulls), and let the caller persist the manifest as
    `completed`.
    """
    base = dict(_null_metrics())
    if existing_metrics:
        for key in base:
            if existing_metrics.get(key) is not None:
                base[key] = existing_metrics[key]

    if not wandb_project:
        print(
            "  metrics: skipped write-back (wandb.project not configured); "
            "run src/backfill_manifests.py later to fill the manifest."
        )
        return base

    try:
        fetched = collect_run_metrics(
            manifest=manifest,
            project=wandb_project,
            entity=wandb_entity,
        )
    except MetricsBackfillError as exc:
        print(f"  metrics: write-back failed ({exc}); manifest left with nulls.")
        return base
    except Exception as exc:
        print(f"  metrics: unexpected write-back error ({exc}); manifest left with nulls.")
        return base

    for key, value in fetched.items():
        if value is not None:
            base[key] = value

    print("  metrics: written back from W&B")
    return base


def _baseline_reuse_key(
    model_cfg_name: str,
    split_id: str,
    seed: int,
    cfg: DictConfig,
) -> tuple[Any, ...]:
    """
    Build the baseline reuse key from settings that define baseline equivalence.

    Intervention scope is intentionally NOT part of this key because the
    baseline Likert run does not apply any multipliers (no hooks registered);
    the same baseline output is reusable across all scope variants.
    """
    likert_cfg = cfg.get("likert", {})
    data_cfg = cfg.get("data", {})
    ipi_test_dataset = data_cfg.get("ipi_test_dataset", cfg.ipi_eval.questions_csv)
    return (
        model_cfg_name,
        split_id,
        str(ipi_test_dataset),
        str(likert_cfg.get("prompt_template_version", "default")),
        str(likert_cfg.get("parser_version", "default")),
        float(likert_cfg.get("temperature", 0)),
        str(likert_cfg.get("decoding_strategy", "greedy")),
        int(seed),
    )


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("experiment") is None:
        raise ValueError(
            "Missing required experiment config. Example: experiment=k80_trials"
        )

    experiment = cfg.experiment
    split_id = str(experiment.split_id)
    seed = int(experiment.seed)
    ranking_top_n = int(cfg.feature_selection.get("ranking_top_n", 256))
    extraction_cfg = cfg.get("extraction", {})
    layers_cfg = extraction_cfg.get("layers", "all")
    layers = "all" if isinstance(layers_cfg, str) else str(list(layers_cfg))

    # Intervention scope axis. Missing `scopes` field means single-scope sweep
    # at the legacy default, which keeps existing experiment yamls
    # (k80_trials, small_k_trials) bit-identical on disk.
    scopes_cfg = experiment.get("scopes", None)
    if scopes_cfg is None:
        scopes = [DEFAULT_SCOPE]
    else:
        scopes = [str(s) for s in scopes_cfg]
    for scope in scopes:
        assert_scope(scope)
    intervention_last_k = int(experiment.get("intervention_last_k", DEFAULT_LAST_K))
    if intervention_last_k < 0:
        raise ValueError(
            f"experiment.intervention_last_k must be >= 0, got {intervention_last_k!r}."
        )

    output_root = Path("runs/pipeline")
    output_root.mkdir(parents=True, exist_ok=True)
    dry_run = bool(cfg.pipeline.get("dry_run", True))
    project_root = Path(hydra.utils.get_original_cwd())

    wandb_cfg = cfg.get("wandb", {}) or {}
    wandb_project = wandb_cfg.get("project")
    wandb_entity = wandb_cfg.get("entity")

    print("=" * 70)
    print(f"PIPELINE PLAN: {experiment.name}")
    print(f"dry_run={dry_run}")
    print(f"resume={cfg.pipeline.get('resume', True)} force={cfg.pipeline.get('force', False)}")
    print(f"skip_existing={cfg.pipeline.get('skip_existing', True)}")
    print("=" * 70)

    trial_grid = dict(experiment.trial_grid)
    job_count = 0
    skipped_count = 0
    failed_count = 0
    resume = bool(cfg.pipeline.get("resume", True))
    force = bool(cfg.pipeline.get("force", False))
    skip_existing = bool(cfg.pipeline.get("skip_existing", True))
    scheduled_baseline_keys: set[tuple[Any, ...]] = set()

    for model_cfg_name in experiment.models:
        model_cfg_name = str(model_cfg_name)
        for direction in experiment.directions:
            direction = str(direction)
            for top_k_value in experiment.feature_counts:
                top_k = int(top_k_value)
                for n_trials in _trial_values_for_k(trial_grid, top_k):
                    for scope in scopes:
                        run_id = make_run_id(
                            model_name=model_cfg_name,
                            split_id=split_id,
                            direction=direction,
                            top_k=top_k,
                            n_trials=n_trials,
                            seed=seed,
                            scope=scope,
                            last_k=intervention_last_k,
                        )
                        manifest_path = output_root / run_id / "manifest.json"
                        previous_manifest = _read_manifest(manifest_path)
                        previous_status = (
                            previous_manifest.get("status")
                            if previous_manifest is not None
                            else None
                        )
                        baseline_key = _baseline_reuse_key(
                            model_cfg_name=model_cfg_name,
                            split_id=split_id,
                            seed=seed,
                            cfg=cfg,
                        )
                        include_baseline_likert = baseline_key not in scheduled_baseline_keys

                        if _should_skip_existing(previous_status, resume, force, skip_existing):
                            skipped_count += 1
                            print(f"\n[skip] {run_id} (already completed; resume/skip_existing active)")
                            continue

                        try:
                            # Bare artifact names (no `:alias` suffix). Used both
                            # as outputs (passed to scripts via artifacts.*) and,
                            # with `:latest` appended, as inputs to downstream
                            # stages.
                            artifact_names = {
                                "activations": make_activation_artifact_name(
                                    model_name=model_cfg_name,
                                    split_id=split_id,
                                    layers=layers,
                                ),
                                "feature_ranking": make_feature_ranking_artifact_name(
                                    model_name=model_cfg_name,
                                    split_id=split_id,
                                    ranking_top_n=ranking_top_n,
                                ),
                                "multipliers": make_multiplier_artifact_name(
                                    model_name=model_cfg_name,
                                    split_id=split_id,
                                    direction=direction,
                                    top_k=top_k,
                                    n_trials=n_trials,
                                    seed=seed,
                                    scope=scope,
                                    last_k=intervention_last_k,
                                ),
                                "likert_baseline": make_likert_artifact_name(
                                    model_name=model_cfg_name,
                                    split_id=split_id,
                                    condition="baseline",
                                    seed=seed,
                                ),
                                "likert_intervened": make_likert_artifact_name(
                                    model_name=model_cfg_name,
                                    split_id=split_id,
                                    condition="intervened",
                                    seed=seed,
                                    direction=direction,
                                    top_k=top_k,
                                    n_trials=n_trials,
                                    scope=scope,
                                    last_k=intervention_last_k,
                                ),
                            }
                            artifacts = {k: f"{v}:latest" for k, v in artifact_names.items()}
                            commands = _build_commands(
                                model_cfg_name=model_cfg_name,
                                direction=direction,
                                top_k=top_k,
                                n_trials=n_trials,
                                stages=experiment.stages,
                                include_baseline_likert=include_baseline_likert,
                                artifact_names=artifact_names,
                                intervention_scope=scope,
                                intervention_last_k=intervention_last_k,
                            )
                            manifest = {
                                "run_id": run_id,
                                "status": "planned",
                                "model_name": model_cfg_name,
                                "split_id": split_id,
                                "direction": direction,
                                "top_k": top_k,
                                "n_trials": int(n_trials),
                                "seed": seed,
                                "intervention_scope": scope,
                                "intervention_last_k": intervention_last_k,
                                "commands": commands,
                                "artifacts": artifacts,
                                "metrics": _null_metrics(),
                                "error": None,
                            }
                            _write_manifest(manifest_path, manifest)

                            job_count += 1
                            print(f"\n[{job_count}] {run_id}")
                            if previous_status and force:
                                print(f"  forced replan over previous status={previous_status}")
                            for cmd in commands:
                                print(f"  - {cmd}")
                            print(f"  manifest: {manifest_path}")
                            if include_baseline_likert and experiment.stages.get("likert_baseline", False):
                                scheduled_baseline_keys.add(baseline_key)
                            else:
                                print("  baseline likert: reused (not rescheduled)")

                            if not dry_run:
                                _execute_job_commands(
                                    commands=commands,
                                    working_directory=project_root,
                                    manifest_path=manifest_path,
                                    manifest=manifest,
                                    wandb_project=wandb_project,
                                    wandb_entity=wandb_entity,
                                )
                                print("  execution: completed")
                        except Exception as exc:
                            failed_count += 1
                            failed_manifest = {
                                "run_id": run_id,
                                "status": "failed",
                                "model_name": model_cfg_name,
                                "split_id": split_id,
                                "direction": direction,
                                "top_k": top_k,
                                "n_trials": int(n_trials),
                                "seed": seed,
                                "intervention_scope": scope,
                                "intervention_last_k": intervention_last_k,
                                "commands": [],
                                "artifacts": {},
                                "metrics": _null_metrics(),
                                "error": str(exc),
                            }
                            _write_manifest(manifest_path, failed_manifest)
                            print(f"\n[failed] {run_id}")
                            print(f"  error: {exc}")
                            print(f"  manifest: {manifest_path}")

    print("\n" + "=" * 70)
    print(f"Planned jobs: {job_count}")
    print(f"Skipped jobs: {skipped_count}")
    print(f"Failed jobs: {failed_count}")
    print(f"Manifests written under: {output_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
