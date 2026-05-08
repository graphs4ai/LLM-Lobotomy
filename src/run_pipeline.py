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


def _null_metrics() -> dict[str, Any]:
    return {
        "soft_ipi_optimization_baseline": None,
        "soft_ipi_optimization_intervened": None,
        "delta_soft_ipi_optimization": None,
        "soft_ipi_validation_baseline": None,
        "soft_ipi_validation_intervened": None,
        "delta_soft_ipi_validation": None,
        "discrete_ipi_test_baseline": None,
        "discrete_ipi_test_intervened": None,
        "delta_discrete_ipi_test": None,
        "wilcoxon_p_value": None,
    }


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
) -> list[str]:
    cmds: list[str] = []
    if stages.get("extract_activations", False):
        cmds.append(f"python src/extract_activations.py model={model_cfg_name}")
    if stages.get("feature_selection", False):
        cmds.append(f"python src/train_eval_svc.py model={model_cfg_name}")
    if stages.get("optimization", False):
        cmds.append(
            "python src/optimize_intervention.py "
            f"model={model_cfg_name} optimization.direction={direction} "
            f"optimization.top_k={top_k} optimization.n_trials={n_trials}"
        )
    if stages.get("likert_baseline", False):
        cmds.append(
            "python src/likert_scale_test.py "
            f"model={model_cfg_name} likert.condition=baseline"
        )
    if stages.get("likert_intervened", False):
        cmds.append(
            "python src/likert_scale_test.py "
            f"model={model_cfg_name} likert.condition=intervened "
            f"optimization.direction={direction} optimization.top_k={top_k} "
            f"optimization.n_trials={n_trials}"
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
    _write_manifest(manifest_path, completed_manifest)


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

    output_root = Path("runs/pipeline")
    output_root.mkdir(parents=True, exist_ok=True)
    dry_run = bool(cfg.pipeline.get("dry_run", True))
    project_root = Path(hydra.utils.get_original_cwd())

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

    for model_cfg_name in experiment.models:
        model_cfg_name = str(model_cfg_name)
        for direction in experiment.directions:
            direction = str(direction)
            for top_k_value in experiment.feature_counts:
                top_k = int(top_k_value)
                for n_trials in _trial_values_for_k(trial_grid, top_k):
                    run_id = make_run_id(
                        model_name=model_cfg_name,
                        split_id=split_id,
                        direction=direction,
                        top_k=top_k,
                        n_trials=n_trials,
                        seed=seed,
                    )
                    manifest_path = output_root / run_id / "manifest.json"
                    previous_manifest = _read_manifest(manifest_path)
                    previous_status = (
                        previous_manifest.get("status")
                        if previous_manifest is not None
                        else None
                    )

                    if _should_skip_existing(previous_status, resume, force, skip_existing):
                        skipped_count += 1
                        print(f"\n[skip] {run_id} (already completed; resume/skip_existing active)")
                        continue

                    try:
                        artifacts = {
                            "activations": make_activation_artifact_name(
                                model_name=model_cfg_name,
                                split_id=split_id,
                                layers=layers,
                            )
                            + ":latest",
                            "feature_ranking": make_feature_ranking_artifact_name(
                                model_name=model_cfg_name,
                                split_id=split_id,
                                ranking_top_n=ranking_top_n,
                            )
                            + ":latest",
                            "multipliers": make_multiplier_artifact_name(
                                model_name=model_cfg_name,
                                split_id=split_id,
                                direction=direction,
                                top_k=top_k,
                                n_trials=n_trials,
                                seed=seed,
                            )
                            + ":latest",
                            "likert_baseline": make_likert_artifact_name(
                                model_name=model_cfg_name,
                                split_id=split_id,
                                condition="baseline",
                                seed=seed,
                            )
                            + ":latest",
                            "likert_intervened": make_likert_artifact_name(
                                model_name=model_cfg_name,
                                split_id=split_id,
                                condition="intervened",
                                seed=seed,
                                direction=direction,
                                top_k=top_k,
                                n_trials=n_trials,
                            )
                            + ":latest",
                        }
                        commands = _build_commands(
                            model_cfg_name=model_cfg_name,
                            direction=direction,
                            top_k=top_k,
                            n_trials=n_trials,
                            stages=experiment.stages,
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

                        if not dry_run:
                            _execute_job_commands(
                                commands=commands,
                                working_directory=project_root,
                                manifest_path=manifest_path,
                                manifest=manifest,
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
