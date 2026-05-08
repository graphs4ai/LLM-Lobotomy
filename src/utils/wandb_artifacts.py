from __future__ import annotations
from pathlib import Path
from typing import Any
import wandb

"""Centralized W&B artifact operations with metadata validation."""


_REQUIRED_OPTIMIZATION_METADATA = {
    "stage",
    "model_name",
    "split_id",
    "direction",
    "top_k",
    "n_trials",
    "seed",
    "objective_mode",
    "feature_artifact_name",
    "optimization_dataset",
    "validation_dataset",
}


def _is_optimization_metadata(metadata: dict[str, Any]) -> bool:
    return metadata.get("stage") == "optimization"


def _validate_optimization_metadata(metadata: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED_OPTIMIZATION_METADATA - set(metadata.keys()))
    if missing:
        raise ValueError(
            "Optimization artifact metadata is missing required keys: "
            + ", ".join(missing)
        )


def _validate_required_metadata(
    actual: dict[str, Any], required: dict[str, Any] | None
) -> None:
    if not required:
        return
    mismatches: list[str] = []
    for key, expected_value in required.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected={expected_value!r}, actual={actual_value!r}"
            )
    if mismatches:
        raise ValueError("Artifact metadata mismatch: " + "; ".join(mismatches))


def _canonical_artifact_ref(artifact: Any, fallback_name: str) -> str:
    entity = getattr(artifact, "entity", None)
    project = getattr(artifact, "project", None)
    name = getattr(artifact, "name", None)
    version = getattr(artifact, "version", None)
    if entity and project and name and version:
        return f"{entity}/{project}/{name}:{version}"
    return fallback_name


def artifact_exists(
    artifact_name: str,
    artifact_type: str | None = None,
) -> bool:
    """Return True when artifact is resolvable in W&B."""
    api = wandb.Api()
    try:
        artifact = api.artifact(artifact_name, type=artifact_type)
        return artifact is not None
    except Exception:
        return False


def resolve_artifact(
    artifact_name: str,
    artifact_type: str | None = None,
    required_metadata: dict | None = None,
) -> str:
    """
    Resolve an artifact and validate optional metadata constraints.

    Returns a canonical artifact reference when available.
    """
    api = wandb.Api()
    try:
        artifact = api.artifact(artifact_name, type=artifact_type)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not resolve artifact '{artifact_name}'"
            + (f" (type='{artifact_type}')" if artifact_type else "")
            + f": {exc}"
        ) from exc

    metadata = dict(getattr(artifact, "metadata", {}) or {})
    if _is_optimization_metadata(metadata):
        _validate_optimization_metadata(metadata)
    _validate_required_metadata(metadata, required_metadata)
    return _canonical_artifact_ref(artifact, artifact_name)


def log_artifact_with_metadata(
    artifact_name: str,
    artifact_type: str,
    files: list[str],
    metadata: dict,
    aliases: list[str] | None = None,
) -> str:
    """
    Log artifact files with validated metadata.

    Requires an active `wandb.run`.
    """
    if not metadata:
        raise ValueError("Artifact metadata must not be empty.")
    if _is_optimization_metadata(metadata):
        _validate_optimization_metadata(metadata)
    if not files:
        raise ValueError("At least one file is required to log an artifact.")
    if wandb.run is None:
        raise RuntimeError(
            "wandb.run is not initialized. Call wandb.init(...) before logging artifacts."
        )

    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata,
    )
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file does not exist: {file_path}")
        artifact.add_file(str(path))

    logged_artifact = wandb.run.log_artifact(artifact, aliases=aliases)
    if logged_artifact is None:
        return artifact_name
    return _canonical_artifact_ref(logged_artifact, artifact_name)
