"""Read pipeline run metrics back from W&B into the manifest schema.

This module is shared by `src/run_pipeline.py` (post-execution write-back) and
`src/backfill_manifests.py` (ad-hoc rebuild) so both call sites use one source
of truth for how a job's W&B output is mapped onto the local manifest
`metrics` block.

Mapping (W&B -> manifest.metrics):
  multipliers artifact metadata
      soft_ipi_optimization_baseline    -> soft_ipi_optimization_baseline
      soft_ipi_optimization_intervened  -> soft_ipi_optimization_intervened
      delta_soft_ipi_optimization       -> delta_soft_ipi_optimization
      soft_ipi_validation_baseline      -> soft_ipi_validation_baseline
      soft_ipi_validation_intervened    -> soft_ipi_validation_intervened
      delta_soft_ipi_validation         -> delta_soft_ipi_validation
  intervened likert run summary
      baseline_pi                       -> discrete_ipi_validation_* or discrete_ipi_test_*
      intervention_pi                   -> (same split bucket)
      pi_shift                          -> (same split bucket)
      test_pvalue (when Wilcoxon)       -> wilcoxon_validation_p_value or wilcoxon_p_value
"""

from __future__ import annotations

import math
from typing import Any

import wandb


NULL_METRICS: dict[str, Any] = {
    "soft_ipi_optimization_baseline": None,
    "soft_ipi_optimization_intervened": None,
    "delta_soft_ipi_optimization": None,
    "soft_ipi_validation_baseline": None,
    "soft_ipi_validation_intervened": None,
    "delta_soft_ipi_validation": None,
    "discrete_ipi_validation_baseline": None,
    "discrete_ipi_validation_intervened": None,
    "delta_discrete_ipi_validation": None,
    "wilcoxon_validation_p_value": None,
    "discrete_ipi_test_baseline": None,
    "discrete_ipi_test_intervened": None,
    "delta_discrete_ipi_test": None,
    "wilcoxon_p_value": None,
}

_SOFT_METRIC_KEYS = (
    "soft_ipi_optimization_baseline",
    "soft_ipi_optimization_intervened",
    "delta_soft_ipi_optimization",
    "soft_ipi_validation_baseline",
    "soft_ipi_validation_intervened",
    "delta_soft_ipi_validation",
)


class MetricsBackfillError(RuntimeError):
    """Raised when manifest metric backfill cannot complete."""


def _qualified_artifact_path(
    artifact_ref: str,
    project: str,
    entity: str | None,
) -> str:
    """
    Normalize an artifact reference to `[entity/]project/name:alias`.

    `wandb.Api().artifact(...)` requires at minimum `project/name:alias`; the
    manifest stores the bare form `name:alias`, so we add the missing prefix
    components when they are not already present.
    """
    if "/" in artifact_ref:
        return artifact_ref
    if entity:
        return f"{entity}/{project}/{artifact_ref}"
    return f"{project}/{artifact_ref}"


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(coerced):
        return None
    return coerced


def _extract_soft_metrics(metadata: dict[str, Any]) -> dict[str, float | None]:
    return {key: _coerce_float(metadata.get(key)) for key in _SOFT_METRIC_KEYS}


def _fetch_multipliers_metadata(
    artifact_ref: str,
    project: str,
    entity: str | None,
) -> dict[str, Any]:
    api = wandb.Api()
    qualified = _qualified_artifact_path(artifact_ref, project=project, entity=entity)
    artifact = api.artifact(qualified, type="model-weights")
    return dict(getattr(artifact, "metadata", {}) or {})


def _select_latest_run(runs: list[Any]) -> Any | None:
    """Pick the most recently created run from a wandb runs iterable."""
    if not runs:
        return None

    def _sort_key(run: Any) -> str:
        created_at = getattr(run, "created_at", None)
        return str(created_at) if created_at is not None else ""

    return sorted(runs, key=_sort_key, reverse=True)[0]


def _likert_eval_split(run: Any) -> str:
    """Classify which IPI split a Likert run evaluated."""
    summary = dict(getattr(run, "summary", {}) or {})
    summary_split = summary.get("likert_eval_split")
    if summary_split in {"validation", "holdout_test"}:
        return str(summary_split)

    config = dict(getattr(run, "config", {}) or {})
    likert_cfg = config.get("likert") or {}
    if isinstance(likert_cfg, dict):
        cfg_split = likert_cfg.get("eval_split")
        if cfg_split in {"validation", "holdout_test"}:
            return str(cfg_split)

    data_cfg = config.get("data") or {}
    if not isinstance(data_cfg, dict):
        data_cfg = {}

    eval_dataset = summary.get("likert_eval_dataset")
    if eval_dataset is None:
        return "unknown"

    eval_path = str(eval_dataset)
    validation_path = data_cfg.get("validation_dataset")
    if validation_path is not None and eval_path.endswith(str(validation_path)):
        return "validation"
    test_path = data_cfg.get("ipi_test_dataset")
    if test_path is not None and eval_path.endswith(str(test_path)):
        return "holdout_test"
    if "ipi_questions_val" in eval_path:
        return "validation"
    if "ipi_questions_test" in eval_path:
        return "holdout_test"
    return "unknown"


def _empty_likert_metrics() -> dict[str, float | None]:
    return {
        "discrete_ipi_validation_baseline": None,
        "discrete_ipi_validation_intervened": None,
        "delta_discrete_ipi_validation": None,
        "wilcoxon_validation_p_value": None,
        "discrete_ipi_test_baseline": None,
        "discrete_ipi_test_intervened": None,
        "delta_discrete_ipi_test": None,
        "wilcoxon_p_value": None,
    }


def _map_likert_summary_to_metrics(
    summary: dict[str, Any],
    eval_split: str,
) -> dict[str, float | None]:
    baseline_pi = _coerce_float(summary.get("baseline_pi"))
    intervention_pi = _coerce_float(summary.get("intervention_pi"))
    pi_shift = _coerce_float(summary.get("pi_shift"))
    test_type = summary.get("test_type")
    test_pvalue = _coerce_float(summary.get("test_pvalue"))
    wilcoxon = None
    if test_pvalue is not None and isinstance(test_type, str) and "wilcoxon" in test_type.lower():
        wilcoxon = test_pvalue

    metrics = _empty_likert_metrics()
    if eval_split == "holdout_test":
        metrics.update(
            {
                "discrete_ipi_test_baseline": baseline_pi,
                "discrete_ipi_test_intervened": intervention_pi,
                "delta_discrete_ipi_test": pi_shift,
                "wilcoxon_p_value": wilcoxon,
            }
        )
        return metrics

    # Pipeline sweeps default to validation; unknown legacy runs stay on test keys.
    if eval_split == "validation":
        target_prefix = "validation"
    else:
        target_prefix = "test"

    if target_prefix == "validation":
        metrics.update(
            {
                "discrete_ipi_validation_baseline": baseline_pi,
                "discrete_ipi_validation_intervened": intervention_pi,
                "delta_discrete_ipi_validation": pi_shift,
                "wilcoxon_validation_p_value": wilcoxon,
            }
        )
    else:
        metrics.update(
            {
                "discrete_ipi_test_baseline": baseline_pi,
                "discrete_ipi_test_intervened": intervention_pi,
                "delta_discrete_ipi_test": pi_shift,
                "wilcoxon_p_value": wilcoxon,
            }
        )
    return metrics


def _fetch_likert_metrics(
    multipliers_ref: str,
    project: str,
    entity: str | None,
) -> dict[str, float | None]:
    """
    Locate the intervened Likert W&B run whose config references our
    multipliers artifact, then read the 4 discrete-IPI/Wilcoxon values out of
    its summary.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project

    filters = {
        "$or": [
            {"config.multiplier_artifact_name": multipliers_ref},
            {"config.ipi_eval.multiplier_artifact_name": multipliers_ref},
        ]
    }
    try:
        runs_iter = api.runs(path=path, filters=filters)
        runs = list(runs_iter)
    except Exception as exc:
        raise MetricsBackfillError(
            f"Failed to query Likert runs for multipliers={multipliers_ref!r}: {exc}"
        ) from exc

    run = _select_latest_run(runs)
    if run is None:
        return _empty_likert_metrics()

    summary = dict(getattr(run, "summary", {}) or {})
    eval_split = _likert_eval_split(run)
    return _map_likert_summary_to_metrics(summary, eval_split=eval_split)


def collect_run_metrics(
    manifest: dict[str, Any],
    project: str,
    entity: str | None = None,
) -> dict[str, float | None]:
    """
    Build the manifest `metrics` block for one job by reading W&B.

    Returns a dict aligned with `NULL_METRICS`. Missing values are returned as
    None rather than raising, so callers can merge the partial dict onto the
    existing manifest without erasing previously populated keys.

    Raises `MetricsBackfillError` only on infrastructure failures (e.g. the
    multipliers artifact itself is unresolvable).
    """
    artifacts = manifest.get("artifacts") or {}
    multipliers_ref = artifacts.get("multipliers")
    if not multipliers_ref:
        raise MetricsBackfillError(
            f"Manifest is missing artifacts.multipliers reference: run_id="
            f"{manifest.get('run_id')!r}"
        )

    try:
        metadata = _fetch_multipliers_metadata(
            multipliers_ref, project=project, entity=entity
        )
    except Exception as exc:
        raise MetricsBackfillError(
            f"Failed to fetch multipliers artifact {multipliers_ref!r}: {exc}"
        ) from exc

    soft_metrics = _extract_soft_metrics(metadata)
    likert_metrics = _fetch_likert_metrics(
        multipliers_ref, project=project, entity=entity
    )

    result = dict(NULL_METRICS)
    result.update(soft_metrics)
    result.update(likert_metrics)
    return result


def collect_run_identity(
    manifest: dict[str, Any],
    project: str,
    entity: str | None = None,
) -> dict[str, Any]:
    """
    Pull non-metric identity fields (intervention_scope, intervention_last_k)
    out of the multipliers artifact metadata so old manifests that pre-date the
    scope field can be patched in-place.

    Returns a dict containing only the keys whose values could be resolved.
    Missing keys are simply omitted (the caller decides whether to keep the
    existing manifest value or fall back to a default).
    """
    artifacts = manifest.get("artifacts") or {}
    multipliers_ref = artifacts.get("multipliers")
    if not multipliers_ref:
        raise MetricsBackfillError(
            f"Manifest is missing artifacts.multipliers reference: run_id="
            f"{manifest.get('run_id')!r}"
        )

    try:
        metadata = _fetch_multipliers_metadata(
            multipliers_ref, project=project, entity=entity
        )
    except Exception as exc:
        raise MetricsBackfillError(
            f"Failed to fetch multipliers artifact {multipliers_ref!r}: {exc}"
        ) from exc

    out: dict[str, Any] = {}
    scope = metadata.get("intervention_scope")
    if scope is not None:
        out["intervention_scope"] = str(scope)
    last_k = metadata.get("intervention_last_k")
    if last_k is not None:
        try:
            out["intervention_last_k"] = int(last_k)
        except (TypeError, ValueError):
            pass
    return out


_REQUIRED_METRIC_KEYS = (
    "soft_ipi_optimization_baseline",
    "soft_ipi_optimization_intervened",
    "delta_soft_ipi_optimization",
    "soft_ipi_validation_baseline",
    "soft_ipi_validation_intervened",
    "delta_soft_ipi_validation",
    "discrete_ipi_validation_baseline",
    "discrete_ipi_validation_intervened",
    "delta_discrete_ipi_validation",
    "wilcoxon_validation_p_value",
)


def metrics_are_complete(metrics: dict[str, Any] | None) -> bool:
    """Return True when sweep-relevant metric keys are populated."""
    if not metrics:
        return False
    return all(metrics.get(key) is not None for key in _REQUIRED_METRIC_KEYS)
