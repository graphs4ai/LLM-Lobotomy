"""Generate CSV and Markdown summaries from pipeline manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "run_id",
    "model_name",
    "split_id",
    "direction",
    "top_k",
    "n_trials",
    "seed",
    "activation_artifact",
    "feature_ranking_artifact",
    "multiplier_artifact",
    "likert_baseline_artifact",
    "likert_intervened_artifact",
    "soft_ipi_optimization_baseline",
    "soft_ipi_optimization_intervened",
    "delta_soft_ipi_optimization",
    "soft_ipi_validation_baseline",
    "soft_ipi_validation_intervened",
    "delta_soft_ipi_validation",
    "discrete_ipi_test_baseline",
    "discrete_ipi_test_intervened",
    "delta_discrete_ipi_test",
    "wilcoxon_p_value",
    "status",
    "error",
]


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _flatten_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts", {}) or {}
    metrics = manifest.get("metrics", {}) or {}
    return {
        "run_id": manifest.get("run_id"),
        "model_name": manifest.get("model_name"),
        "split_id": manifest.get("split_id"),
        "direction": manifest.get("direction"),
        "top_k": manifest.get("top_k"),
        "n_trials": manifest.get("n_trials"),
        "seed": manifest.get("seed"),
        "activation_artifact": artifacts.get("activations"),
        "feature_ranking_artifact": artifacts.get("feature_ranking"),
        "multiplier_artifact": artifacts.get("multipliers"),
        "likert_baseline_artifact": artifacts.get("likert_baseline"),
        "likert_intervened_artifact": artifacts.get("likert_intervened"),
        "soft_ipi_optimization_baseline": metrics.get("soft_ipi_optimization_baseline"),
        "soft_ipi_optimization_intervened": metrics.get("soft_ipi_optimization_intervened"),
        "delta_soft_ipi_optimization": metrics.get("delta_soft_ipi_optimization"),
        "soft_ipi_validation_baseline": metrics.get("soft_ipi_validation_baseline"),
        "soft_ipi_validation_intervened": metrics.get("soft_ipi_validation_intervened"),
        "delta_soft_ipi_validation": metrics.get("delta_soft_ipi_validation"),
        "discrete_ipi_test_baseline": metrics.get("discrete_ipi_test_baseline"),
        "discrete_ipi_test_intervened": metrics.get("discrete_ipi_test_intervened"),
        "delta_discrete_ipi_test": metrics.get("delta_discrete_ipi_test"),
        "wilcoxon_p_value": metrics.get("wilcoxon_p_value"),
        "status": manifest.get("status"),
        "error": manifest.get("error"),
    }


def _read_manifests(pipeline_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for manifest_path in sorted(pipeline_dir.glob("*/manifest.json")):
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifests.append(_flatten_manifest(manifest))
    return manifests


def _write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _format_cell(row.get(col)) for col in SUMMARY_COLUMNS})


def _write_markdown(rows: list[dict[str, Any]], md_path: Path) -> None:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

    lines: list[str] = []
    lines.append("# Sweep Summary")
    lines.append("")
    lines.append(f"- Total jobs: {len(rows)}")
    for status in sorted(status_counts):
        lines.append(f"- {status}: {status_counts[status]}")
    lines.append("")

    if not rows:
        lines.append("No manifests found under `runs/pipeline/*/manifest.json`.")
    else:
        lines.append("| " + " | ".join(SUMMARY_COLUMNS) + " |")
        lines.append("| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |")
        for row in rows:
            values = [_format_cell(row.get(col)).replace("\n", " ") for col in SUMMARY_COLUMNS]
            lines.append("| " + " | ".join(values) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    pipeline_dir = project_root / "runs" / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_manifests(pipeline_dir)
    rows = sorted(rows, key=lambda r: str(r.get("run_id") or ""))

    csv_path = pipeline_dir / "summary.csv"
    md_path = pipeline_dir / "summary.md"
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path)

    print(f"Manifests scanned: {len(rows)}")
    print(f"CSV written: {csv_path}")
    print(f"Markdown written: {md_path}")


if __name__ == "__main__":
    main()
