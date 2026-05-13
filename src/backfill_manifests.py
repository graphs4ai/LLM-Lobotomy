"""Rebuild pipeline manifest metrics from W&B without re-running heavy stages.

Standalone CLI (no Hydra). Walks `runs/pipeline/*/manifest.json`, looks up the
W&B run that produced each completed job, and rewrites the manifest in place
with the resolved metric values.

Usage:
    python src/backfill_manifests.py --project LLM-Lobotomy
    python src/backfill_manifests.py --project LLM-Lobotomy --dry-run
    python src/backfill_manifests.py --project LLM-Lobotomy --force
    python src/backfill_manifests.py --project LLM-Lobotomy --run-id "*k4*"
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

# Make src/utils/* importable when invoked from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.metrics_backfill import (  # noqa: E402
    NULL_METRICS,
    MetricsBackfillError,
    collect_run_identity,
    collect_run_metrics,
    metrics_are_complete,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill pipeline manifest metrics from W&B without rerunning "
            "heavy jobs."
        )
    )
    parser.add_argument(
        "--project",
        required=True,
        help="W&B project name (e.g. 'LLM-Lobotomy').",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (team/user). Optional; falls back to your default.",
    )
    parser.add_argument(
        "--pipeline-dir",
        default=str(_PROJECT_ROOT / "runs" / "pipeline"),
        help="Directory containing `<run_id>/manifest.json` entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed metric values without writing manifests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Backfill even manifests whose status is not 'completed' or "
            "whose metrics block is already fully populated."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional fnmatch pattern restricting which run_id directories "
            "are visited (e.g. '*k4*' or '*minimize*')."
        ),
    )
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _merged_metrics(
    existing: dict[str, Any] | None,
    fetched: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """
    Overlay fetched values on top of an existing metrics block.

    Returns the merged block plus the list of keys whose value actually
    changed, so callers can report and short-circuit on no-op runs.
    """
    base = dict(NULL_METRICS)
    if existing:
        for key in base:
            if existing.get(key) is not None:
                base[key] = existing[key]

    changed: list[str] = []
    for key, value in fetched.items():
        if value is None:
            continue
        if base.get(key) != value:
            base[key] = value
            changed.append(key)
    return base, changed


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _print_metrics_table(metrics: dict[str, Any]) -> None:
    width = max(len(k) for k in NULL_METRICS)
    for key in NULL_METRICS:
        print(f"    {key.ljust(width)}  {_format_value(metrics.get(key))}")


def _matches_filter(run_id: str, pattern: str | None) -> bool:
    if pattern is None:
        return True
    return fnmatch.fnmatch(run_id, pattern)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pipeline_dir = Path(args.pipeline_dir).resolve()
    if not pipeline_dir.exists():
        print(f"Pipeline directory not found: {pipeline_dir}")
        return 1

    manifest_paths = sorted(pipeline_dir.glob("*/manifest.json"))
    if not manifest_paths:
        print(f"No manifests found under {pipeline_dir}/*/manifest.json")
        return 0

    print(
        f"Backfill plan: project={args.project!r} entity={args.entity!r} "
        f"dry_run={args.dry_run} force={args.force}"
    )
    print(f"Scanning {len(manifest_paths)} manifests in {pipeline_dir}")
    if args.run_id:
        print(f"Filter: run_id matches {args.run_id!r}")

    scanned = 0
    updated = 0
    skipped = 0
    failed = 0

    for manifest_path in manifest_paths:
        manifest = _load_manifest(manifest_path)
        run_id = str(manifest.get("run_id") or manifest_path.parent.name)

        if not _matches_filter(run_id, args.run_id):
            continue

        scanned += 1
        status = manifest.get("status")
        existing_metrics = manifest.get("metrics") or {}

        print(f"\n[{scanned}] {run_id}")
        print(f"  status: {status}")

        if status != "completed" and not args.force:
            print("  skip: status != 'completed' (use --force to override)")
            skipped += 1
            continue

        if metrics_are_complete(existing_metrics) and not args.force:
            print("  skip: metrics already fully populated (use --force to override)")
            skipped += 1
            continue

        try:
            fetched = collect_run_metrics(
                manifest=manifest,
                project=args.project,
                entity=args.entity,
            )
        except MetricsBackfillError as exc:
            print(f"  fail: {exc}")
            failed += 1
            continue
        except Exception as exc:
            print(f"  fail: unexpected error: {exc}")
            failed += 1
            continue

        # Also recover non-metric identity fields (intervention_scope,
        # intervention_last_k) from the multipliers artifact metadata. This
        # patches manifests that pre-date the scope axis without forcing a
        # rerun.
        identity_changed: list[str] = []
        try:
            identity = collect_run_identity(
                manifest=manifest,
                project=args.project,
                entity=args.entity,
            )
        except MetricsBackfillError as exc:
            print(f"  warn: identity backfill failed ({exc})")
            identity = {}
        except Exception as exc:
            print(f"  warn: identity backfill unexpected error ({exc})")
            identity = {}

        for key, value in identity.items():
            if manifest.get(key) != value:
                identity_changed.append(key)

        merged, changed = _merged_metrics(existing_metrics, fetched)

        print("  fetched metrics:")
        _print_metrics_table(merged)
        if identity:
            print("  fetched identity:")
            for key, value in identity.items():
                print(f"    {key}: {value}")

        if not changed and not identity_changed:
            print("  skip: no new values to write")
            skipped += 1
            continue

        if args.dry_run:
            all_keys = list(changed) + identity_changed
            print(
                f"  dry-run: would update {len(all_keys)} keys: "
                f"{', '.join(all_keys)}"
            )
            updated += 1
            continue

        manifest["metrics"] = merged
        for key in identity_changed:
            manifest[key] = identity[key]
        _write_manifest(manifest_path, manifest)
        total_updates = len(changed) + len(identity_changed)
        print(f"  wrote: {manifest_path} ({total_updates} keys updated)")
        updated += 1

    print("\n" + "=" * 60)
    print(f"Scanned: {scanned}")
    print(f"{'Would update' if args.dry_run else 'Updated'}: {updated}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print("=" * 60)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
