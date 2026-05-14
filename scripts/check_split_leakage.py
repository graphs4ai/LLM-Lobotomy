"""Check whether dataset splits leak by comparing question text, not only pair_id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATASETS = {
    "feature_selection": "resources/split_FS.csv",
    "optimization": "resources/split_optim.csv",
    "ipi_validation": "resources/ipi_questions_val.csv",
    "ipi_test": "resources/ipi_questions_test.csv",
    "ipi_full": "resources/ipi_questions.csv",
}


def _normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split())


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _ipi_rows(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    required = {"pair_id", "tipo_pergunta", "pergunta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset_name} is missing IPI columns {sorted(missing)}; got {list(df.columns)}"
        )

    rows = df.loc[:, ["pair_id", "tipo_pergunta", "pergunta"]].copy()
    rows["pair_id"] = rows["pair_id"].astype(int)
    rows["tipo_pergunta"] = rows["tipo_pergunta"].astype(str)
    rows["text_norm"] = rows["pergunta"].map(_normalize_text)
    rows["text_key"] = rows["tipo_pergunta"] + "::" + rows["text_norm"]
    rows["pair_key"] = (
        rows["pair_id"].astype(str) + "::" + rows["tipo_pergunta"].astype(str)
    )
    return rows


def _feature_selection_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "statement" not in df.columns:
        raise ValueError("feature_selection dataset must contain a statement column")

    rows = df.loc[:, ["statement"]].copy()
    if "id" in df.columns:
        rows["id"] = df["id"].astype(str)
    rows["text_norm"] = rows["statement"].map(_normalize_text)
    return rows


def _pair_id_overlap(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, object]:
    left_pairs = set(left["pair_id"].unique())
    right_pairs = set(right["pair_id"].unique())
    shared_pairs = sorted(left_pairs & right_pairs)
    return {
        "left_pair_count": len(left_pairs),
        "right_pair_count": len(right_pairs),
        "shared_pair_ids": shared_pairs,
        "shared_pair_id_count": len(shared_pairs),
    }


def _pair_key_text_overlap(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, object]:
    merged = left.merge(
        right,
        on=["pair_id", "tipo_pergunta"],
        how="inner",
        suffixes=("_left", "_right"),
    )
    exact_text_matches = merged[merged["text_norm_left"] == merged["text_norm_right"]]
    mismatched = merged[merged["text_norm_left"] != merged["text_norm_right"]]
    return {
        "shared_pair_key_count": int(len(merged)),
        "exact_text_match_count": int(len(exact_text_matches)),
        "mismatched_pair_key_count": int(len(mismatched)),
        "sample_mismatches": mismatched[
            ["pair_id", "tipo_pergunta", "pergunta_left", "pergunta_right"]
        ].head(5).to_dict(orient="records"),
    }


def _text_overlap_ignoring_pair_id(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, object]:
    shared_text_keys = sorted(set(left["text_key"]) & set(right["text_key"]))
    return {
        "shared_text_key_count": len(shared_text_keys),
        "sample_shared_text_keys": shared_text_keys[:5],
    }


def _compare_ipi_splits(
    left_name: str,
    left_df: pd.DataFrame,
    right_name: str,
    right_df: pd.DataFrame,
) -> dict[str, object]:
    left_rows = _ipi_rows(left_df, left_name)
    right_rows = _ipi_rows(right_df, right_name)
    return {
        "left": left_name,
        "right": right_name,
        "left_row_count": int(len(left_rows)),
        "right_row_count": int(len(right_rows)),
        "pair_id_overlap": _pair_id_overlap(left_rows, right_rows),
        "pair_id_and_type_overlap": _pair_key_text_overlap(left_rows, right_rows),
        "text_overlap_ignoring_pair_id": _text_overlap_ignoring_pair_id(left_rows, right_rows),
    }


def _compare_feature_selection_to_ipi(
    feature_df: pd.DataFrame,
    ipi_name: str,
    ipi_df: pd.DataFrame,
) -> dict[str, object]:
    feature_rows = _feature_selection_rows(feature_df)
    ipi_rows = _ipi_rows(ipi_df, ipi_name)
    feature_texts = set(feature_rows["text_norm"]) - {""}
    ipi_texts = set(ipi_rows["text_norm"]) - {""}
    shared_texts = sorted(feature_texts & ipi_texts)
    return {
        "left": "feature_selection",
        "right": ipi_name,
        "feature_selection_row_count": int(len(feature_rows)),
        "ipi_row_count": int(len(ipi_rows)),
        "shared_text_count": len(shared_texts),
        "sample_shared_texts": shared_texts[:5],
    }


def _full_union_check(
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> dict[str, object]:
    validation_rows = _ipi_rows(validation_df, "ipi_validation")
    test_rows = _ipi_rows(test_df, "ipi_test")
    full_rows = _ipi_rows(full_df, "ipi_full")
    validation_keys = set(validation_rows["text_key"])
    test_keys = set(test_rows["text_key"])
    full_keys = set(full_rows["text_key"])
    union_keys = validation_keys | test_keys
    return {
        "validation_text_count": len(validation_keys),
        "test_text_count": len(test_keys),
        "full_text_count": len(full_keys),
        "union_equals_full": union_keys == full_keys,
        "only_in_full": sorted(full_keys - union_keys)[:5],
        "only_in_union": sorted(union_keys - full_keys)[:5],
        "validation_test_text_overlap_count": len(validation_keys & test_keys),
    }


def build_report(project_root: Path) -> dict[str, object]:
    paths = {name: project_root / rel for name, rel in DEFAULT_DATASETS.items()}
    frames = {name: _load_csv(path) for name, path in paths.items()}

    comparisons = [
        _compare_ipi_splits("optimization", frames["optimization"], "ipi_validation", frames["ipi_validation"]),
        _compare_ipi_splits("optimization", frames["optimization"], "ipi_test", frames["ipi_test"]),
        _compare_ipi_splits("ipi_validation", frames["ipi_validation"], "ipi_test", frames["ipi_test"]),
    ]

    mismatches: list[dict[str, object]] = []
    for comparison in comparisons:
        left_name = str(comparison["left"])
        right_name = str(comparison["right"])
        for row in comparison["pair_id_and_type_overlap"]["sample_mismatches"]:
            mismatches.append(
                {
                    "left": left_name,
                    "right": right_name,
                    **row,
                }
            )

    return {
        "datasets": {name: str(path) for name, path in paths.items()},
        "comparisons": comparisons,
        "pair_id_overlap_with_different_text": mismatches,
        "pair_id_overlap_with_different_text_count": sum(
            comparison["pair_id_and_type_overlap"]["mismatched_pair_key_count"]
            for comparison in comparisons
        ),
        "feature_selection_vs_ipi": [
            _compare_feature_selection_to_ipi(frames["feature_selection"], "ipi_validation", frames["ipi_validation"]),
            _compare_feature_selection_to_ipi(frames["feature_selection"], "ipi_test", frames["ipi_test"]),
        ],
        "validation_test_full_union": _full_union_check(
            frames["ipi_validation"],
            frames["ipi_test"],
            frames["ipi_full"],
        ),
    }


def _print_summary(report: dict[str, object]) -> None:
    print("Split leakage check (text-aware)")
    print("=" * 72)
    for comparison in report["comparisons"]:
        left = comparison["left"]
        right = comparison["right"]
        pair_overlap = comparison["pair_id_overlap"]
        pair_key_overlap = comparison["pair_id_and_type_overlap"]
        text_overlap = comparison["text_overlap_ignoring_pair_id"]
        print(f"\n{left} vs {right}")
        print(
            f"  shared pair_id: {pair_overlap['shared_pair_id_count']} "
            f"(left={pair_overlap['left_pair_count']}, right={pair_overlap['right_pair_count']})"
        )
        print(
            f"  shared pair_id+tipo_pergunta rows: {pair_key_overlap['shared_pair_key_count']}"
        )
        print(
            f"  exact text matches on shared pair_id+tipo_pergunta: "
            f"{pair_key_overlap['exact_text_match_count']}"
        )
        print(
            f"  shared pair_id+tipo_pergunta with different text: "
            f"{pair_key_overlap['mismatched_pair_key_count']}"
        )
        print(
            f"  exact text matches ignoring pair_id: {text_overlap['shared_text_key_count']}"
        )

    print("\nfeature_selection vs IPI")
    for item in report["feature_selection_vs_ipi"]:
        print(
            f"  {item['left']} vs {item['right']}: shared_text_count={item['shared_text_count']}"
        )

    union = report["validation_test_full_union"]
    print("\nvalidation + test vs full IPI file")
    print(f"  union_equals_full: {union['union_equals_full']}")
    print(
        f"  validation/test text overlap: {union['validation_test_text_overlap_count']}"
    )
    print(
        "  pair_id overlaps with different text: "
        f"{report['pair_id_overlap_with_different_text_count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root containing resources/*.csv",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full JSON report",
    )
    args = parser.parse_args()

    report = build_report(args.project_root)
    _print_summary(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nJSON report written to: {args.json_out}")


if __name__ == "__main__":
    main()
