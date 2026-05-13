from __future__ import annotations

"""Deterministic experiment/run and artifact name builders."""


# Kept in sync with `src/utils/intervention_hooks.py`. Hard-coding the default
# here (instead of importing) keeps this module dependency-free, as documented.
_DEFAULT_SCOPE = "prompt_without_buffer"


def _normalize(value: str) -> str:
    """Normalize free-text identifiers into stable slug-like fragments."""
    return value.strip().replace("/", "-").replace(" ", "-")


def _scope_suffix(scope: str | None, last_k: int | None) -> str:
    """Return the optional `__{scope}__lastk{last_k}` fragment.

    Returns an empty string when scope is None or equals the legacy default,
    so existing run_ids and artifact names stay byte-identical on disk.
    """
    if scope is None:
        return ""
    suffix = f"__{_normalize(scope)}"
    if last_k is not None:
        suffix = f"{suffix}__lastk{int(last_k)}"
    return suffix


def make_run_id(
    model_name: str,
    split_id: str,
    direction: str | None,
    top_k: int | None,
    n_trials: int | None,
    seed: int,
    condition: str | None = None,
    scope: str | None = None,
    last_k: int | None = None,
) -> str:
    parts: list[str] = [_normalize(model_name), _normalize(split_id)]
    if condition:
        parts.append(_normalize(condition))
    if direction:
        parts.append(_normalize(direction))
    if top_k is not None:
        parts.append(f"k{int(top_k)}")
    if n_trials is not None:
        parts.append(f"trials{int(n_trials)}")
    parts.append(f"seed{int(seed)}")
    base = "__".join(parts)
    return f"{base}{_scope_suffix(scope, last_k)}"


def make_activation_artifact_name(model_name: str, split_id: str, layers: str) -> str:
    return f"activations-{_normalize(model_name)}-{_normalize(split_id)}-{_normalize(layers)}"


def make_feature_ranking_artifact_name(model_name: str, split_id: str, ranking_top_n: int) -> str:
    return f"feature-ranking-{_normalize(model_name)}-{_normalize(split_id)}-top{int(ranking_top_n)}"


def make_multiplier_artifact_name(
    model_name: str,
    split_id: str,
    direction: str,
    top_k: int,
    n_trials: int,
    seed: int,
    scope: str | None = None,
    last_k: int | None = None,
) -> str:
    return (
        f"multipliers-{_normalize(model_name)}-{_normalize(split_id)}-"
        f"{_normalize(direction)}-k{int(top_k)}-trials{int(n_trials)}-seed{int(seed)}"
        f"{_scope_suffix(scope, last_k)}"
    )


def make_likert_artifact_name(
    model_name: str,
    split_id: str,
    condition: str,
    seed: int,
    direction: str | None = None,
    top_k: int | None = None,
    n_trials: int | None = None,
    scope: str | None = None,
    last_k: int | None = None,
) -> str:
    condition_norm = _normalize(condition)
    if condition_norm == "baseline":
        # Baseline never sees the intervention, so scope is intentionally
        # excluded from the baseline artifact name — preserves baseline reuse
        # across scope variants.
        return f"likert-baseline-{_normalize(model_name)}-{_normalize(split_id)}-seed{int(seed)}"
    if condition_norm == "intervened":
        if direction is None or top_k is None or n_trials is None:
            raise ValueError(
                "Intervened likert artifact name requires direction, top_k, and n_trials."
            )
        return (
            f"likert-intervened-{_normalize(model_name)}-{_normalize(split_id)}-"
            f"{_normalize(direction)}-k{int(top_k)}-trials{int(n_trials)}-seed{int(seed)}"
            f"{_scope_suffix(scope, last_k)}"
        )
    raise ValueError(
        f"Unsupported condition={condition!r}. Expected 'baseline' or 'intervened'."
    )


if __name__ == "__main__":
    demo = {
        "run_id": make_run_id(
            model_name="gemma-3-4b",
            split_id="three_way_split_v1",
            direction="minimize",
            top_k=80,
            n_trials=3000,
            seed=42,
        ),
        "run_id_with_scope": make_run_id(
            model_name="gemma-3-4b",
            split_id="three_way_split_v1",
            direction="minimize",
            top_k=16,
            n_trials=1000,
            seed=42,
            scope="prompt_last_token",
            last_k=3,
        ),
        "activations": make_activation_artifact_name(
            model_name="gemma-3-4b",
            split_id="three_way_split_v1",
            layers="all",
        ),
        "feature_ranking": make_feature_ranking_artifact_name(
            model_name="gemma-3-4b",
            split_id="three_way_split_v1",
            ranking_top_n=256,
        ),
        "multipliers": make_multiplier_artifact_name(
            model_name="gemma-3-4b",
            split_id="three_way_split_v1",
            direction="minimize",
            top_k=80,
            n_trials=3000,
            seed=42,
        ),
        "multipliers_with_scope": make_multiplier_artifact_name(
            model_name="gemma-3-4b",
            split_id="three_way_split_v1",
            direction="minimize",
            top_k=16,
            n_trials=1000,
            seed=42,
            scope="prompt_last_token",
            last_k=3,
        ),
        "likert_baseline": make_likert_artifact_name(
            model_name="gemma-3-4b",
            split_id="three_way_split_v1",
            condition="baseline",
            seed=42,
        ),
        "likert_intervened": make_likert_artifact_name(
            model_name="gemma-3-4b",
            split_id="three_way_split_v1",
            condition="intervened",
            direction="minimize",
            top_k=80,
            n_trials=3000,
            seed=42,
        ),
        "likert_intervened_with_scope": make_likert_artifact_name(
            model_name="gemma-3-4b",
            split_id="three_way_split_v1",
            condition="intervened",
            direction="minimize",
            top_k=16,
            n_trials=1000,
            seed=42,
            scope="prompt_last_token",
            last_k=3,
        ),
    }
    for key, value in demo.items():
        print(f"{key}={value}")
