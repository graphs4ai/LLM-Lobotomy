"""Shared TransformerLens hook builders for activation interventions.

This module centralizes the position-masking logic that all wrappers used to
inline as `buffer_size = 3 / prompt_len = input_ids.shape[-1] - buffer_size`.
Promoting that hidden design choice to an explicit `intervention_scope` axis
lets the codebase test:

    prompt_all              -> every prompt position is modified
    prompt_without_buffer   -> all positions except the last `last_k` (legacy)
    prompt_last_token       -> only the final prompt position
    prompt_last_k           -> the last `last_k` prompt positions
    generated_only          -> only positions produced during generation
    all_tokens              -> every position seen by the hook

The hook is safe for both full-sequence recompute (the default TransformerLens
generate path: each step sees seq_len = input_len + n_generated) AND cached
decoding (each step after prefill sees seq_len == 1). For the cached case we
cannot tell from inside a single hook whether the lone position is a prompt
token or a generated one, so we apply the conservative convention used in the
project's design proposal: treat the lone position as a generated token unless
the scope explicitly does not want it modified.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch


DEFAULT_SCOPE: str = "prompt_without_buffer"
DEFAULT_LAST_K: int = 3

VALID_SCOPES: tuple[str, ...] = (
    "prompt_all",
    "prompt_without_buffer",
    "prompt_last_token",
    "prompt_last_k",
    "generated_only",
    "all_tokens",
)


def assert_scope(scope: str) -> None:
    """Raise a clear error if `scope` is not one of the supported values."""
    if scope not in VALID_SCOPES:
        raise ValueError(
            f"Unknown intervention_scope={scope!r}. "
            f"Expected one of: {', '.join(VALID_SCOPES)}."
        )


def _build_mask(
    seq_len: int,
    input_len: int,
    scope: str,
    last_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute the boolean per-position mask for a single forward pass.

    The caller must already have validated `scope` via `assert_scope`.
    """
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    if scope == "prompt_all":
        end = min(input_len, seq_len)
        if end > 0:
            mask[:end] = True

    elif scope == "prompt_without_buffer":
        end = max(0, min(input_len - last_k, seq_len))
        if end > 0:
            mask[:end] = True

    elif scope == "prompt_last_token":
        if seq_len >= input_len and input_len > 0:
            mask[input_len - 1] = True
        # If seq_len == 1 (cached decoding step), the lone position is most
        # likely a generated token, not the final prompt token, so leave it
        # alone. The first forward pass (seq_len == input_len) handles the
        # actual final-prompt modification.

    elif scope == "prompt_last_k":
        if seq_len >= input_len and input_len > 0:
            start = max(0, input_len - last_k)
            end = input_len
            mask[start:end] = True

    elif scope == "generated_only":
        if seq_len > input_len:
            mask[input_len:] = True
        elif seq_len == 1:
            # Cached decoding step after prefill: lone position is generated.
            mask[:] = True

    elif scope == "all_tokens":
        mask[:] = True

    return mask


def make_intervention_hook(
    neuron_mults: Dict[int, float],
    input_len: int,
    scope: str = DEFAULT_SCOPE,
    last_k: int = DEFAULT_LAST_K,
    debug_seq_lens: Optional[List[int]] = None,
) -> Callable[[torch.Tensor, object], torch.Tensor]:
    """Build a TransformerLens forward hook that scales selected neurons.

    Args:
        neuron_mults: mapping of neuron index -> multiplier value for the
            current layer.
        input_len: prompt length captured before generation begins. Used to
            distinguish prompt positions from generated ones.
        scope: one of VALID_SCOPES (see module docstring).
        last_k: width of the trailing prompt window for the
            `prompt_without_buffer` and `prompt_last_k` scopes.
        debug_seq_lens: optional list; when provided, the hook will append the
            observed `seq_len` of each forward pass (capped at 20 entries) so
            callers can inspect whether generation uses full recompute or
            cached decoding.
    """
    assert_scope(scope)
    if last_k < 0:
        raise ValueError(f"last_k must be >= 0, got {last_k!r}.")
    if input_len < 0:
        raise ValueError(f"input_len must be >= 0, got {input_len!r}.")

    def hook(resid_pre: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        seq_len = int(resid_pre.shape[1])
        if debug_seq_lens is not None and len(debug_seq_lens) < 20:
            debug_seq_lens.append(seq_len)

        if not neuron_mults:
            return resid_pre

        mask = _build_mask(
            seq_len=seq_len,
            input_len=input_len,
            scope=scope,
            last_k=last_k,
            device=resid_pre.device,
        )
        if not bool(mask.any()):
            return resid_pre

        modified = resid_pre.clone()
        for n, m in neuron_mults.items():
            modified[:, mask, n] = modified[:, mask, n] * m
        return modified

    return hook


__all__ = [
    "DEFAULT_SCOPE",
    "DEFAULT_LAST_K",
    "VALID_SCOPES",
    "assert_scope",
    "make_intervention_hook",
]
