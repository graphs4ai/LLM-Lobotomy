import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from typing import List, Union, Optional, Dict, Tuple

from utils.intervention_hooks import (
    DEFAULT_LAST_K,
    DEFAULT_SCOPE,
    assert_scope,
    make_intervention_hook,
)


class Llama3dot1Wrapper:
    """
    A wrapper to load a HookedTransformer model and retrieve layer activations 
    using forward hooks.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda", dtype: torch.dtype = torch.float16, n_devices: int = 1):
        """
        Initializes the model, mimicking the configuration in the provided files.

        Args:
            model_name (str): The name of the model to load.
            device (str): The device (e.g., 'cuda', 'cpu') to load the model onto.
            dtype (torch.dtype): Model precision (torch.float16 or torch.bfloat16).
            n_devices (int): Number of GPUs to split the model across (model parallelism).
                             When > 1, transformer blocks are distributed across GPUs.
        """
        self.device = device
        self.n_devices = n_devices

        # Load the base model using HookedTransformer
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            dtype=dtype,
            n_devices=n_devices,
        )

        # When using multi-GPU, input must go to the first device explicitly
        self.input_device = "cuda:0" if n_devices > 1 else device

        # Ensure a pad token exists for masking if needed
        if self.model.tokenizer.pad_token_id is None:
            self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id

        # Store number of layers for 'all' option
        self.n_layers = self.model.cfg.n_layers

    def get_layer_activations(
        self,
        tokens: torch.Tensor,
        layers: Union[List[int], str] = [19],
        activation_multipliers: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Runs a forward pass and returns the residual stream (resid_pre) activations 
        for the specified layers, concatenated along the feature dimension.

        Args:
            tokens (torch.Tensor): A tensor of token IDs, shape [batch, seq_len].
            layers (Union[List[int], str]): List of layer indices to retrieve activations from,
                                            or 'all' to get activations from all layers.
            activation_multipliers (Optional[Dict[str, float]]): Dictionary mapping neuron identifiers
                                                                  (format: 'layer_{L}-neuron_{N}') to 
                                                                  multiplier values. If provided,
                                                                  specific neurons will be multiplied by
                                                                  the corresponding factor before propagating
                                                                  to subsequent layers.
                                                                  Example: {'layer_10-neuron_512': 0.5, 'layer_15-neuron_100': 2.0}

        Returns:
            torch.Tensor: The concatenated activation tensor, shape [batch, seq_len, n_layers * d_model], 
                          stored on the CPU.

        Raises:
            ValueError: If layers is invalid.
            RuntimeError: If any activation could not be retrieved.
        """
        # Handle 'all' option
        if isinstance(layers, str):
            if layers.lower() == 'all':
                layers = list(range(self.n_layers))
            else:
                raise ValueError(
                    f"Invalid layers string: '{layers}'. Use 'all' or a list of integers.")

        if not layers:
            raise ValueError("layers list cannot be empty")

        # Default to empty dict if no multipliers provided
        if activation_multipliers is None:
            activation_multipliers = {}

        # Parse neuron-wise multipliers into per-layer dictionaries
        # Format: {'layer_10-neuron_512': 0.5} -> {10: {512: 0.5}}
        layer_neuron_multipliers = {}
        for feature_name, multiplier in activation_multipliers.items():
            # Parse 'layer_X-neuron_Y' format
            parts = feature_name.split('-')
            layer_idx = int(parts[0].split('_')[1])
            neuron_idx = int(parts[1].split('_')[1])
            if layer_idx not in layer_neuron_multipliers:
                layer_neuron_multipliers[layer_idx] = {}
            layer_neuron_multipliers[layer_idx][neuron_idx] = multiplier

        # Dictionary to store the activations captured by hooks
        activations = {}

        # get_layer_activations operates on a fixed input (no generation),
        # so all positions are "prompt" positions. We apply multipliers to all
        # positions here, which is correct and consistent with get_soft_stance_score.
        def make_layer_hook(layer_idx: int, neuron_multipliers: Optional[Dict[int, float]] = None):
            def layer_hook(resid_pre: torch.Tensor, hook):
                # resid_pre: [batch, seq, d_model] - This is the input to the attention/MLP block.
                # Apply neuron-specific multipliers if provided
                if neuron_multipliers:
                    modified = resid_pre.clone()
                    for neuron_idx, multiplier in neuron_multipliers.items():
                        modified[:, :, neuron_idx] = modified[:,
                                                              :, neuron_idx] * multiplier
                    # Store the modified activation
                    activations[layer_idx] = modified.detach().clone().cpu()
                    # Return modified tensor to propagate changes to subsequent layers
                    return modified
                else:
                    # Store a copy of the tensor for external use
                    activations[layer_idx] = resid_pre.detach().clone().cpu()
                    return resid_pre
            return layer_hook

        # Determine all layers that need hooks (requested layers + intervention layers)
        intervention_layers = set(layer_neuron_multipliers.keys())
        all_hook_layers = set(layers) | intervention_layers

        # Create hook points for all needed layers
        fwd_hooks = []
        for layer in all_hook_layers:
            hook_point = f"blocks.{layer}.hook_resid_pre"
            neuron_mults = layer_neuron_multipliers.get(layer, None)
            fwd_hooks.append(
                (hook_point, make_layer_hook(layer, neuron_mults)))

        # Run the forward pass with hooks
        # Need to run through at least the max layer we care about
        stop_at_layer = max(max(layers), max(all_hook_layers)) + 1

        with torch.no_grad():
            self.model.run_with_hooks(
                tokens.to(self.input_device),
                fwd_hooks=fwd_hooks,
                stop_at_layer=stop_at_layer
            )

        # Verify all activations were captured
        for layer in layers:
            if layer not in activations:
                raise RuntimeError(
                    f"Could not retrieve activation from hook point blocks.{layer}.hook_resid_pre")

        # Concatenate activations from all layers along the feature dimension
        # Each activation is [batch, seq_len, d_model]
        # Result is [batch, seq_len, n_layers * d_model]
        layer_tensors = [activations[layer] for layer in sorted(layers)]
        concatenated = torch.cat(layer_tensors, dim=-1)

        return concatenated

    def generate_with_intervention(
        self,
        input_ids: torch.Tensor,
        activation_multipliers: Optional[Dict[str, float]] = None,
        max_new_tokens: int = 10,
        temperature: Optional[float] = None,
        do_sample: bool = False,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        verbose: bool = False,
        intervention_scope: str = DEFAULT_SCOPE,
        last_k: int = DEFAULT_LAST_K,
        debug_seq_lens: Optional[List[int]] = None,
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Generates text with optional activation interventions applied during the forward pass.

        Which positions are modified is controlled by `intervention_scope`
        (see `src/utils/intervention_hooks.py` for the full set). Default
        `prompt_without_buffer` preserves the legacy `buffer_size = 3` behavior.

        Args:
            input_ids: Input token IDs, shape [batch, seq_len].
            activation_multipliers: Dictionary mapping neuron identifiers
                (format: 'layer_{L}-neuron_{N}') to multiplier values.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. If None, greedy decoding is used.
            do_sample: Whether to use sampling instead of greedy decoding.
            stop_at_eos: Whether to stop generation at EOS token.
            eos_token_id: EOS token ID to stop at. If None, uses tokenizer default.
            verbose: Whether to show generation progress.
            intervention_scope: which token positions receive the multiplier.
            last_k: trailing-prompt window for scopes that use it.
            debug_seq_lens: optional list; when provided, observed `resid_pre`
                seq lengths (up to 20) are appended so the caller can inspect
                whether generation uses full recompute or cached decoding.
            **generate_kwargs: Additional arguments passed to model.generate().

        Returns:
            torch.Tensor: Generated token IDs including the input tokens.
        """
        if eos_token_id is None:
            eos_token_id = self.model.tokenizer.eos_token_id

        if activation_multipliers is None or len(activation_multipliers) == 0:
            return self.model.generate(
                input_ids.to(self.input_device),
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature is not None else 1.0,
                do_sample=do_sample,
                stop_at_eos=stop_at_eos,
                eos_token_id=eos_token_id,
                verbose=verbose,
                **generate_kwargs
            )

        assert_scope(intervention_scope)

        layer_neuron_multipliers: Dict[int, Dict[int, float]] = {}
        for feature_name, multiplier in activation_multipliers.items():
            parts = feature_name.split('-')
            layer_idx = int(parts[0].split('_')[1])
            neuron_idx = int(parts[1].split('_')[1])
            if layer_idx not in layer_neuron_multipliers:
                layer_neuron_multipliers[layer_idx] = {}
            layer_neuron_multipliers[layer_idx][neuron_idx] = multiplier

        input_len = int(input_ids.shape[-1])
        fwd_hooks = [
            (
                f"blocks.{layer_idx}.hook_resid_pre",
                make_intervention_hook(
                    neuron_mults=neuron_mults,
                    input_len=input_len,
                    scope=intervention_scope,
                    last_k=last_k,
                    debug_seq_lens=debug_seq_lens,
                ),
            )
            for layer_idx, neuron_mults in layer_neuron_multipliers.items()
        ]

        with torch.no_grad():
            for hook_point, hook_fn in fwd_hooks:
                self.model.add_hook(hook_point, hook_fn)

            try:
                output_ids = self.model.generate(
                    input_ids.to(self.input_device),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature is not None else 1.0,
                    do_sample=do_sample,
                    stop_at_eos=stop_at_eos,
                    eos_token_id=eos_token_id,
                    verbose=verbose,
                    **generate_kwargs
                )
            finally:
                self.model.reset_hooks()

        return output_ids

    def get_stance_token_ids(self, language: str = "pt") -> Tuple[int, int]:
        """
        Gets the token IDs for positive (Agree) and negative (Disagree) stance words.

        Note: Llama 3 tokenizer is space-sensitive. The first token of a response
        typically includes a leading space.

        Args:
            language: Language code ("pt" for Portuguese, "en" for English)

        Returns:
            Tuple of (positive_token_id, negative_token_id)
        """
        if language == "pt":
            # Portuguese: "Concordo" (Agree) and "Discordo" (Disagree)
            positive_word = "Con"
            negative_word = "Dis"
        else:
            # English: "Agree" and "Disagree"
            positive_word = "Agree"
            negative_word = "Disagree"

        # Encode and take the first token ID (the word itself)
        positive_tokens = self.model.tokenizer.encode(
            positive_word, add_special_tokens=False)
        negative_tokens = self.model.tokenizer.encode(
            negative_word, add_special_tokens=False)

        # The first token should be the stance word
        positive_token_id = positive_tokens[0]
        negative_token_id = negative_tokens[0]

        return positive_token_id, negative_token_id

    def get_soft_stance_score(
        self,
        input_ids: torch.Tensor,
        activation_multipliers: Optional[Dict[str, float]] = None,
        positive_token_id: Optional[int] = None,
        negative_token_id: Optional[int] = None,
        language: str = "pt",
        intervention_scope: str = DEFAULT_SCOPE,
        last_k: int = DEFAULT_LAST_K,
    ) -> Tuple[float, float]:
        """
        Computes a continuous score [-1, 1] representing the probability gap
        between positive (Agree) and negative (Disagree) tokens at the last position.

        This uses a single forward pass (no generation) for efficient optimization.
        Position scope of the intervention is controlled by `intervention_scope`
        and `last_k`; see `src/utils/intervention_hooks.py`.

        Args:
            input_ids: Input token IDs, shape [batch, seq_len] or [seq_len]
            activation_multipliers: Optional dict mapping neuron identifiers
                (format: 'layer_{L}-neuron_{N}') to multiplier values
            positive_token_id: Token ID for positive stance word. If None, uses language default.
            negative_token_id: Token ID for negative stance word. If None, uses language default.
            language: Language code for default token IDs ("pt" or "en")
            intervention_scope: which token positions receive the multiplier.
            last_k: trailing-prompt window for scopes that use it.

        Returns:
            Tuple of (score, prob_sum).
        """
        if positive_token_id is None or negative_token_id is None:
            pos_id, neg_id = self.get_stance_token_ids(language)
            positive_token_id = positive_token_id or pos_id
            negative_token_id = negative_token_id or neg_id

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_ids = input_ids.to(self.input_device)

        if activation_multipliers is None or len(activation_multipliers) == 0:
            with torch.no_grad():
                logits = self.model(input_ids)
        else:
            assert_scope(intervention_scope)
            layer_neuron_multipliers: Dict[int, Dict[int, float]] = {}
            for feature_name, multiplier in activation_multipliers.items():
                parts = feature_name.split('-')
                layer_idx = int(parts[0].split('_')[1])
                neuron_idx = int(parts[1].split('_')[1])
                if layer_idx not in layer_neuron_multipliers:
                    layer_neuron_multipliers[layer_idx] = {}
                layer_neuron_multipliers[layer_idx][neuron_idx] = multiplier

            input_len = int(input_ids.shape[1])
            fwd_hooks = [
                (
                    f"blocks.{layer_idx}.hook_resid_pre",
                    make_intervention_hook(
                        neuron_mults=neuron_mults,
                        input_len=input_len,
                        scope=intervention_scope,
                        last_k=last_k,
                    ),
                )
                for layer_idx, neuron_mults in layer_neuron_multipliers.items()
            ]

            with torch.no_grad():
                logits = self.model.run_with_hooks(
                    input_ids, fwd_hooks=fwd_hooks)

        last_token_logits = logits[0, -1, :]
        probs = F.softmax(last_token_logits, dim=-1)

        prob_positive = probs[positive_token_id].item()
        prob_negative = probs[negative_token_id].item()

        score = prob_positive - prob_negative
        prob_sum = prob_positive + prob_negative

        return score, prob_sum
