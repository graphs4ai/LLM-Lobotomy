import optuna
from optuna.samplers import TPESampler, CmaEsSampler
import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Generator
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import os
import sys
import json
from datetime import datetime
import wandb

from model_factory import get_model_wrapper
from likert_scale_test import (
    run_likert_test_streaming,
    compute_kl_divergence,
    LIKERT_SCALE,
    create_likert_prompt,
    format_chat_prompt
)


def _ranked_feature_to_neuron_name(entry: dict[str, Any]) -> str:
    """Convert ranked feature entries into `layer_X-neuron_Y` names."""
    feature_name = entry.get("feature_name")
    if feature_name:
        return str(feature_name)

    layer = entry.get("layer")
    feature = entry.get("feature")
    if layer is None or feature is None:
        raise ValueError(
            "Each ranked feature must provide either feature_name or both layer and feature."
        )
    return f"layer_{int(layer)}-neuron_{int(feature)}"


class TeeOutput:
    """
    Duplicates output to both a file and the original stream (stdout/stderr).
    This captures all terminal output including print() statements.
    """

    def __init__(self, filepath: str, stream):
        self.filepath = filepath
        self.stream = stream
        self.file = open(filepath, 'a', encoding='utf-8')

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def isatty(self):
        """Check if the underlying stream is a TTY."""
        return self.stream.isatty() if hasattr(self.stream, 'isatty') else False

    def fileno(self):
        """Return the file descriptor of the underlying stream."""
        return self.stream.fileno() if hasattr(self.stream, 'fileno') else -1

    @property
    def encoding(self):
        """Return the encoding of the underlying stream."""
        return getattr(self.stream, 'encoding', 'utf-8')


class OutputLogger:
    """
    Context manager to capture all stdout/stderr to a log file.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.original_stdout = None
        self.original_stderr = None
        self.tee_stdout = None
        self.tee_stderr = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(
                f"=== Optimization Log Started: {datetime.now().isoformat()} ===\n\n")

        # Redirect stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.tee_stdout = TeeOutput(self.log_path, self.original_stdout)
        self.tee_stderr = TeeOutput(self.log_path, self.original_stderr)
        sys.stdout = self.tee_stdout
        sys.stderr = self.tee_stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Write footer
        print(
            f"\n=== Optimization Log Ended: {datetime.now().isoformat()} ===")

        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.tee_stdout is None or self.tee_stderr is None:
            raise RuntimeError(
                "TeeOutput instances were not properly initialized")
        self.tee_stdout.close()
        self.tee_stderr.close()
        return False


def build_multipliers_from_trial(
    trial: optuna.Trial,
    target_neurons: List[str],
    bounds: Tuple[float, float]
) -> Dict[str, float]:
    """
    Builds activation multipliers dictionary from Optuna trial suggestions.

    Args:
        trial: Optuna trial object
        target_neurons: List of neuron identifiers (format: 'layer_X-neuron_Y')
        bounds: Tuple of (min, max) bounds for multiplier values

    Returns:
        Dictionary mapping neuron identifiers to suggested multiplier values
    """
    multipliers = {}
    for neuron in target_neurons:
        multipliers[neuron] = trial.suggest_float(
            neuron,
            bounds[0],
            bounds[1]
        )
    return multipliers


def soft_objective(
    trial: optuna.Trial,
    wrapper,
    questions_df: pd.DataFrame,
    target_neurons: List[str],
    bounds: Tuple[float, float],
    positive_token_id: int,
    negative_token_id: int,
    language: str = "pt",
    use_absolute: bool = False,
    direction: str = "maximize"
) -> float:
    """
    Soft objective function using logit differences instead of discrete PI.

    This provides a continuous gradient signal to the optimizer by computing
    the probability gap between "Agree" and "Disagree" tokens at the first
    response position.

    The soft metric for each pair is:
        pair_diff = soft_score(P+) - soft_score(P-)

    where soft_score = P("Concordo") - P("Discordo") at the first token position.

    Args:
        trial: Optuna trial object
        wrapper: LLaMA model wrapper
        questions_df: DataFrame with Likert questions (must have 'pergunta', 'pair_id', 'tipo_pergunta')
        target_neurons: List of neuron identifiers to optimize
        bounds: Multiplier bounds (min, max)
        positive_token_id: Token ID for positive stance word ("Concordo"/"Agree")
        negative_token_id: Token ID for negative stance word ("Discordo"/"Disagree")
        language: Prompt language
        use_absolute: If True, return abs(avg_signed_pi) instead of the signed value
        direction: Optimization direction ('maximize' or 'minimize')

    Returns:
        Average soft PI (signed or absolute depending on use_absolute)
    """
    # Build multipliers from trial suggestions
    multipliers = build_multipliers_from_trial(trial, target_neurons, bounds)

    # Get unique pair IDs
    pair_ids = sorted(questions_df['pair_id'].unique())

    # Sum SIGNED differences to ensure ideological consistency
    total_signed_score = 0.0
    valid_pairs = 0

    for pair_id in pair_ids:
        pair_data = questions_df[questions_df['pair_id'] == pair_id]

        soft_p_plus = None
        soft_p_minus = None

        # Process P+ and P- questions for this pair
        for _, row in pair_data.iterrows():
            statement = row['pergunta']
            tipo = row['tipo_pergunta']

            # Create and format prompt
            user_message = create_likert_prompt(statement, language)
            prompt = format_chat_prompt(
                wrapper.model.tokenizer, user_message, language)

            # Tokenize
            if wrapper.model.tokenizer is None:
                raise RuntimeError(
                    "Tokenizer is not initialized in the model wrapper")

            tokenized = wrapper.model.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=1024
            )
            input_ids = tokenized['input_ids']

            # Get soft stance score (single forward pass, no generation)
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError(
                    f"Expected input_ids to be torch.Tensor, got {type(input_ids).__name__}")

            soft_score, prob_sum = wrapper.get_soft_stance_score(
                input_ids=input_ids,
                activation_multipliers=multipliers,
                positive_token_id=positive_token_id,
                negative_token_id=negative_token_id,
                language=language
            )

            # Store based on question type
            if tipo == 'P+':
                soft_p_plus = soft_score
            elif tipo == 'P-':
                soft_p_minus = soft_score

        # Compute SIGNED soft PI for this pair (DO NOT use abs() here)
        # This ensures ideological consistency across questions
        if soft_p_plus is not None and soft_p_minus is not None:
            # If model is right-wing: P+ agreement high, P- agreement low → positive
            # If model is left-wing: P+ agreement low, P- agreement high → negative
            pair_diff = soft_p_plus - soft_p_minus
            total_signed_score += pair_diff
            valid_pairs += 1

    if valid_pairs == 0:
        return 0.0

    # Calculate average SIGNED soft PI
    avg_signed_pi = total_signed_score / valid_pairs

    if np.isnan(avg_signed_pi) or np.isinf(avg_signed_pi):
        return 0.0

    if use_absolute:
        return abs(avg_signed_pi)
    return avg_signed_pi


def objective(
    trial: optuna.Trial,
    wrapper,  # Llama3dot1Wrapper or Gemma3Wrapper
    questions_df: pd.DataFrame,
    baseline_scores: List[int],
    target_neurons: List[str],
    bounds: Tuple[float, float],
    language: str = "pt",
    max_new_tokens: int = 10,
    temperature: float = 0.0
) -> Tuple[float, float]:
    """
    Multi-objective function for Optuna optimization.

    Objectives:
        1. Maximize Polarization Index (PI)
        2. Minimize KL Divergence from baseline

    Note: Pruning is not supported for multi-objective optimization in Optuna,
    so we run the full evaluation for each trial.

    Args:
        trial: Optuna trial object
        wrapper: LLaMA model wrapper
        questions_df: DataFrame with Likert questions
        baseline_scores: List of baseline Likert scores for KL computation
        target_neurons: List of neuron identifiers to optimize
        bounds: Multiplier bounds (min, max)
        language: Prompt language
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (polarization_index, kl_divergence)
    """
    # Build multipliers from trial suggestions
    multipliers = build_multipliers_from_trial(trial, target_neurons, bounds)

    # Track PI and scores
    running_pi_sum = 0.0
    valid_pairs_count = 0
    intervention_scores = []

    # Stream through question pairs
    pair_generator = run_likert_test_streaming(
        wrapper=wrapper,
        questions_df=questions_df,
        language=language,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        activation_multipliers=multipliers,
        verbose=False
    )

    for pair_result in pair_generator:
        # Collect scores for KL divergence
        if pair_result['p_plus_score'] is not None:
            intervention_scores.append(pair_result['p_plus_score'])
        if pair_result['p_minus_score'] is not None:
            intervention_scores.append(pair_result['p_minus_score'])

        # Update running PI
        if pair_result['valid']:
            running_pi_sum += pair_result['polarization_index']
            valid_pairs_count += 1

    # Compute final metrics
    if valid_pairs_count == 0:
        # No valid pairs - return worst possible values
        return float('-inf'), float('inf')

    final_pi = running_pi_sum / valid_pairs_count
    kl_div = compute_kl_divergence(baseline_scores, intervention_scores)

    return final_pi, kl_div


def run_baseline(
    wrapper,  # Llama3dot1Wrapper or Gemma3Wrapper
    questions_df: pd.DataFrame,
    language: str = "pt",
    max_new_tokens: int = 10,
    temperature: float = 0.0
) -> Tuple[List[int], float]:
    """
    Runs baseline evaluation without interventions.

    Args:
        wrapper: Model wrapper (Llama or Gemma)
        questions_df: DataFrame with questions
        language: Prompt language
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        Tuple of (baseline_scores, baseline_pi)
    """
    print("Running baseline evaluation (no intervention)...")

    baseline_scores = []
    pair_results = []

    for pair_result in run_likert_test_streaming(
        wrapper=wrapper,
        questions_df=questions_df,
        language=language,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        activation_multipliers=None,
        verbose=True
    ):
        if pair_result['p_plus_score'] is not None:
            baseline_scores.append(pair_result['p_plus_score'])
        if pair_result['p_minus_score'] is not None:
            baseline_scores.append(pair_result['p_minus_score'])
        pair_results.append(pair_result)

    # Compute baseline PI
    valid_pis = [p['polarization_index'] for p in pair_results if p['valid']]
    baseline_pi = sum(valid_pis) / len(valid_pis) if valid_pis else 0.0

    print(f"Baseline PI: {baseline_pi:.4f}")
    print(f"Baseline scores collected: {len(baseline_scores)}")

    return baseline_scores, baseline_pi


def sample_questions(
    questions_df: pd.DataFrame,
    n_pairs: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Samples a subset of question pairs for fast mode.

    Args:
        questions_df: Full questions DataFrame
        n_pairs: Number of pairs to sample
        random_state: Random seed for reproducibility

    Returns:
        Sampled DataFrame with complete pairs
    """
    np.random.seed(random_state)

    all_pair_ids = questions_df['pair_id'].unique()
    n_pairs = min(n_pairs, len(all_pair_ids))

    sampled_pair_ids = np.random.choice(
        all_pair_ids, size=n_pairs, replace=False)
    sampled_df = questions_df[questions_df['pair_id'].isin(
        sampled_pair_ids)].copy()

    return sampled_df


def save_optimization_results(
    study: optuna.Study,
    output_dir: str,
    baseline_pi: float,
    config: Dict[Any, Any],
    baseline_soft_score: Optional[float] = None,
    use_soft_metric: bool = False,
    soft_metrics: Optional[Dict[str, float]] = None,
) -> str:
    """
    Saves optimization results to JSON file.

    Args:
        study: Completed Optuna study
        output_dir: Output directory
        baseline_pi: Baseline polarization index
        config: Optimization configuration
        baseline_soft_score: Baseline soft score (for soft metric mode)
        use_soft_metric: Whether soft metric optimization was used

    Returns:
        Path to saved results file
    """
    os.makedirs(output_dir, exist_ok=True)

    if use_soft_metric:
        # Single-objective soft metric results
        best_trial = study.best_trial
        results = {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'baseline_pi': baseline_pi,
            'baseline_soft_score': baseline_soft_score,
            'optimization_mode': 'soft_metric',
            'config': config,
            'soft_metrics': soft_metrics or {},
            'best_trial': {
                'trial_number': best_trial.number,
                'soft_score': best_trial.value,
                'multipliers': best_trial.params
            },
            'all_trials': [
                {
                    'number': t.number,
                    'state': str(t.state),
                    'value': t.value if t.value is not None else None,
                    'params': t.params
                }
                for t in study.trials
            ]
        }
    else:
        # Multi-objective Pareto front results
        pareto_trials = study.best_trials
        results = {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'baseline_pi': baseline_pi,
            'optimization_mode': 'multi_objective',
            'config': config,
            'soft_metrics': soft_metrics or {},
            'pareto_front': [
                {
                    'trial_number': t.number,
                    'values': {
                        'polarization_index': t.values[0],
                        'kl_divergence': t.values[1]
                    },
                    'multipliers': t.params
                }
                for t in pareto_trials
            ],
            'all_trials': [
                {
                    'number': t.number,
                    'state': str(t.state),
                    'values': t.values if t.values else None,
                    'params': t.params
                }
                for t in study.trials
            ]
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        output_dir, f"optimization_results_{timestamp}.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    return results_path


def print_pareto_front(study: optuna.Study, baseline_pi: float):
    """
    Prints the Pareto front solutions (for multi-objective optimization).

    Args:
        study: Completed Optuna study
        baseline_pi: Baseline polarization index for comparison
    """
    print("\n" + "=" * 70)
    print("PARETO FRONT SOLUTIONS")
    print("=" * 70)
    print(f"Baseline PI: {baseline_pi:.4f}")
    print("-" * 70)

    pareto_trials = study.best_trials

    for i, trial in enumerate(pareto_trials):
        pi, kl = trial.values
        pi_delta = pi - baseline_pi

        print(f"\nSolution {i + 1} (Trial #{trial.number}):")
        print(f"  Polarization Index: {pi:.4f} (Δ = {pi_delta:+.4f})")
        print(f"  KL Divergence:      {kl:.4f}")
        print(f"  Multipliers:")
        for neuron, mult in trial.params.items():
            print(f"    {neuron}: {mult:.4f}")

    print("\n" + "=" * 70)


def print_best_soft_trial(
    study: optuna.Study,
    baseline_soft_score: float,
    objective_mode: str = "absolute",
    direction: str = "maximize"
):
    """
    Prints the best trial for soft metric single-objective optimization.

    Args:
        study: Completed Optuna study
        baseline_soft_score: Baseline soft score for comparison
        objective_mode: 'signed' or 'absolute'
        direction: 'maximize' or 'minimize'
    """
    mode_label = f"{direction.upper()} / {objective_mode.upper()}"
    print("\n" + "=" * 70)
    print(f"BEST SOFT METRIC SOLUTION ({mode_label})")
    print("=" * 70)

    score_label = "|Soft Score|" if objective_mode == "absolute" else "Signed Soft Score"
    print(f"Baseline {score_label}: {baseline_soft_score:.6f}")
    print("-" * 70)

    best_trial = study.best_trial
    best_value = best_trial.value if best_trial.value is not None else float(
        'nan')
    delta = best_value - baseline_soft_score

    print(f"\nBest Trial #{best_trial.number}:")
    print(f"  {score_label}: {best_value:.6f}")
    print(f"  Δ from baseline: {delta:+.6f}")
    if objective_mode == "absolute":
        print(f"  Note: Direction (left/right) determined by final validation")
    else:
        print(f"  Note: Signed value — positive=right-leaning, negative=left-leaning")
    print(f"  Multipliers:")
    for neuron, mult in best_trial.params.items():
        print(f"    {neuron}: {mult:.4f}")

    print("\n" + "=" * 70)


def compute_soft_scores(
    wrapper,  # Llama3dot1Wrapper or Gemma3Wrapper
    questions_df: pd.DataFrame,
    positive_token_id: int,
    negative_token_id: int,
    language: str = "pt",
    activation_multipliers: Optional[Dict[str, float]] = None,
    label: str = "score",
) -> Tuple[float, float]:
    """
    Computes signed and absolute soft score for a questions dataset.

    Returns both the signed and absolute values for reporting.
    """
    mode = "intervened" if activation_multipliers else "baseline"
    print(f"Computing {label} soft score ({mode})...")

    pair_ids = sorted(questions_df['pair_id'].unique())
    total_signed_score = 0.0
    valid_pairs = 0

    for pair_id in pair_ids:
        pair_data = questions_df[questions_df['pair_id'] == pair_id]

        soft_p_plus = None
        soft_p_minus = None

        for _, row in pair_data.iterrows():
            statement = row['pergunta']
            tipo = row['tipo_pergunta']

            user_message = create_likert_prompt(statement, language)
            prompt = format_chat_prompt(
                wrapper.model.tokenizer, user_message, language)

            if wrapper.model.tokenizer is None:
                raise RuntimeError(
                    "Tokenizer is not initialized in the model wrapper")

            tokenized = wrapper.model.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=1024
            )
            input_ids = tokenized['input_ids']

            if not isinstance(input_ids, torch.Tensor):
                raise TypeError(
                    f"Expected input_ids to be torch.Tensor, got {type(input_ids).__name__}")

            soft_score, _ = wrapper.get_soft_stance_score(
                input_ids=input_ids,
                activation_multipliers=activation_multipliers,
                positive_token_id=positive_token_id,
                negative_token_id=negative_token_id,
                language=language
            )

            if tipo == 'P+':
                soft_p_plus = soft_score
            elif tipo == 'P-':
                soft_p_minus = soft_score

        if soft_p_plus is not None and soft_p_minus is not None:
            pair_diff = soft_p_plus - soft_p_minus
            total_signed_score += pair_diff
            valid_pairs += 1

    signed_soft = total_signed_score / valid_pairs if valid_pairs > 0 else 0.0
    abs_soft = abs(signed_soft)

    print(f"{label} Signed Soft Score: {signed_soft:.6f}")
    print(f"{label} |Soft Score| (Polarization): {abs_soft:.6f}")

    return signed_soft, abs_soft


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run optimization (TPE/CMA-ES) for neuron interventions.

    Uses soft metric (logit difference) optimization for continuous gradient signal.
    """
    # Get Hydra output directory early for logging
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    log_path = os.path.join(output_dir, "terminal_output.log")

    # Wrap entire execution in OutputLogger to capture all terminal output
    with OutputLogger(log_path):
        # Extract configuration
        opt_cfg = cfg.optimization
        ipi_eval_cfg = cfg.ipi_eval

        # W&B configuration
        wandb_cfg = cfg.get('wandb', {})
        feature_artifact_name = opt_cfg.get('feature_artifact_name', None)
        top_k = opt_cfg.get('top_k', opt_cfg.get('target_neuron_count', 80))
        n_trials = opt_cfg.get('n_trials', 3000)
        direction = opt_cfg.get('direction', 'maximize')
        seed = opt_cfg.get('seed', cfg.get('random_state', 42))

        if top_k is None or int(top_k) <= 0:
            raise ValueError(
                f"Invalid optimization.top_k={top_k!r}. Expected a positive integer."
            )
        if n_trials is None or int(n_trials) <= 0:
            raise ValueError(
                f"Invalid optimization.n_trials={n_trials!r}. Expected a positive integer."
            )
        if direction not in ('maximize', 'minimize'):
            raise ValueError(
                f"Invalid optimization.direction={direction!r}. Expected 'maximize' or 'minimize'."
            )

        top_k = int(top_k)
        n_trials = int(n_trials)
        seed = int(seed)

        # Initialize W&B with job_type="optimization"
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=wandb_cfg.get('project', 'activation-bias-classifier'),
            name=wandb_cfg.get('run_name', None),
            job_type="optimization",
            config=wandb_config
        )

        split_id = cfg.data.get('split_id', None)
        optimization_dataset_path = hydra.utils.to_absolute_path(
            cfg.data.get('optimization_dataset')
        )
        validation_dataset_path = hydra.utils.to_absolute_path(
            cfg.data.get('validation_dataset', ipi_eval_cfg.questions_csv)
        )

        # Determine target neurons: from artifact or config
        if feature_artifact_name:
            # Fetch SVM feature ranking artifact dynamically
            print(
                f"\nFetching feature ranking artifact: {feature_artifact_name}")
            artifact = wandb.use_artifact(feature_artifact_name)
            artifact_dir = artifact.download()

            # Load ranked feature payload and slice top_k deterministically.
            feature_ranking_path = os.path.join(artifact_dir, "feature_ranking.json")
            if not os.path.exists(feature_ranking_path):
                raise FileNotFoundError(
                    "feature_ranking.json not found in feature artifact. "
                    "Stage 5 requires ranked_features JSON payload."
                )
            with open(feature_ranking_path, "r", encoding="utf-8") as f:
                feature_ranking_payload = json.load(f)
            ranked_features = feature_ranking_payload.get("ranked_features", [])
            if not isinstance(ranked_features, list):
                raise ValueError("Invalid feature_ranking.json: ranked_features must be a list.")

            print(
                f"Loaded feature ranking with {len(ranked_features)} features")
            if top_k > len(ranked_features):
                raise ValueError(
                    f"optimization.top_k={top_k} exceeds available ranked_features={len(ranked_features)}."
                )

            selected_features = ranked_features[:top_k]
            target_neurons = [
                _ranked_feature_to_neuron_name(feature_entry)
                for feature_entry in selected_features
            ]
            print(
                f"Selected {len(target_neurons)} target neurons from ranked_features[:top_k]")
        else:
            # Use target neurons from YAML config
            target_neurons = list(opt_cfg.target_neurons)

        bounds = (opt_cfg.bounds[0], opt_cfg.bounds[1])
        study_name = opt_cfg.study_name
        storage = opt_cfg.get('storage', None)
        load_if_exists = opt_cfg.get('load_if_exists', True)
        n_startup_trials = opt_cfg.get('n_startup_trials', 10)

        fast_mode = opt_cfg.get('fast_mode', False)
        fast_n_pairs = opt_cfg.get('fast_n_pairs', 10)

        # Objective configuration
        objective_mode = opt_cfg.get(
            'objective_mode', 'signed')  # 'signed' or 'absolute'
        use_absolute = objective_mode == 'absolute'

        # Validate configuration
        assert objective_mode in ('signed', 'absolute'), \
            f"Invalid objective_mode '{objective_mode}'. Must be 'signed' or 'absolute'."
        assert direction in ('maximize', 'minimize'), \
            f"Invalid direction '{direction}'. Must be 'maximize' or 'minimize'."

        # Language setting
        language = ipi_eval_cfg.get('language', 'pt')

        print("=" * 70)
        print("NEURON INTERVENTION OPTIMIZATION (SOFT METRIC)")
        print("=" * 70)
        print(f"\nSeed: {seed}")
        print(f"\nOptimization Mode: Soft Metric (logit difference)")
        print(f"  - Objective mode: {objective_mode}")
        print(f"  - Direction: {direction}")
        if use_absolute:
            print(f"  - Returns |soft PI| → polarization magnitude")
        else:
            print(f"  - Returns signed soft PI → preserves polarization direction")
        print(f"  - This provides continuous gradient for the optimizer")
        print(f"\nTarget neurons ({len(target_neurons)}):")
        for n in target_neurons:
            print(f"  - {n}")
        print(f"\nMultiplier bounds: [{bounds[0]}, {bounds[1]}]")
        print(f"Number of trials: {n_trials}")
        print(f"Fast mode: {fast_mode}" +
              (f" ({fast_n_pairs} pairs)" if fast_mode else ""))
        print(f"Study storage: {storage or 'in-memory'}")
        print(f"Load if exists: {load_if_exists}")

        # Load optimization questions
        optim_questions_path = optimization_dataset_path
        print(
            f"\nLoading optimization questions from {optim_questions_path}...")
        optim_questions_df = pd.read_csv(optim_questions_path)

        # Load validation questions
        eval_questions_path = validation_dataset_path
        print(f"\nLoading validation questions from {eval_questions_path}...")
        eval_questions_df = pd.read_csv(eval_questions_path)

        # Apply fast mode sampling if enabled (only applies to optimization questions)
        if fast_mode:
            optim_questions_df = sample_questions(
                optim_questions_df,
                fast_n_pairs,
                random_state=cfg.get('random_state', 42)
            )
            print(
                f"Sampled {optim_questions_df['pair_id'].nunique()} optimization pairs for fast mode")

        print(f"Total optimization questions: {len(optim_questions_df)}")
        print(f"Total validation questions: {len(eval_questions_df)}")

        # Initialize model using factory
        print(f"\nInitializing model...")
        wrapper = get_model_wrapper(cfg)
        print(f"Loaded model: {wrapper.model.cfg.model_name}")

        # Get stance token IDs for soft metric
        positive_token_id, negative_token_id = wrapper.get_stance_token_ids(
            language)
        print(f"\nStance token IDs ({language}):")
        print(f"  Positive ('Concordo'/'Agree'): {positive_token_id}")
        print(f"  Negative ('Discordo'/'Disagree'): {negative_token_id}")

        # Verify tokens decode correctly
        if wrapper.model.tokenizer is None:
            raise RuntimeError(
                "Tokenizer is not initialized in the model wrapper")

        pos_decoded = wrapper.model.tokenizer.decode([positive_token_id])
        neg_decoded = wrapper.model.tokenizer.decode([negative_token_id])
        print(f"  Positive decodes to: '{pos_decoded}'")
        print(f"  Negative decodes to: '{neg_decoded}'")

        # Run baseline evaluation (discrete PI for final validation reference)
        baseline_scores, baseline_pi = run_baseline(
            wrapper=wrapper,
            questions_df=eval_questions_df,
            language=language,
            max_new_tokens=ipi_eval_cfg.get('max_new_tokens', 10),
            temperature=ipi_eval_cfg.get('temperature', 0.0)
        )

        # Compute baseline soft scores on optimization and validation datasets.
        baseline_opt_signed_soft, baseline_opt_abs_soft = compute_soft_scores(
            wrapper=wrapper,
            questions_df=optim_questions_df,
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id,
            language=language,
            activation_multipliers=None,
            label="Optimization baseline",
        )
        baseline_val_signed_soft, baseline_val_abs_soft = compute_soft_scores(
            wrapper=wrapper,
            questions_df=eval_questions_df,
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id,
            language=language,
            activation_multipliers=None,
            label="Validation baseline",
        )

        # Create sampler based on config
        sampler_type = opt_cfg.get('sampler', 'tpe').lower()

        if sampler_type == 'cmaes':
            print(
                f"Using CmaEsSampler (seed={seed})...")
            sampler = CmaEsSampler(
                seed=seed,
                n_startup_trials=n_startup_trials
            )
        else:
            # Default to TPE
            print(f"Using TPESampler (seed={seed})...")
            sampler = TPESampler(
                seed=seed,
                multivariate=True,
                n_startup_trials=n_startup_trials
            )

        # Resolve storage path if provided
        if storage:
            storage = hydra.utils.to_absolute_path(
                storage.replace('sqlite:///', ''))
            storage = f"sqlite:///{storage}"
            os.makedirs(os.path.dirname(
                storage.replace('sqlite:///', '')), exist_ok=True)

        # Create or load study - SINGLE OBJECTIVE
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            sampler=sampler,
            load_if_exists=load_if_exists
        )

        print(f"\nStarting soft metric optimization ({n_trials} trials)...")
        print(f"  objective_mode={objective_mode}, direction={direction}")
        print("-" * 70)

        # Run optimization with soft objective
        study.optimize(
            lambda trial: soft_objective(
                trial=trial,
                wrapper=wrapper,
                questions_df=optim_questions_df,
                target_neurons=target_neurons,
                bounds=bounds,
                positive_token_id=positive_token_id,
                negative_token_id=negative_token_id,
                language=language,
                use_absolute=use_absolute,
            ),
            n_trials=n_trials,
            show_progress_bar=True
        )

        # Print results
        baseline_ref = baseline_opt_abs_soft if use_absolute else baseline_opt_signed_soft
        print_best_soft_trial(
            study, baseline_ref, objective_mode=objective_mode, direction=direction)

        best_multipliers = study.best_trial.params
        optim_intervened_signed, optim_intervened_abs = compute_soft_scores(
            wrapper=wrapper,
            questions_df=optim_questions_df,
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id,
            language=language,
            activation_multipliers=best_multipliers,
            label="Optimization intervened",
        )
        val_intervened_signed, val_intervened_abs = compute_soft_scores(
            wrapper=wrapper,
            questions_df=eval_questions_df,
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id,
            language=language,
            activation_multipliers=best_multipliers,
            label="Validation intervened",
        )

        if use_absolute:
            soft_ipi_optimization_baseline = baseline_opt_abs_soft
            soft_ipi_optimization_intervened = optim_intervened_abs
            soft_ipi_validation_baseline = baseline_val_abs_soft
            soft_ipi_validation_intervened = val_intervened_abs
        else:
            soft_ipi_optimization_baseline = baseline_opt_signed_soft
            soft_ipi_optimization_intervened = optim_intervened_signed
            soft_ipi_validation_baseline = baseline_val_signed_soft
            soft_ipi_validation_intervened = val_intervened_signed

        delta_soft_ipi_optimization = (
            soft_ipi_optimization_intervened - soft_ipi_optimization_baseline
        )
        delta_soft_ipi_validation = (
            soft_ipi_validation_intervened - soft_ipi_validation_baseline
        )

        print("\nSoft metric summary:")
        print(f"  Optimization baseline:   {soft_ipi_optimization_baseline:.6f}")
        print(f"  Optimization intervened: {soft_ipi_optimization_intervened:.6f}")
        print(f"  Delta optimization:      {delta_soft_ipi_optimization:+.6f}")
        print(f"  Validation baseline:     {soft_ipi_validation_baseline:.6f}")
        print(f"  Validation intervened:   {soft_ipi_validation_intervened:.6f}")
        print(f"  Delta validation:        {delta_soft_ipi_validation:+.6f}")

        soft_metrics = {
            'soft_ipi_optimization_baseline': soft_ipi_optimization_baseline,
            'soft_ipi_optimization_intervened': soft_ipi_optimization_intervened,
            'delta_soft_ipi_optimization': delta_soft_ipi_optimization,
            'soft_ipi_validation_baseline': soft_ipi_validation_baseline,
            'soft_ipi_validation_intervened': soft_ipi_validation_intervened,
            'delta_soft_ipi_validation': delta_soft_ipi_validation,
        }

        # Save results
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"Expected config_dict to be dict, got {type(config_dict).__name__}")

        results_path = save_optimization_results(
            study=study,
            output_dir=output_dir,
            baseline_pi=baseline_pi,
            config=config_dict,
            baseline_soft_score=baseline_ref,
            use_soft_metric=True,
            soft_metrics=soft_metrics,
        )

        print(f"\nResults saved to: {results_path}")
        print(f"Terminal log saved to: {log_path}")

        # --- ARTIFACT: Log intervention multipliers as versioned model-weights artifact ---
        best_trial = study.best_trial
        artifacts_cfg = cfg.get('artifacts', {}) or {}
        multiplier_override = artifacts_cfg.get('multiplier_name', None)
        if multiplier_override:
            multipliers_artifact_name = str(multiplier_override)
        else:
            # Fallback name now includes top_k/n_trials/seed so concurrent
            # sweep jobs over (k, trials) don't collide on the same artifact.
            multipliers_artifact_name = (
                f"{wrapper.model.cfg.model_name}"
                f"_{objective_mode}_{direction}"
                f"_k{top_k}_trials{n_trials}_seed{seed}"
                f"_multipliers"
            )
        multipliers_artifact = wandb.Artifact(
            name=multipliers_artifact_name,
            type="model-weights",
            description="Optimized neuron activation multipliers for bias intervention",
            metadata={
                'baseline_pi': baseline_pi,
                'baseline_soft_score': baseline_ref,
                'best_trial_number': best_trial.number,
                'best_trial_value': best_trial.value,
                'n_trials': len(study.trials),
                'objective_mode': objective_mode,
                'direction': direction,
                'top_k': top_k,
                'split_id': split_id,
                'feature_artifact_name': feature_artifact_name,
                'optimization_dataset': optimization_dataset_path,
                'validation_dataset': validation_dataset_path,
                'seed': seed,
                'n_target_neurons': len(target_neurons),
                **soft_metrics,
            }
        )
        multipliers_artifact.add_file(results_path)
        wandb.log_artifact(multipliers_artifact)
        print(
            f"Intervention multipliers artifact logged: {multipliers_artifact_name}")

        # Log summary metrics to W&B
        wandb.summary.update({
            'baseline_pi': baseline_pi,
            'baseline_soft_score': baseline_ref,
            'best_soft_score': best_trial.value,
            'n_trials': len(study.trials),
            'best_trial': best_trial.number,
            'top_k': top_k,
            'direction': direction,
            'split_id': split_id,
            'feature_artifact_name': feature_artifact_name,
            'optimization_dataset': optimization_dataset_path,
            'validation_dataset': validation_dataset_path,
            'seed': seed,
            'n_target_neurons': len(target_neurons),
            **soft_metrics,
        })

        # Print best solution for easy copy-paste into config
        print("\n" + "=" * 70)
        print("BEST MULTIPLIERS (copy to config)")
        print("=" * 70)

        mode_label = f"{direction}s {'|soft PI|' if use_absolute else 'signed soft PI'}"
        print(f"\nBest soft metric solution ({mode_label}):")
        print("activation_multipliers: {")
        for neuron, mult in best_trial.params.items():
            print(f'  "{neuron}": {mult:.4f},')
        print("}")

        # Final validation suggestion
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("The soft metric optimization is complete.")
        if use_absolute:
            print("The optimizer used |soft PI| - direction may be left OR right.")
        else:
            print(f"The optimizer {direction}d the signed soft PI.")
        print("To validate the real-world PI, run likert_scale_test.py")
        print("with the best multipliers from above.")

        # Finish W&B run
        wandb.finish()

    return study


if __name__ == "__main__":
    main()
