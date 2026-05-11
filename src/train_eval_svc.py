import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import os
import json
import re
# import loguru as logging (we'll use it later)
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from mrmr import mrmr_classif
import wandb
from omegaconf import DictConfig, OmegaConf


def _extract_layer_and_feature(feature_name: str) -> tuple[int | None, int | None]:
    """
    Parse feature strings like `layer_15-neuron_1215` into integers.
    Returns (None, None) when pattern is unavailable.
    """
    match = re.match(r"layer_(\d+)-neuron_(\d+)$", str(feature_name))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def plot_svm_decision_boundary(X, y, clf, le, output_path, title="SVC Decision Boundary (PCA reduced)"):
    """
    Plots the decision boundary of a 2D SVC.
    """
    plt.figure(figsize=(10, 8))

    # Create the display
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=cm.get_cmap('coolwarm'),
        plot_method="pcolormesh",
        shading="auto",
        alpha=0.6
    )

    # Scatter plot of the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y,
                          cmap=cm.get_cmap('coolwarm'), edgecolors="k", s=50)

    # Highlight support vectors
    support_vectors = clf.support_vectors_
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=150,
                linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    handles = []
    cmap = cm.get_cmap('coolwarm')
    unique_classes = np.unique(y)
    for cls in unique_classes:
        color = cmap(cls / (len(unique_classes) - 1)
                     if len(unique_classes) > 1 else 0.5)
        handles.append(mpatches.Patch(color=color, label=le.classes_[cls]))

    # Add support vectors legend entry
    handles.append(mlines.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='none', markeredgecolor='k',
                                 markersize=10, label='Support Vectors'))

    plt.legend(handles=handles)

    # Save
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path)
    plt.close()


def mean_classification_report(reports):
    """
    Averages multiple classification reports.
    """
    # Convert reports to dictionaries and collect metrics
    all_metrics = {}

    for report in reports:
        # Parse the classification report string
        lines = report.strip().split('\n')

        for line in lines:
            parts = line.split()
            if len(parts) >= 5 and parts[0] not in ['accuracy', 'macro', 'weighted']:
                # This is a class-specific line
                class_name = parts[0]
                precision = float(parts[1])
                recall = float(parts[2])
                f1 = float(parts[3])
                support = int(parts[4])

                if class_name not in all_metrics:
                    all_metrics[class_name] = {
                        'precision': [], 'recall': [], 'f1-score': [], 'support': []}

                all_metrics[class_name]['precision'].append(precision)
                all_metrics[class_name]['recall'].append(recall)
                all_metrics[class_name]['f1-score'].append(f1)
                all_metrics[class_name]['support'].append(support)

    # Calculate means
    mean_report = {}
    for class_name, metrics in all_metrics.items():
        mean_report[class_name] = {
            'precision_mean': np.mean(metrics['precision']),
            'precision_std': np.std(metrics['precision']),
            'recall_mean': np.mean(metrics['recall']),
            'recall_std': np.std(metrics['recall']),
            'f1-score_mean': np.mean(metrics['f1-score']),
            'f1-score_std': np.std(metrics['f1-score']),
            'support_mean': int(np.mean(metrics['support']))
        }
    mean_report_df = pd.DataFrame(mean_report).T
    return mean_report_df


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    run_dir = HydraConfig.get().runtime.output_dir
    output_image_path = os.path.join(run_dir, cfg.training.image_file)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # W&B configuration
    wandb_cfg = cfg.get('wandb', {})
    data_cfg = cfg.get('data', {})
    activations_artifact_name = data_cfg.get('activations_artifact_name', None)

    # Initialize W&B early (needed if using artifacts)
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=wandb_cfg.get('project', 'activation-bias-classifier'),
        name=wandb_cfg.get('run_name', None),
        job_type="svm_training",
        config=wandb_config,
    )

    # 1. Load Data - from artifact or local file
    if activations_artifact_name:
        # Fetch activations from W&B artifact
        logger.info(
            f"Fetching activations artifact: {activations_artifact_name}")
        artifact = wandb.use_artifact(activations_artifact_name)
        artifact_dir = artifact.download()

        # Find the parquet file in the artifact
        import glob
        parquet_files = glob.glob(os.path.join(artifact_dir, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet file found in artifact: {artifact_dir}")
        input_path = parquet_files[0]
        logger.info(f"Using activations from artifact: {input_path}")
    else:
        # Use local file path from config
        input_path = hydra.utils.to_absolute_path(cfg.data.activations_file)

    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        print(
            f"Error: {input_path} not found. Please run extract_activations.py first or specify the path.")
        return

    # Check if it's CSV or Parquet based on extension or try both
    try:
        if input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    logger.info(f"Data shape: {df.shape}")

    # 2. Preprocessing
    # Assume last column is 'class' and others are features
    # Or find 'class' column specifically
    if 'class' not in df.columns:
        logger.error("Error: 'class' column not found in dataframe.")
        return

    X = df.drop(columns=['class']).values
    y_labels = df['class'].values

    # Encode labels
    le = LabelEncoder()
    y = np.asarray(le.fit_transform(np.asarray(y_labels)))
    logger.info(f"Classes: {le.classes_}")

    # Note: Scaling is now done inside each fold to prevent data leakage

    # 3. Split into Train and Holdout Test Set FIRST (before any CV)
    # This ensures we have a completely unseen test set for final evaluation
    test_size = cfg.training.get('test_size', 0.2)
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=cfg.training.random_state
    )
    logger.info(f"\n--- Data Split ---")
    logger.info(f"Total samples: {len(y)}")
    logger.info(
        f"Training set: {len(y_train_full)} samples ({100*(1-test_size):.0f}%)")
    logger.info(
        f"Holdout test set: {len(y_holdout)} samples ({100*test_size:.0f}%) - UNTOUCHED until final evaluation")

    # 4. K-Fold Cross-Validation on Training Set ONLY
    # Feature selection is performed INSIDE each fold to prevent data leakage

    use_mrmr = cfg.feature_selection.enabled
    n_features_to_select = cfg.feature_selection.n_features
    # Default to 500 features for SelectKBest
    selectkbest_k = cfg.feature_selection.get('selectkbest_k', 500)
    feature_names = df.drop(columns=['class']).columns.tolist()

    if use_mrmr:
        logger.info(
            f"\n--- Training SVC with 3-Fold CV + SelectKBest (k={selectkbest_k}) + MRMR (n={n_features_to_select}) ---")
    else:
        logger.info(
            "\n--- Training SVC with 3-Fold Cross-Validation on Training Set ---")

    # Initialize K-Fold (on training set only, NOT on holdout)
    k_folds = 3
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True,
                          random_state=cfg.training.random_state)

    # Store results for each fold
    fold_accuracies = []
    fold_reports = []
    fold_selected_features = []  # Track selected features per fold

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        logger.info(f"\n=== Fold {fold}/{k_folds} ===")

        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

        # Step 1: Fit scaler on training data only (prevent data leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Feature Selection Pipeline (applied on training set only to prevent data leakage)
        if use_mrmr:
            # Step 2: SelectKBest to prune search space before MRMR
            k_best = min(selectkbest_k, X_train_scaled.shape[1])
            SCORE_FUNC = cfg.feature_selection.prefilter
            if SCORE_FUNC not in ['f_classif', 'mutual_info_classif']:
                raise ValueError(
                    f"Unsupported score function: {SCORE_FUNC}")
            SCORE_FUNC = f_classif if SCORE_FUNC == 'f_classif' else mutual_info_classif
            selector = SelectKBest(score_func=SCORE_FUNC, k=k_best)
            X_train_kbest = selector.fit_transform(X_train_scaled, y_train)
            X_val_kbest = selector.transform(X_val_scaled)

            # Get feature names after SelectKBest
            kbest_mask = selector.get_support()
            kbest_feature_names = [f for f, selected in zip(
                feature_names, kbest_mask) if selected]
            logger.info(
                f"SelectKBest reduced features from {len(feature_names)} to {len(kbest_feature_names)}")

            # Step 3: MRMR on the reduced feature set
            X_train_df = pd.DataFrame(
                X_train_kbest, columns=kbest_feature_names)
            y_train_series = pd.Series(y_train, name='class')

            # Perform MRMR feature selection on training data only
            selected_features = mrmr_classif(
                X=X_train_df,
                y=y_train_series,
                K=min(n_features_to_select, len(kbest_feature_names)),
            )
            fold_selected_features.append(selected_features)

            logger.info(
                f"MRMR selected {len(selected_features)} features from {len(kbest_feature_names)} candidates")
            logger.info(f"Top 5 features: {selected_features[:5]}")

            # Step 4: Get indices of selected features within kbest-reduced data
            selected_indices = [kbest_feature_names.index(
                str(f)) for f in selected_features]

            # Apply feature selection to both train and val
            X_train_selected = X_train_kbest[:, selected_indices]
            X_val_selected = X_val_kbest[:, selected_indices]
        else:
            X_train_selected = X_train_scaled
            X_val_selected = X_val_scaled

        # Train SVC
        svc_full = SVC(kernel=cfg.training.kernel,
                       random_state=cfg.training.random_state,
                       class_weight=cfg.training.class_weight)
        svc_full.fit(X_train_selected, y_train)

        # Evaluate on validation fold
        y_pred = svc_full.predict(X_val_selected)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        fold_accuracies.append(balanced_acc)

        logger.info(f"Fold {fold} Balanced Accuracy: {balanced_acc:.4f}")
        report = classification_report(
            y_val, y_pred, target_names=le.classes_)
        fold_reports.append(report)
        logger.info(f"Classification Report:\n{report}")

    # Log aggregate results
    logger.info("\n--- Cross-Validation Summary ---")
    logger.info(
        f"Fold Balanced Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    logger.info(
        f"Mean Balanced Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")

    mean_report_df = mean_classification_report(fold_reports)
    logger.info("Mean Classification Report:")
    logger.info(f"\n{mean_report_df}")

    # Analyze feature selection stability across folds
    if use_mrmr and fold_selected_features:
        logger.info("\n--- Feature Selection Stability Analysis ---")
        # Find features selected in all folds
        common_features = set(fold_selected_features[0])
        for fold_features in fold_selected_features[1:]:
            common_features &= set(fold_features)
        logger.info(
            f"Features selected in ALL folds ({len(common_features)}): {sorted(common_features)[:20]}")

        # Count feature selection frequency
        from collections import Counter
        all_selected = [
            f for fold_features in fold_selected_features for f in fold_features]
        feature_counts = Counter(all_selected)
        # Build a full ranking over all available features so downstream stages can
        # request arbitrary top_k slices from a stable ranked artifact.
        full_feature_rows = []
        for feature in feature_names:
            selection_count = int(feature_counts.get(feature, 0))
            selection_frequency = selection_count / k_folds
            full_feature_rows.append({
                'feature': feature,
                'selection_count': selection_count,
                'selection_frequency': selection_frequency,
            })
        full_feature_ranking_df = pd.DataFrame(full_feature_rows)
        full_feature_ranking_df = full_feature_ranking_df.sort_values(
            by=['selection_count', 'selection_frequency', 'feature'],
            ascending=[False, False, True]
        ).reset_index(drop=True)
        full_feature_ranking_df['rank'] = range(1, len(full_feature_ranking_df) + 1)
        full_feature_ranking_df = full_feature_ranking_df[
            ['rank', 'feature', 'selection_count', 'selection_frequency']
        ]

        ranking_top_n = int(cfg.feature_selection.get('ranking_top_n', 256))
        if ranking_top_n <= 0:
            raise ValueError(
                f"feature_selection.ranking_top_n must be > 0, got {ranking_top_n}"
            )
        if ranking_top_n > len(full_feature_ranking_df):
            logger.warning(
                "Requested ranking_top_n=%s but only %s features available; using all features.",
                ranking_top_n, len(full_feature_ranking_df)
            )

        feature_ranking_df = full_feature_ranking_df.head(ranking_top_n).copy()

        # Feature ranking
        logger.info(
            f"\n--- Feature Ranking (by selection frequency across folds) ---")
        logger.info(f"\n{feature_ranking_df.to_string()}")

        # Use most frequently selected features for final model
        final_selected_features = [
            f for f, _ in feature_counts.most_common(n_features_to_select)]
        final_selected_indices = [feature_names.index(
            f) for f in final_selected_features]

        # For final model, fit scaler on TRAINING data only, then transform both train and holdout
        final_scaler = StandardScaler()
        X_train_scaled = final_scaler.fit_transform(X_train_full)
        X_holdout_scaled = final_scaler.transform(X_holdout)

        X_train_scaled_selected = X_train_scaled[:, final_selected_indices]
        X_holdout_scaled_selected = X_holdout_scaled[:, final_selected_indices]
    else:
        final_scaler = StandardScaler()
        X_train_scaled = final_scaler.fit_transform(X_train_full)
        X_holdout_scaled = final_scaler.transform(X_holdout)

        X_train_scaled_selected = X_train_scaled
        X_holdout_scaled_selected = X_holdout_scaled
        final_selected_features = feature_names

    # 5. Train Final Model on Full Training Set and Evaluate on Holdout Test Set
    logger.info("\n--- Training Final Model on Full Training Set ---")
    svc_final = SVC(kernel=cfg.training.kernel,
                    random_state=cfg.training.random_state,
                    class_weight=cfg.training.class_weight)
    svc_final.fit(X_train_scaled_selected, y_train_full)

    logger.info(f"Final Model Training Complete")
    logger.info(
        f"Number of support vectors: {len(svc_final.support_vectors_)}")

    # Evaluate on HOLDOUT TEST SET - this is the true generalization performance
    logger.info("\n--- Final Evaluation on Holdout Test Set ---")
    y_pred_holdout = svc_final.predict(X_holdout_scaled_selected)
    holdout_accuracy = balanced_accuracy_score(y_holdout, y_pred_holdout)
    logger.info(
        f"🎯 HOLDOUT TEST SET Balanced Accuracy: {holdout_accuracy:.4f}")

    holdout_report = classification_report(
        y_holdout, y_pred_holdout, target_names=le.classes_)
    logger.info(f"Holdout Test Set Classification Report:\n{holdout_report}")

    # Summary comparison
    logger.info("\n--- Performance Summary ---")
    logger.info(
        f"Cross-Validation Mean Balanced Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    logger.info(
        f"Holdout Test Set Balanced Accuracy:      {holdout_accuracy:.4f}")
    logger.info("(If these differ significantly, the model may be overfitting)")

    # 6. Visualization (2D Projection)
    logger.info("\n--- Generating Visualization (PCA -> 2D SVC) ---")
    # Reduce to 2D for visualization (using training data with selected features)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled_selected)

    logger.info(
        f"Explained variance ratio of first 2 components: {pca.explained_variance_ratio_}")

    # Train a new 2D SVC for visualization purposes (on training data)
    svc_2d = SVC(kernel="rbf",
                 random_state=cfg.training.random_state,
                 class_weight='balanced')
    svc_2d.fit(X_train_pca, y_train_full)

    plot_svm_decision_boundary(X_train_pca, y_train_full, svc_2d, le, output_image_path,
                               title=f"SVC Decision Boundary (PCA 2D Projection - Training Data)\nClasses: {le.classes_}")
    logger.info(f"Results saved to {run_dir}")
    logger.info(f"Visualization: {output_image_path}")

    # Update W&B config with additional runtime info
    wandb.config.update({
        'data_file': input_path,
        'full_dataset_length': len(y),
        'training_set_length': len(y_train_full),
        'holdout_set_length': len(y_holdout),
        'k_folds': k_folds,
    })

    # 1. Static scalar attributes go to summary (no more 1-point charts)
    wandb.summary.update({
        'cv_mean_balanced_accuracy': np.mean(fold_accuracies),
        'cv_std_balanced_accuracy': np.std(fold_accuracies),
        'holdout_test_balanced_accuracy': holdout_accuracy,
        'n_support_vectors': len(svc_final.support_vectors_),
        'pca_variance_explained_pc1': pca.explained_variance_ratio_[0],
        'pca_variance_explained_pc2': pca.explained_variance_ratio_[1],
    })

    # 2. Balanced accuracy table (folds + holdout)
    accuracy_data = [[f"Fold {i+1}", acc]
                     for i, acc in enumerate(fold_accuracies)]
    accuracy_data.append(["Holdout", holdout_accuracy])
    accuracy_table = wandb.Table(
        data=accuracy_data,
        columns=["split", "balanced_accuracy"]
    )

    # 3. Classification reports as interactive tables
    cv_report_table = wandb.Table(
        dataframe=mean_report_df.reset_index().rename(columns={'index': 'class'}))

    holdout_report_dict = classification_report(
        y_holdout, y_pred_holdout, target_names=le.classes_, output_dict=True)
    holdout_df = pd.DataFrame(holdout_report_dict).T.drop(['accuracy'])
    holdout_table = wandb.Table(
        dataframe=holdout_df.reset_index().rename(columns={'index': 'class'}))

    # 4. Feature selection info
    if use_mrmr and fold_selected_features:
        wandb.summary.update({
            'n_common_features_all_folds': len(common_features),
            'n_final_selected_features': len(final_selected_features),
        })

        feature_ranking_table = wandb.Table(
            dataframe=feature_ranking_df
        )

        # Log all tables and images in one call
        wandb.log({
            'balanced_accuracy_table': accuracy_table,
            'feature_ranking': feature_ranking_table,
            'cv_mean_classification_report': cv_report_table,
            'holdout_classification_report': holdout_table,
            'decision_boundary': wandb.Image(output_image_path),
            'holdout_confusion_matrix': wandb.plot.confusion_matrix(
                y_true=y_holdout.tolist(),
                preds=y_pred_holdout.tolist(),
                class_names=le.classes_.tolist()
            ),
        })

        # --- ARTIFACT: Log feature ranking as versioned dataset artifact ---
        feature_ranking_csv_path = os.path.join(run_dir, "feature_ranking.csv")
        feature_ranking_df.to_csv(feature_ranking_csv_path, index=False)
        logger.info(
            f"Feature ranking CSV saved to: {feature_ranking_csv_path}")

        model_name = cfg.model.name.split('/')[-1]
        split_id = cfg.data.get('split_id', None)
        feature_selection_dataset = cfg.data.get('feature_selection_dataset', None)
        if use_mrmr:
            prefilter_name = str(cfg.feature_selection.get('prefilter', 'f_classif'))
            mrmr_name = str(cfg.feature_selection.get('method_mrmr', 'MIQ'))
            feature_selection_method = f"{prefilter_name}+mrmr:{mrmr_name}"
        else:
            feature_selection_method = "no_feature_selection"
        fs_seed = int(cfg.feature_selection.get(
            'seed',
            cfg.training.get('random_state', cfg.get('random_state', 42))
        ))
        ranking_top_n_effective = len(feature_ranking_df)

        ranked_features = []
        for row in feature_ranking_df.to_dict(orient='records'):
            layer_idx, feature_idx = _extract_layer_and_feature(row['feature'])
            ranked_features.append({
                'rank': int(row['rank']),
                'layer': layer_idx,
                'feature': feature_idx,
                'feature_name': row['feature'],
                'score': None,
                'selection_frequency': float(row['selection_frequency']),
                'selection_count': int(row['selection_count']),
            })

        ranking_payload = {
            'model_name': model_name,
            'split_id': split_id,
            'feature_selection_dataset': feature_selection_dataset,
            'method': feature_selection_method,
            'ranking_top_n': ranking_top_n_effective,
            'seed': fs_seed,
            'ranked_features': ranked_features,
        }
        feature_ranking_json_path = os.path.join(run_dir, "feature_ranking.json")
        with open(feature_ranking_json_path, 'w', encoding='utf-8') as f:
            json.dump(ranking_payload, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Feature ranking JSON saved to: {feature_ranking_json_path}")

        # Create and log artifact. Orchestrator-supplied name wins over the
        # default derived from the input activations artifact, so sweep jobs
        # can advertise deterministic feature-ranking identities.
        artifacts_cfg = cfg.get('artifacts', {}) or {}
        feature_ranking_override = artifacts_cfg.get('feature_ranking_name', None)
        if feature_ranking_override:
            feature_artifact_name_out = str(feature_ranking_override)
        elif activations_artifact_name:
            feature_artifact_name_out = (
                f"svm-feature-ranking-"
                f"{activations_artifact_name.split(':')[0].split('/')[-1]}"
            )
        else:
            feature_artifact_name_out = (
                f"svm-feature-ranking-{cfg.model.name.split('/')[-1]}"
            )

        feature_artifact = wandb.Artifact(
            name=feature_artifact_name_out,
            type="dataset",
            description="SVM feature ranking from MRMR selection across CV folds",
            metadata={
                'n_features': len(feature_ranking_df),
                'k_folds': k_folds,
                'n_features_to_select': n_features_to_select,
                'selectkbest_k': selectkbest_k,
                'holdout_accuracy': holdout_accuracy,
                'model_name': model_name,
                'split_id': split_id,
                'feature_selection_dataset': feature_selection_dataset,
                'method': feature_selection_method,
                'ranking_top_n': ranking_top_n_effective,
                'seed': fs_seed,
            }
        )
        feature_artifact.add_file(feature_ranking_csv_path)
        feature_artifact.add_file(feature_ranking_json_path)
        wandb.log_artifact(feature_artifact)
        logger.info(f"Feature ranking artifact logged: {feature_artifact.name}")
    else:
        # Log tables and images without layer selection (when MRMR is disabled)
        wandb.log({
            'balanced_accuracy_table': accuracy_table,
            'cv_mean_classification_report': cv_report_table,
            'holdout_classification_report': holdout_table,
            'decision_boundary': wandb.Image(output_image_path),
            'holdout_confusion_matrix': wandb.plot.confusion_matrix(
                y_true=y_holdout.tolist(),
                preds=y_pred_holdout.tolist(),
                class_names=le.classes_.tolist()
            ),
        })

    wandb.finish()


if __name__ == "__main__":
    main()
