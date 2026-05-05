import os
import sys
import pandas as pd
import hydra
from omegaconf import DictConfig
import wandb

# Add src to path for any shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Get input path from config
    input_path = hydra.utils.to_absolute_path(cfg.data.feature_selection_statements)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load and validate the dataset
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples with columns: {list(df.columns)}")
    
    # Derive artifact name from filename
    dataset_name = os.path.basename(input_path).replace('.csv', '')
    artifact_name = f"dataset-{dataset_name}"
    
    # W&B configuration
    wandb_cfg = cfg.get('wandb', {})
    
    # Initialize W&B
    wandb.init(
        project=wandb_cfg.get('project', 'activation-stance-classifier'),
        name=wandb_cfg.get('run_name', f"log-{dataset_name}"),
        job_type="dataset_upload",
        config={
            'feature_selection_statements': input_path,
            'n_samples': len(df),
            'columns': list(df.columns),
        }
    )
    
    # Create and log artifact
    dataset_artifact = wandb.Artifact(
        name=artifact_name,
        type="dataset",
        description=f"Input dataset: {dataset_name}",
        metadata={
            'source_path': input_path,
            'n_samples': len(df),
            'columns': list(df.columns),
        }
    )
    dataset_artifact.add_file(input_path)
    wandb.log_artifact(dataset_artifact)
    
    print(f"Dataset artifact logged: {artifact_name}")
    
    # Log summary statistics
    label_counts = {}
    if 'pol_label_human' in df.columns:
        label_counts = df['pol_label_human'].value_counts().to_dict()
        wandb.summary.update({'label_distribution': label_counts})
    
    if 'topic_label_human' in df.columns:
        topic_counts = df['topic_label_human'].value_counts().to_dict()
        wandb.summary.update({'topic_distribution': topic_counts})
    
    wandb.summary.update({
        'n_samples': len(df),
        'n_columns': len(df.columns),
    })
    
    print(f"Summary: {len(df)} samples")
    if label_counts:
        print(f"Label distribution: {label_counts}")
    
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
