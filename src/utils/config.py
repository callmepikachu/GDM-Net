import yaml
import os
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.create(config_dict)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        OmegaConf.save(config, f)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """Merge two configurations."""
    return OmegaConf.merge(base_config, override_config)


def validate_config(config: DictConfig) -> None:
    """Validate configuration parameters."""
    required_keys = [
        'model.bert_model_name',
        'model.hidden_size', 
        'data.train_path',
        'data.val_path',
        'training.max_epochs',
        'training.batch_size'
    ]
    
    for key in required_keys:
        if not OmegaConf.select(config, key):
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate paths
    for path_key in ['data.train_path', 'data.val_path']:
        path = OmegaConf.select(config, path_key)
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
