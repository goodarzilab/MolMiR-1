"""Helper functions for model configuration."""
from typing import Dict, Optional
import logging
from transformers import AutoConfig
from huggingface_hub.utils import RepositoryNotFoundError

def get_model_type(model_name: str) -> str:
    """
    Determine model type from name with more specific matching.
    Returns the most specific matching type.
    """
    model_name = model_name.lower()
    
    # More specific matches first
    if "chemberta-77m-mlm" in model_name:
        return "chemberta_mlm"
    elif "chemberta-77m-mtr" in model_name:
        return "chemberta_mlm"  # Uses same config as MLM
    elif "chemberta" in model_name:
        return "chemberta"
    elif "molformer-xl" in model_name:
        return "molformer"
    elif "molformer" in model_name:
        return "molformer"
    elif "pubmedbert" in model_name:
        return "pubchem"
    elif "biobert" in model_name:
        return "pubchem"
    elif "scibert" in model_name:
        return "pubchem"
        
    logging.warning(f"No specific configuration found for model {model_name}, using default")
    return "default"

def validate_model_access(model_name: str) -> bool:
    """Validate if model can be accessed."""
    try:
        # Add trust_remote_code for MolFormer
        trust_remote_code = "molformer" in model_name.lower()
        AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        return True
    except (OSError, RepositoryNotFoundError) as e:
        if "private repository" in str(e):
            logging.error(f"Model {model_name} is private. Please login using 'huggingface-cli login'")
        else:
            logging.error(f"Could not access model {model_name}: {str(e)}")
        return False

def get_model_config(model_name: str, default_config: dict, model_configs: dict) -> Optional[dict]:
    """
    Get model-specific configuration.
    
    Args:
        model_name: Name of the model
        default_config: Default configuration
        model_configs: Model-specific configurations
        
    Returns:
        Combined configuration dictionary
    """
    # Get model type
    model_type = get_model_type(model_name)
    logging.info(f"Model {model_name} mapped to type {model_type}")
    
    # Start with default config
    config = default_config.copy()
    
    # Add trust_remote_code for MolFormer
    if model_type == "molformer":
        config['trust_remote_code'] = True
    
    # Update with model-specific config if available
    if model_type in model_configs:
        type_config = model_configs[model_type]
        logging.info(f"Using {model_type} specific config: {type_config}")
        config.update(type_config)
    
    # Log final configuration
    logging.info(f"Final model configuration: {config}")
    
    return config
