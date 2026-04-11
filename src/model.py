# src/model.py
import torch
from aurora import Aurora, AuroraSmallPretrained
from aurora.normalisation import locations, scales

def load_model(config, norm_stats=None):
    """
    Args:
        config (dict): Configuration dictionary containing model parameters.
        norm_stats (dict, optional): Normalization statistics for the new variables. Required if using 'ttr' and/or 'tcwv'.
    """
    print(f"Initializing Aurora model. Type: {config['model_type']}")

    extended_surf_vars = tuple(config['surface_variables'])

    model_class = Aurora if config['model_type'] == 'huge' else AuroraSmallPretrained

    model = model_class(
        surf_vars = extended_surf_vars,
        use_lora = config['use_lora'],
        lora_mode = config.get('lora_mode', 'single')
    )

    print("Loading pre-trained weights")
    model.load_checkpoint(strict=False)

    if norm_stats:
        print("Injecting normalization statistics for new variables")
        for var_name, stats in norm_stats.items():
            locations[var_name] = stats['mean']
            scales[var_name] = stats['std']
            print(f"   - {var_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    if config.get('gradient_checkpointing', False):
        print("Enabling gradient checkpointing")
        model.configure_activation_checkpointing()
    
    return model