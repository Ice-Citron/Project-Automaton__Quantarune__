"""Action head utilities for the flow matching model.

The main action prediction is handled directly in flow_matching.py via
output_proj. This module provides the model builder that dispatches
between FlowMatchingModel and Pi05Wrapper based on config.
"""

import torch.nn as nn

from quantarune_training.models.flow_matching import FlowMatchingModel


def build_model(cfg) -> nn.Module:
    """Build model from Hydra config.

    Args:
        cfg: OmegaConf model config with 'name' field.

    Returns:
        nn.Module: The model.
    """
    if cfg.name == "flow_matching":
        return FlowMatchingModel(cfg)
    elif cfg.name == "pi05":
        from quantarune_training.models.pi05_wrapper import Pi05Wrapper
        return Pi05Wrapper(cfg)
    else:
        raise ValueError(f"Unknown model name: {cfg.name}")
