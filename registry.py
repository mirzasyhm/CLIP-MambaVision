""" 
Scripts to register and load models.
Adapted from:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_registry.py
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_factory.py
"""

import torch
import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Callable

import sys
import re
import fnmatch

__all__ = [
    'list_models', 'is_model', 'model_entrypoint',
    'list_modules', 'is_model_in_modules',
    'is_model_default_key', 'has_model_default_key',
    'get_model_default_value', 'is_model_pretrained'
]

# Initialize the registry
_model_entrypoints = {}  # mapping of model names to entrypoint functions
_module_to_models = defaultdict(set)  # mapping of module names to model names
_model_to_module = {}  # mapping of model names to their modules
_model_has_pretrained = set()  # models with pretrained weights
_model_default_cfgs = {}  # default configurations for models


def register_pip_model(name: str, fn: Callable):
    """
    Register a model factory function with a unique name.

    Args:
        name (str): The unique name of the model.
        fn (Callable): A function that returns an instance of the model.
    """
    if name in _model_entrypoints:
        raise ValueError(f"Model '{name}' is already registered.")
    
    _model_entrypoints[name] = fn
    # Optionally, extract module information if needed
    module_name = fn.__module__.split('.')[-1]
    _model_to_module[name] = module_name
    _module_to_models[module_name].add(name)


def list_models(filter: str = '') -> list:
    """
    List all registered models, optionally filtered by a wildcard pattern.

    Args:
        filter (str): Wildcard pattern to filter model names.

    Returns:
        list: A sorted list of matching model names.
    """
    if not filter:
        return sorted(_model_entrypoints.keys())
    else:
        return sorted([name for name in _model_entrypoints if fnmatch.fnmatch(name, filter)])


def is_model(model_name: str) -> bool:
    """
    Check if a model is registered.

    Args:
        model_name (str): The name of the model.

    Returns:
        bool: True if the model is registered, False otherwise.
    """
    return model_name in _model_entrypoints


def model_entrypoint(model_name: str) -> Callable:
    """
    Retrieve the factory function for a given model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        Callable: The factory function to create the model instance.
    """
    if model_name not in _model_entrypoints:
        raise ValueError(f"Model '{model_name}' is not registered.")
    return _model_entrypoints[model_name]


def list_modules() -> list:
    """
    List all module names that contain registered models.

    Returns:
        list: A sorted list of module names.
    """
    return sorted(_module_to_models.keys())


def is_model_in_modules(model_name: str, module_names: list) -> bool:
    """
    Check if a model exists within a subset of modules.

    Args:
        model_name (str): Name of the model to check.
        module_names (list): List of module names to search in.

    Returns:
        bool: True if the model exists in any of the specified modules, False otherwise.
    """
    return any(model_name in _module_to_models[module] for module in module_names)


def has_model_default_key(model_name: str, cfg_key: str) -> bool:
    """
    Check if a model's default configuration has a specific key.

    Args:
        model_name (str): Name of the model.
        cfg_key (str): Configuration key to check.

    Returns:
        bool: True if the key exists, False otherwise.
    """
    return model_name in _model_default_cfgs and cfg_key in _model_default_cfgs[model_name]


def is_model_default_key(model_name: str, cfg_key: str) -> bool:
    """
    Check if a model's default configuration has a truthy value for a specific key.

    Args:
        model_name (str): Name of the model.
        cfg_key (str): Configuration key to check.

    Returns:
        bool: True if the key exists and is truthy, False otherwise.
    """
    return has_model_default_key(model_name, cfg_key) and bool(_model_default_cfgs[model_name][cfg_key])


def get_model_default_value(model_name: str, cfg_key: str):
    """
    Get a specific model's default configuration value by key.

    Args:
        model_name (str): Name of the model.
        cfg_key (str): Configuration key to retrieve.

    Returns:
        Any: The value of the configuration key, or None if it doesn't exist.
    """
    return _model_default_cfgs.get(model_name, {}).get(cfg_key, None)


def is_model_pretrained(model_name: str) -> bool:
    """
    Check if a model has pretrained weights.

    Args:
        model_name (str): Name of the model.

    Returns:
        bool: True if pretrained weights are available, False otherwise.
    """
    return model_name in _model_has_pretrained


def load_state_dict(checkpoint_path: str, use_ema: bool = False) -> dict:
    """
    Load a state dictionary from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        use_ema (bool): Whether to use EMA weights if available.

    Returns:
        dict: The loaded state dictionary.
    """
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        print(f"Loaded {state_dict_key} from checkpoint '{checkpoint_path}'")
        return state_dict
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, use_ema: bool = False, strict: bool = True):
    """
    Load a checkpoint into a model.

    Args:
        model (torch.nn.Module): The model to load the checkpoint into.
        checkpoint_path (str): Path to the checkpoint file.
        use_ema (bool): Whether to use EMA weights if available.
        strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by model's state_dict().
    """
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


def create_model(model_name: str, pretrained: bool = False, checkpoint_path: str = '', **kwargs) -> torch.nn.Module:
    """
    Create a model instance based on the registered entrypoint.

    Args:
        model_name (str): Name of the model to create.
        pretrained (bool): Whether to load pretrained weights.
        checkpoint_path (str): Path to a checkpoint file to load.
        **kwargs: Additional keyword arguments for the model factory function.

    Returns:
        torch.nn.Module: The instantiated model.
    """
    create_fn = model_entrypoint(model_name)
    model = create_fn(pretrained=pretrained, **kwargs)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model
