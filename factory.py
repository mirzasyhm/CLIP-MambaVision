# factory.py

from .model import CLIP
from .mamba_vision import (
    MambaVision,
    mamba_vision_T,
    mamba_vision_S,
    mamba_vision_B,
    mamba_vision_L,
    mamba_vision_L2
)
from collections import defaultdict

_model_entrypoints = defaultdict(set)

def clip_with_encoder(image_encoder_type='Original', **kwargs):
    embed_dim = 512  # Adjust based on your requirements
    model = CLIP(
        embed_dim=embed_dim,
        image_encoder_type=image_encoder_type,
        **kwargs
    )
    return model

def register_models():
    # Register CLIP with Original Image Encoder
    _model_entrypoints['CLIP_Original'] = lambda pretrained=False, **kwargs: clip_with_encoder(image_encoder_type='Original', **kwargs)
    
    # Register CLIP with MambaVision Image Encoder
    _model_entrypoints['CLIP_MambaVision_T'] = lambda pretrained=False, **kwargs: clip_with_encoder(image_encoder_type='MambaVision', **kwargs)
    _model_entrypoints['CLIP_MambaVision_S'] = lambda pretrained=False, **kwargs: clip_with_encoder(image_encoder_type='MambaVision', **kwargs)
    _model_entrypoints['CLIP_MambaVision_B'] = lambda pretrained=False, **kwargs: clip_with_encoder(image_encoder_type='MambaVision', **kwargs)
    _model_entrypoints['CLIP_MambaVision_L'] = lambda pretrained=False, **kwargs: clip_with_encoder(image_encoder_type='MambaVision', **kwargs)
    _model_entrypoints['CLIP_MambaVision_L2'] = lambda pretrained=False, **kwargs: clip_with_encoder(image_encoder_type='MambaVision', **kwargs)

# Initialize the registry
register_models()

def get_model_entrypoint(model_name):
    return _model_entrypoints.get(model_name, None)
