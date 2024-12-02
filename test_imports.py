# test_imports.py

from registry import list_models, is_model, model_entrypoint

def main():
    # List all registered models
    models = list_models()
    print("Available Models:", models)
    
    # Check if specific models are registered
    print("Is 'CLIP_Original' a registered model?", is_model('CLIP_Original'))
    print("Is 'CLIP_MambaVision_T' a registered model?", is_model('CLIP_MambaVision_T'))
    
    # Instantiate a model
    if is_model('CLIP_Original'):
        model_original_fn = model_entrypoint('CLIP_Original')
        model_original = model_original_fn(pretrained=False)
        print("Loaded CLIP_Original Model:", model_original)
    
    if is_model('CLIP_MambaVision_T'):
        model_mamba_fn = model_entrypoint('CLIP_MambaVision_T')
        model_mamba = model_mamba_fn(pretrained=False)
        print("Loaded CLIP_MambaVision_T Model:", model_mamba)

if __name__ == "__main__":
    main()
