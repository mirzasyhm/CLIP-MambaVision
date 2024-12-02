import hashlib
import os
import urllib
import warnings
from packaging import version
from typing import Union, List, Dict

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from model import build_model
from simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS: Dict[str, Dict[str, str]] = {
    "RN50": {
        "url": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
        "sha256": "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
    },
    "RN101": {
        "url": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
        "sha256": "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599"
    },
    "RN50x4": {
        "url": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
        "sha256": "7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd"
    },
    "RN50x16": {
        "url": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
        "sha256": "52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa"
    },
    "RN50x64": {
        "url": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
        "sha256": "be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c"
    },
    "ViT-B/32": {
        "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "sha256": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
    },
    "ViT-B/16": {
        "url": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "sha256": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
    },
    "ViT-L/14": {
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "sha256": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
    },
    "ViT-L/14@336px": {
        "url": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
        "sha256": "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
    },
    # MambaVision models
    "MambaVision-T": {
        "url": "https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar",
        "sha256": None  # Set to actual SHA256 if available
    },
    "MambaVision-T2": {
        "url": "https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar",
        "sha256": None
    },
    "MambaVision-S": {
        "url": "https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar",
        "sha256": None
    },
    "MambaVision-B": {
        "url": "https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar",
        "sha256": None
    },
    "MambaVision-L": {
        "url": "https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar",
        "sha256": None
    },
    "MambaVision-L2": {
        "url": "https://huggingface.co/nvidia/MambaVision-L2-1K/resolve/main/mambavision_large2_1k.pth.tar",
        "sha256": None
    },
}


def _download(model_info: dict, root: str):
    """
    Downloads the model from the specified URL and verifies its integrity using SHA256 if provided.

    Parameters:
        model_info (dict): Dictionary containing 'url' and 'sha256' keys.
        root (str): Directory to save the downloaded model.

    Returns:
        str: Path to the downloaded model file.
    """
    url = model_info['url']
    sha256 = model_info.get('sha256')

    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target):
        if sha256:
            existing_sha256 = hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            if existing_sha256 == sha256:
                print(f"Using cached model at {download_target}")
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            print(f"Using cached model at {download_target}")
            return download_target

    # Download the file
    print(f"Downloading {url} to {download_target}...")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        total_size = int(source.info().get("Content-Length", 0))
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    # Verify SHA256 if provided
    if sha256:
        downloaded_sha256 = hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        if downloaded_sha256 != sha256:
            raise RuntimeError("Model has been downloaded but the SHA256 checksum does not match")
        else:
            print("SHA256 checksum verification passed.")
    else:
        print("No SHA256 checksum provided; skipping verification.")

    return download_target

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, vision_type: str, mamba_params: dict, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    
    vision_type : str
        The type of vision encoder to use ('resnet', 'vit', 'mambavision').
    
    mamba_params : dict
        Parameters specific to MambaVision if `vision_type` is 'mambavision'.
    
    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_info = _MODELS[name]
        model_path = _download(model_info, download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        # Determine vision_type based on model name if not explicitly provided
        inferred_vision_type = vision_type
        if 'mambavision' in name.lower():
            inferred_vision_type = 'mambavision'
        elif 'vit' in name.lower():
            inferred_vision_type = 'vit'
        else:
            inferred_vision_type = 'resnet'

        # If vision_type is 'mambavision' and mamba_params not provided, extract from state_dict
        if inferred_vision_type == 'mambavision' and mamba_params is None:
            # Extract MambaVision parameters from state_dict or define defaults
            # Adjust the keys based on your state_dict structure
            mamba_params = {
                'dim': state_dict.get('visual.dim', 128),
                'in_dim': state_dict.get('visual.in_dim', 64),
                'depths': state_dict.get('visual.depths', [3, 3, 10, 5]),
                'num_heads': state_dict.get('visual.num_heads', [2, 4, 8, 16]),
                'window_size': state_dict.get('visual.window_size', [8, 8, 14, 7]),
                'mlp_ratio': state_dict.get('visual.mlp_ratio', 4),
                'drop_path_rate': state_dict.get('visual.drop_path_rate', 0.2),
                'in_chans': state_dict.get('visual.in_chans', 3),
                'num_classes': state_dict.get('visual.num_classes', 1000),
                'qkv_bias': state_dict.get('visual.qkv_bias', True),
                'qk_scale': state_dict.get('visual.qk_scale', None),
                'drop_rate': state_dict.get('visual.drop_rate', 0.0),
                'attn_drop_rate': state_dict.get('visual.attn_drop_rate', 0.0),
                'layer_scale': state_dict.get('visual.layer_scale', None),
                'layer_scale_conv': state_dict.get('visual.layer_scale_conv', None),
            }

        model = build_model(state_dict or model.state_dict(), vision_type=inferred_vision_type)

        model = model.to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a node which is polymorphic over return type.
        
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
