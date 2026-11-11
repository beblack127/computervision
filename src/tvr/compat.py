"""Compatibility shims so the 2022 X-Pool code reproduces faithfully on a 2026 stack.

Every shim here is listed (with the reason) in docs/BASELINE_XPOOL.md, section
"Deviations / compatibility fixes". None of them changes the method.

1. Image transforms: the official code was run with torchvision 0.9 (PyTorch 1.8.1).
   On *tensor* inputs, torchvision 0.9 resized WITHOUT antialiasing; torchvision>=0.17
   defaults to antialias=True. That silently changes every test image
   (max |diff| ~0.13 on [0,1] pixels in our check). We pin antialias=False to match.
2. Checkpoints: transformers 4.6.1 stored `*.embeddings.position_ids` as persistent
   buffers in CLIPModel.state_dict(); recent transformers do not. Loading the official
   checkpoint with strict=True would fail on these two unexpected keys, so we drop
   exactly those keys and load everything else strictly.
3. torch.load: PyTorch>=2.6 defaults to weights_only=True. Official checkpoints are
   plain dicts of tensors/ints, so we try weights_only=True first and only fall back
   to weights_only=False for files we downloaded from the authors' links.
"""
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Keys present in transformers-4.6.1 checkpoints but not in recent CLIPModel state dicts.
LEGACY_BUFFER_SUFFIXES = (
    "text_model.embeddings.position_ids",
    "vision_model.embeddings.position_ids",
)


def init_transform_dict(input_res=224):
    """Same pipeline as third_party/xpool/datasets/model_transforms.py, with antialias pinned off."""
    return {
        "clip_test": transforms.Compose([
            transforms.Resize(input_res, interpolation=InterpolationMode.BICUBIC, antialias=False),
            transforms.CenterCrop(input_res),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]),
        "clip_train": transforms.Compose([
            # official: RandomResizedCrop(input_res, scale=(0.5, 1.0)) -> default bilinear, no antialias in tv0.9
            transforms.RandomResizedCrop(input_res, scale=(0.5, 1.0), antialias=False),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0, saturation=0, hue=0),  # no-op, kept for RNG parity
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]),
    }


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:  # noqa: BLE001 - legacy pickles from the authors' Google Drive
        return torch.load(path, map_location=map_location, weights_only=False)


def load_xpool_state_dict(model, state_dict, verbose=True):
    """Load an official/our checkpoint into a CLIPTransformer, tolerating legacy buffers only."""
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model_keys = set(model.state_dict().keys())
    dropped = [k for k in state_dict if k not in model_keys and k.endswith(LEGACY_BUFFER_SUFFIXES)]
    cleaned = {k: v for k, v in state_dict.items() if k not in dropped}
    result = model.load_state_dict(cleaned, strict=True)  # raises on any other mismatch
    if verbose and dropped:
        print(f"[compat] dropped legacy buffers (expected): {dropped}")
    return {"dropped_legacy_keys": dropped, "load_result": str(result)}


def build_tokenizer(huggingface=True):
    if huggingface:
        from transformers import CLIPTokenizer
        # Official code passes TOKENIZERS_PARALLELISM=False as a kwarg; the env var is equivalent.
        return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    from modules.tokenization_clip import SimpleTokenizer  # official non-HF tokenizer
    return SimpleTokenizer()
