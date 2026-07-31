# utils/image_utils.py
from pathlib import Path
from typing import Union, Optional

from PIL import Image
from torchvision import transforms


def get_image_transform(image_size: int = 224):
    """Standard ImageNet-normalized transform used by the model."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])


def load_image_safely(source: Union[str, Path, Image.Image]) -> Optional[Image.Image]:
    """
    Load an image from path or return the PIL image directly.
    Returns None if loading fails.
    """
    try:
        if isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
        else:
            img = source.convert("RGB")
        return img
    except Exception as e:
        print(f"Failed to load image: {e}")
        return None