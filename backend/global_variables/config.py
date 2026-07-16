from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parents[1]  # backend/
ROOT_DIR = BASE_DIR.parent                       # repo root


def _find_artifact(filename: str) -> Path:
    # Prefer backend/saved_models; fall back to a saved_models/ extracted at repo root
    for directory in (BASE_DIR / "saved_models", ROOT_DIR / "saved_models"):
        if (directory / filename).exists():
            return directory / filename
    return BASE_DIR / "saved_models" / filename


MODEL_PATH = _find_artifact("lunglens_resnet18.pt")
CLASSES_PATH = _find_artifact("classes.json")

# Fallback if classes.json is missing (ImageFolder sorts alphabetically)
DEFAULT_CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
