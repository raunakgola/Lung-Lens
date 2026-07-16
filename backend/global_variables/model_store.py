"""Loads the fine-tuned ResNet-18 once at startup and shares it across routes."""
import json

import torch
import torch.nn as nn
from torchvision import models, transforms

from global_variables.config import (
    MODEL_PATH,
    CLASSES_PATH,
    DEFAULT_CLASS_NAMES,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEVICE,
)
from logger import get_logger

log = get_logger(__name__)

_model = None
_class_names = DEFAULT_CLASS_NAMES

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model() -> bool:
    global _model, _class_names
    if not MODEL_PATH.exists():
        log.warning(f"Model file not found at {MODEL_PATH} — /predict and /explain will return 503")
        return False

    if CLASSES_PATH.exists():
        with open(CLASSES_PATH) as f:
            _class_names = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(_class_names))
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    _model = model
    log.info(f"Model loaded on {DEVICE} with classes: {_class_names}")
    return True


def get_model():
    return _model


def get_class_names():
    return _class_names


def get_transform():
    return _transform
