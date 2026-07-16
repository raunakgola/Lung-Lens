from fastapi import APIRouter

from global_variables.config import DEVICE
from global_variables.model_store import get_model, get_class_names
from schema import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    classes = get_class_names()
    return HealthResponse(
        status="ok",
        model_loaded=get_model() is not None,
        device=DEVICE,
        num_classes=len(classes),
        class_names=classes,
    )
