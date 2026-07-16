import io
import time

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from global_variables.config import DEVICE
from global_variables.model_store import get_model, get_class_names, get_transform
from schema import PredictionResponse

router = APIRouter(tags=["inference"])


def read_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image (JPEG/PNG expected)")


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — train and export lunglens_resnet18.pt first")

    image = read_image(await file.read())
    class_names = get_class_names()

    start = time.perf_counter()
    tensor = get_transform()(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    pred_idx = int(probs.argmax())
    return PredictionResponse(
        predicted_class=class_names[pred_idx],
        confidence=round(float(probs[pred_idx]), 4),
        probabilities={name: round(float(p), 4) for name, p in zip(class_names, probs)},
        inference_time_ms=round(elapsed_ms, 1),
    )
