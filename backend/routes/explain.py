import base64
import io
import time

import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from global_variables.config import DEVICE, IMG_SIZE
from global_variables.model_store import get_model, get_class_names, get_transform
from routes.predict import read_image
from schema import ExplainResponse

router = APIRouter(tags=["inference"])


@router.post("/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...)):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — train and export lunglens_resnet18.pt first")

    image = read_image(await file.read())
    class_names = get_class_names()

    start = time.perf_counter()
    tensor = get_transform()(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred_idx = int(probs.argmax())

    # Grad-CAM on the last conv block — needs gradients, so no no_grad here
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]

    rgb = np.array(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode()
    elapsed_ms = (time.perf_counter() - start) * 1000

    return ExplainResponse(
        predicted_class=class_names[pred_idx],
        confidence=round(float(probs[pred_idx]), 4),
        probabilities={name: round(float(p), 4) for name, p in zip(class_names, probs)},
        heatmap_png_base64=heatmap_b64,
        explanation=(
            f"Grad-CAM heatmap over ResNet-18's last conv block: highlighted regions contributed most "
            f"to the '{class_names[pred_idx]}' prediction. This shows where the model looked, "
            f"not a medical diagnosis."
        ),
        inference_time_ms=round(elapsed_ms, 1),
    )
