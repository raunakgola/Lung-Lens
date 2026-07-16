from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    inference_time_ms: float


class ExplainResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    heatmap_png_base64: str
    explanation: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int
    class_names: list[str]
