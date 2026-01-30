from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_error: str | None = None
    model_diagnostics: dict | None = None


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_index: int


class ExplainResponse(BaseModel):
    image_base64: str
    predicted_class: str
    confidence: float
