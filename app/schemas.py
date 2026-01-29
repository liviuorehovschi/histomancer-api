from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_index: int


class ExplainResponse(BaseModel):
    image_base64: str
    predicted_class: str
    confidence: float
