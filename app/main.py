import io
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from PIL import Image

from app.model import load_model, get_class_names, get_model_diagnostics, preprocess_image, predict
from app.explainability import gradcam_image, saliency_map, encode_png
from app.schemas import HealthResponse, PredictResponse, ExplainResponse

app = FastAPI(title="Histomancer API", version="1.0.0")
_model_loaded = False
_model_load_error: str | None = None


@app.on_event("startup")
def startup():
    global _model_loaded, _model_load_error
    try:
        load_model()
        _model_loaded = True
        _model_load_error = None
        logger.info("Model loaded successfully")
    except Exception as e:
        _model_loaded = False
        _model_load_error = f"{type(e).__name__}: {e}"
        logger.exception("Model failed to load: %s", e)


@app.get("/health", response_model=HealthResponse)
def health():
    if _model_loaded:
        return HealthResponse(status="ok", model_loaded=True)
    return HealthResponse(
        status="ok",
        model_loaded=False,
        model_error=_model_load_error,
        model_diagnostics=get_model_diagnostics(),
    )


async def _load_image(file: UploadFile) -> Image.Image:
    raw = await file.read()
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        img = await _load_image(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    x = preprocess_image(img)
    class_names = get_class_names()
    idx, probs = predict(x)
    return PredictResponse(
        predicted_class=class_names[idx],
        confidence=float(probs[idx]),
        class_index=idx,
    )


@app.post("/gradcam", response_model=ExplainResponse)
async def gradcam_endpoint(file: UploadFile = File(...)):
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        img = await _load_image(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    overlay, pred_idx, conf = gradcam_image(img)
    class_names = get_class_names()
    b64 = encode_png(overlay)
    return ExplainResponse(
        image_base64=b64,
        predicted_class=class_names[pred_idx],
        confidence=conf,
    )


@app.post("/saliency", response_model=ExplainResponse)
async def saliency_endpoint(file: UploadFile = File(...)):
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        img = await _load_image(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    sal_rgb, pred_idx, conf = saliency_map(img)
    class_names = get_class_names()
    b64 = encode_png(sal_rgb)
    return ExplainResponse(
        image_base64=b64,
        predicted_class=class_names[pred_idx],
        confidence=conf,
    )
