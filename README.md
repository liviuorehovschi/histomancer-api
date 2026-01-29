---
title: Histomancer API
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

## Histomancer API

FastAPI backend for 3-class lung histopathology image classification.

- **GET /health** — Health check
- **POST /predict** — Image upload, returns predicted class and confidence
- **POST /gradcam** — Grad-CAM explainability (base64 PNG)
- **POST /saliency** — Saliency map explainability (base64 PNG)
