from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect
import re
import requests
import os
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Preprocessing Service")

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model_service:8001/infer")

REQUESTS = Counter('preproc_requests_total', 'Total preproc requests')
LATENCY = Histogram('preproc_latency_seconds', 'Preproc latency seconds')

class RawInput(BaseModel):
    text: str
    id: str = None

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

@app.post("/process_and_forward")
def process(payload: RawInput):
    REQUESTS.inc()
    start = time.time()
    try:
        text = clean_text(payload.text)
        lang = detect(text) if text else "unknown"
        # Optionally block unsupported languages
        if lang != 'en':
            # still forward, but include lang info
            pass
        # Forward to model service
        resp = requests.post(MODEL_SERVICE_URL, json={"text": text, "id": payload.id}, timeout=10)
        LATENCY.observe(time.time() - start)
        return {"id": payload.id, "language": lang, "model_response": resp.json()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
