from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import *
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

app = FastAPI(title="Model Service")

Model_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")

#metrics
REQUESTS = Counter('model_requests_total', 'Total number of requests to the model')
ERRORS = Counter('model_errors_total', 'Total number of errors in model requests')
LATENCY = Histogram('model_request_latency_seconds', 'Latency of model requests in seconds')

# Load model once
print("Loading model:", Model_NAME)
nlp = pipeline("sentiment-analysis", model=Model_NAME)

class InputText(BaseModel):
    text: str
    id: str = None


@app.post('/infer')
def infer(payload: InputText):
    REQUESTS.inc()
    start = time.time()
    try:
        res = nlp(payload.text, truncation=True)
        LATENCY.observe(time.time()-start)
        return {'id' : payload.id, 'result': res}
    except Exception as e:
        ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict')
def predict(input: InputText):
    return predict_sentiment(input.text)

@app.get('/health')
def health_check():
    return {'status': 'ok'}

@app.get('/metrics')
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)