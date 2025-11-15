from fastapi import FastAPI
from pydantic import BaseModel
from app.model import *

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post('/predict')
def predict(input: InputText):
    return predict_sentiment(input.text)

@app.get('/')
def health_check():
    return {'status': 'ok'}