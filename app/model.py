from transformers import pipeline
import torch

sentiment_analyzer = pipeline("sentiment-analysis")

def predict_sentiment(text:str):
    result = sentiment_analyzer(text)[0]
    print(result['label'])
    return {'label':result['label'], 'score':round(result['score'],4)}
