from typing import List

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from .model import predict_digit

app = FastAPI()


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    pred = predict_digit(image)
    return {'prediction': pred}


@app.post('/predict_batch')
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        image = Image.open(file.file)
        pred = predict_digit(image)
        results.append({'filename': file.filename, 'prediction': pred})
    return results
