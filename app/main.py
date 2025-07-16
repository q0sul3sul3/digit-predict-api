from fastapi import FastAPI, File, UploadFile
from PIL import Image

from .model import predict_digit

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    pred = predict_digit(image)
    return {"prediction": pred}
