import torchvision.transforms as transforms
from fastapi import FastAPI
from PIL import Image
import matplotlib.pyplot as plt
from pydantic import BaseModel
from app.model.model import pre_image
from app.model.model import __version__ as model_version

from fastapi.responses import HTMLResponse
import uvicorn, base64
import numpy as np

app = FastAPI()


class TextIn(BaseModel):
	text: str

class PredictionOut(BaseModel):
	prediction: str
	prob: float

@app.get("/")
def home():
	return{"health_check": "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
async def predict(img_path: TextIn):
	pred, prob = pre_image(img_path.text)
	return{"prediction": pred, "prob": prob}


