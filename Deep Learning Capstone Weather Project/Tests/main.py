from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import tensorflow.lite as tflite
from keras_image_helper import create_preprocessor
import os

app = FastAPI()

# Initialize Preprocessor & Model
preprocessor = create_preprocessor('xception', target_size=(299, 299))
interpreter = tflite.Interpreter(model_path='weather_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
classes = ['cloudy', 'rainy', 'shine', 'sunrise']

class UrlRequest(BaseModel):
    url: str

@app.get("/")
def home():
    return "Weather App is Live!"

@app.post("/predict")
def predict_endpoint(request: UrlRequest):
    X = preprocessor.from_url(request.url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)[0].tolist()
    return dict(zip(classes, preds))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
