from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import tensorflow.lite as tflite
from keras_image_helper import create_preprocessor
import os

app = FastAPI()

# 1. Initialize Preprocessor
preprocessor = create_preprocessor('xception', target_size=(299, 299))

# 2. Load Model
model_file = 'weather_model.tflite'
interpreter = tflite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = ['cloudy', 'rainy', 'shine', 'sunrise']

# Define request body structure
class UrlRequest(BaseModel):
    url: str

def predict_logic(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    return dict(zip(classes, float_predictions))

@app.post("/predict")
def predict_endpoint(request: UrlRequest):
    result = predict_logic(request.url)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
