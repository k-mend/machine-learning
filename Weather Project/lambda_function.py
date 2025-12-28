import tensorflow.lite as tflite
from keras_image_helper import create_preprocessor
import os

# 1. Initialize Preprocessor
preprocessor = create_preprocessor('xception', target_size=(299, 299))

# 2. Load Model (The Fix: Read as bytes first)
model_file = 'weather_model.tflite'

if os.path.exists(model_file):
    with open(model_file, 'rb') as f:
        tflite_model = f.read()
    interpreter = tflite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
else:
    # Diagnostic: If this prints, the file wasn't copied to Docker
    raise RuntimeError(f"Model file '{model_file}' not found. Current dir: {os.getcwd()}, Content: {os.listdir('.')}")

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = ['cloudy', 'rainy', 'shine', 'sunrise']

# 3. Predict Logic
def predict(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    return dict(zip(classes, float_predictions))

# 4. Lambda Handler
def lambda_handler(event, context):
    url = event.get('url')
    return predict(url)
