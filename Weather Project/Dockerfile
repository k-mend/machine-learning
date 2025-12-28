# 1. Use the STABLE 'bookworm' version of the slim image
FROM python:3.10-slim-bookworm

# 2. Install dependencies (Updated libpcre3 -> libpcre2)
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Install the AWS Lambda Runtime Interface Client (RIC)
RUN pip install awslambdaric

# 4. Install ML dependencies
RUN pip install tensorflow-cpu keras-image-helper

# 5. Set up the function directory
WORKDIR /var/task
COPY lambda_function.py .
COPY weather_model.tflite .

# 6. Set the entrypoint
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "lambda_function.lambda_handler" ]
