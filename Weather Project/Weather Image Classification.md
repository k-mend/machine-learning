# Weather Image Classification - Serverless Deployment

This project deploys a Deep Learning model (TensorFlow/Keras) to **AWS Lambda** using **Docker**. The model classifies images into four weather categories: `Cloudy`, `Rainy`, `Shine` (Sunny), and `Sunrise`.

This project serves as a submission for the **ML Zoomcamp** capstone project, demonstrating how to containerize a deep learning model and deploy it as a serverless function.

## ❓ Problem Statement

In many industries—such as agriculture, outdoor event planning, and autonomous transportation—reacting to real-time weather conditions is critical. While traditional weather sensors provide numerical data (temperature, humidity), visual verification remains a challenge.

**The goal of this project is to build an automated system that can:**
1.  **Analyze** an image from a camera or a public URL.
2.  **Classify** the visible weather condition into one of four distinct categories: *Cloudy*, *Rainy*, *Shine*, or *Sunrise*.
3.  **Deploy** this solution to the cloud so it can be accessed via API without maintaining physical servers.

By solving this, we enable applications to "see" the weather, providing a secondary validation layer to standard meteorological data.

---

## 📌 Project Overview

* **Model**: Xception (Pre-trained), converted to TensorFlow Lite (`weather_model.tflite`).
* **Infrastructure**: AWS Lambda (Serverless Compute) + AWS ECR (Container Registry).
* **Container Base**: `python:3.10-slim-bookworm`.
* **Key Libraries**: `tflite_runtime` (or `tensorflow-cpu`), `keras_image_helper`, `awslambdaric`.

---

## 📂 Project Structure

```bash


├── best_weather_model.keras                # Raw Tensorflow model before converting to tensorflow lite
├── Dataset                                 # Final Cleaned dataset used to train models
├── Dockerfile                              # Instructions to build the container image
├── lambda_function.py                      # The entry point for AWS Lambda
├── Multi-class Weather Dataset             # Original Raw Un# The trained TFLite modelrocessed data
├── Multiclass_Weather_dataset .ipynb       # The raw notebook used to train the model
├── test.py                                 # The python file used to test docker response locally
└── weather_model.tflite                    # The trained compressed final TFLite model


## 🐳 Docker Configuration

We use a custom Dockerfile to handle system dependencies required by TensorFlow and the AWS Lambda Runtime Interface Client (RIC).

**`Dockerfile`**:

```dockerfile
# 1. Use the STABLE 'bookworm' version of the slim image to avoid 'libpcre' errors
FROM python:3.10-slim-bookworm

# 2. Install system dependencies
# libpcre2-dev is required for the Lambda Runtime Interface Client
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Python libraries
# awslambdaric: Required to make the container work on AWS Lambda
RUN pip install awslambdaric tensorflow-cpu keras-image-helper

# 4. Set working directory
WORKDIR /var/task

# 5. Copy function code and model
COPY lambda_function.py .
COPY weather_model.tflite .

# 6. Set the entrypoint for AWS Lambda
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "lambda_function.lambda_handler" ]

```

---

## 🚀 Cloud Deployment (Step-by-Step)

This section details how to deploy the container to AWS.

### Prerequisites

* **AWS CLI** installed and configured (`aws configure`).
* **Docker** installed and running.
* **AWS Account** with permissions for ECR and Lambda.

### Step 1: Create an ECR Repository

Elastic Container Registry (ECR) is where we store our Docker images.

1. Log in to the AWS Console.
2. Go to **ECR (Elastic Container Registry)**.
3. Click **Create repository**.
4. Name: `weather-model`.
5. Settings: Private, Mutable tags.

### Step 2: Login and Build

*Note: Replace `[ACCOUNT_ID]` and `[REGION]` with your specific details (e.g., `eu-west-1`).*

```bash
# 1. Authenticate Docker with AWS ECR
aws ecr get-login-password --region [REGION] | sudo docker login --username AWS --password-stdin [ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com

# 2. Build the image locally
# Note: We use the 'bookworm' tag to avoid dependency issues with newer Debian versions.
sudo docker build -t weather-model .

```

### Step 3: Tag and Push

We need to tag the local image so it points to the AWS remote repository.

```bash
# 1. Tag the image
sudo docker tag weather-model:latest [ACCOUNT_ID].dkr.ecr.[REGION][.amazonaws.com/weather-model:latest](https://.amazonaws.com/weather-model:latest)

# 2. Push to AWS
sudo docker push [ACCOUNT_ID].dkr.ecr.[REGION][.amazonaws.com/weather-model:latest](https://.amazonaws.com/weather-model:latest)

```

### Step 4: Create the Lambda Function

1. Go to **AWS Lambda Console**.
2. Click **Create function**.
3. Select **Container image** (Crucial step).
4. Name: `weather-app`.
5. **Container image URI**: Click "Browse images" and select the `weather-model` repo and `latest` tag you just pushed.
6. Architecture: `x86_64`.

### Step 5: Configure Lambda (CRITICAL)

The default settings are too weak for a TensorFlow model. You **must** change these to prevent timeouts.

1. Go to **Configuration** -> **General configuration** -> **Edit**.
2. **Memory**: Set to `1024 MB` (or higher).
3. **Timeout**: Set to `2 min` (Model loading takes time).
4. Click **Save**.

---

## 🧪 Testing

### Test via AWS Console

1. Go to the **Test** tab in Lambda.
2. Create a new event with the following JSON:
```json
{
  "url": "https://images.unsplash.com/photo-1626124902047-f3db8b02f740?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
}
```


3. Click **Test**. You should see a JSON response with prediction probabilities.

### Local Testing (Optional)

You can test the container locally using the Lambda Runtime Interface Emulator (if installed) or by entering the shell:

```bash
# Run container interactively
sudo docker run -it --entrypoint /bin/bash weather-model

# Inside the container:
python3
>>> import lambda_function
>>> lambda_function.predict("[https://some-image-url.jpg](https://some-image-url.jpg)")

```

---

## 🛠 Troubleshooting Common Errors

### 1. `libpcre3-dev` not found

**Cause:** Debian "Trixie" (Testing) deprecated this package.
**Fix:** We switched the base image to `python:3.10-slim-bookworm` (Stable) and installed `libpcre2-dev` instead.

### 2. `KeyError: 'AWS_LAMBDA_RUNTIME_API'`

**Cause:** Running the container locally without the AWS Lambda emulator.
**Fix:** This is normal during local builds. It confirms the container is built correctly for the cloud. Use the manual Python entry method (described above) to verify logic locally.

### 3. `Permission denied` / `no basic auth credentials`

**Cause:** Docker requires `sudo`, but `aws ecr login` credentials were saved to the user profile, not root.
**Fix:** Run the login command with `sudo`:
`aws ecr get-login-password ... | sudo docker login ...`

---

## 📜 Credits

Developed as part of the **Machine Learning Zoomcamp** by DataTalks.Club.

```

```
