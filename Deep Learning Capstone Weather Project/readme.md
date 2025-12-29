# Weather Image Classification (Deep Learning)- Cloud Deployment (Serverless & EC2)

This project utilizes pretrained **Xception** deep learning model using **google colab notebook** in order to utilize the free offered GPUs, to train and deploy the Deep Learning model to the cloud using two distinct architectures: **AWS Lambda (Serverless)** and **AWS EC2 (Web Service)**. The model classifies images into four weather categories: `Cloudy`, `Rainy`, `Shine` (Sunny), and `Sunrise`. Since Goolge colab was used, no local environment setup is required. 

This repository serves as a submission for the **ML Zoomcamp** capstone project, demonstrating how to containerize a deep learning model and deploy it to production using Docker and AWS.

## ❓ Problem Statement

In many industries—such as agriculture, outdoor event planning, and autonomous transportation—reacting to real-time weather conditions is critical. While traditional weather sensors provide numerical data (temperature, humidity), visual verification remains a challenge.

**The goal of this project is to build an automated system that can:**
1.  **Analyze** an image from a camera or a public URL.
2.  **Classify** the visible weather condition into one of four distinct categories: *Cloudy*, *Rainy*, *Shine*, or *Sunrise*.
3.  **Deploy** this solution to the cloud so it can be accessed via API without maintaining physical servers.

By solving this, we enable applications to "see" the weather, providing a secondary validation layer to standard meteorological data.

## 📌 Project Overview

* **Model**: Xception (Pre-trained), converted to TensorFlow Lite (`weather_model.tflite`).
* **Infrastructure**: Deployed in two Methods:
    * **Method A**: **AWS Lambda** (Serverless Compute) + **AWS ECR**.
    * **Method B**: **AWS EC2** (Virtual Server) running **FastAPI** + **Docker**.
* **Container Base**: `python:3.10-slim-bookworm`.
* **Key Libraries**: `tflite_runtime` (or `tensorflow-cpu`), `keras_image_helper`, `awslambdaric` (for Lambda), `fastapi`, `uvicorn`.



```markdown
---
---

## 📂 Project Structure

```bash
.
├── best_weather_model.keras        # Raw Tensorflow model before converting to tensorflow lite
├── Dataset                         # Final Cleaned dataset used to train models
├── Dockerfile                      # Instructions to build the Lambda container image
├── lambda_function.py              # The entry point for AWS Lambda
|
├── Test                            # The Subfolder used to deploy to EC2 service
|--|--/main.py                      # The entry point for FastAPI (EC2)
|--|--/Dockerfile.fastapi           # The Dockerfile Configuration for EC2 service deployment
|--|--/fastapi_deployment.py        # used to load the model during testing
|--|--/weather_model.tflite         # The lighter tensorflow model deployed to EC2 service via FastAPI
├── Multi-class Weather Dataset     # Original Raw Unprocessed data
├── Multiclass_Weather_dataset.ipynb# The raw notebook used to train the model
├── test.py                         # The python file used to test docker response locally before deploying the docker image
├── weather_model.tflite            # The trained compressed final TFLite model
└── README.md                       # Project documentation

```

---

## 🐳 Docker Configuration

We use a custom Dockerfile to handle system dependencies required by TensorFlow and the AWS Lambda Runtime Interface Client (RIC).

**`Dockerfile` (For Lambda)**:

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

## 🚀 Deployment Option 1: AWS Lambda (Serverless)

This section details how to deploy the container to AWS Lambda.

### Prerequisites

* **AWS CLI** installed and configured (`aws configure`).
* **Docker** installed and running.
* **AWS Account** with permissions for ECR and Lambda.

### Step 1: Create an ECR Repository

1. Log in to the AWS Console.
2. Go to **ECR (Elastic Container Registry)**.
3. Click **Create repository**.
4. Name: `weather-model`.
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/ecr%20repository%20configuration%20name.png)
6. Settings: Private, Mutable tags.
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/ecr%20repository%20configuration.png)

### Step 2: Login and Build

*Note: Replace `[ACCOUNT_ID]` and `[REGION]` with your specific details (e.g., `eu-west-1`).*
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/successful%20login%20to%20ecr%20server.png)

```bash
# 1. Authenticate Docker with AWS ECR
aws ecr get-login-password --region [REGION] | sudo docker login --username AWS --password-stdin [ACCOUNT_ID].dkr.ecr.[REGION].amazonaws.com

# 2. Build the image locally
sudo docker build -t weather-model .
```
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/successful%20build%20of%20docker%20image%20locally.png)
### Step 3: Tag and Push


```bash
# 1. Tag the image
sudo docker tag weather-model:latest [ACCOUNT_ID].dkr.ecr.[REGION][.amazonaws.com/weather-model:latest](https://.amazonaws.com/weather-model:latest)

# 2. Push to AWS
sudo docker push [ACCOUNT_ID].dkr.ecr.[REGION][.amazonaws.com/weather-model:latest](https://.amazonaws.com/weather-model:latest)
```
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/pushing%20docker%20image%20to%20ecr.png)


### Step 4: Create & Configure Lambda

![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/create_lambda_function_from_scratch.png)

1. Go to **AWS Lambda Console** -> **Create function**.
2. Select **Container image**.
3. **Container image URI**: Browse and select the `weather-model` repo and `latest` tag.
4. **Configuration (Critical)**:
* **Memory**: Set to `1024 MB`.
* **Timeout**: Set to `30 seconds`. - you can add timeout to say 2 min incase the model is too large and takes time to load.
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/aws%20lamda%20general%20configuration.png)



---

## ☁️ Deployment Option 2: AWS EC2 (Web Service)

This section details how to deploy the model as a continuous web service using FastAPI on an EC2 instance.

### Step 1: Launch an EC2 Instance

1. Go to **EC2 Console** -> **Launch Instance**.
2. **OS**: Ubuntu Server 24.04 LTS.
3. **Instance Type**: `t2.micro` (Free Tier eligible).
4. **Key Pair**: Create/Select a key pair (`.pem` file) and save it securely.
5. **Network**: Allow SSH traffic (Anywhere) and HTTP/HTTPS traffic.

![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/ec2_already%20created.png)

### Step 2: Configure Security Group

1. Click on the instance ID -> **Security** tab -> Click the **Security Group**.
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/change%20security%20groups.png)

3. **Edit inbound rules** -> **Add rule**.
4. Type: `Custom TCP`, Port: `8000`, Source: `0.0.0.0/0` (Anywhere).
5. Save rules.
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/edit%20inbounf%20rules.png)

### Step 3: Connect to Server

```bash
chmod 400 my-key-pair.pem
ssh -i "my-key-pair.pem" ubuntu@[YOUR_EC2_PUBLIC_IP]
```
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/successful%20ssh%20on%20terminal.png)


### Step 4: Install Docker on EC2

Run these commands inside the EC2 terminal:

```bash
sudo apt-get update
sudo apt-get install -y docker.io awscli
sudo systemctl start docker
sudo systemctl enable docker

```

### Step 5: Deploy the Container

1. **Build the FastAPI image locally**:
```bash
sudo docker build -f Dockerfile.fastapi -t weather-fastapi .
```
![alt text](https://github.com/k-mend/machine-learning/blob/main/Weather%20Project/screenshots/successful%20build%20of%20docker%20image%20locally.png)


2. **Push to ECR (using a new tag)**:
```bash
sudo docker tag weather-fastapi:latest [ACCOUNT_ID].dkr.ecr.[REGION][.amazonaws.com/weather-model:fastapi](https://.amazonaws.com/weather-model:fastapi)
sudo docker push [ACCOUNT_ID].dkr.ecr.[REGION][.amazonaws.com/weather-model:fastapi](https://.amazonaws.com/weather-model:fastapi)
```
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/terminal%20pushing%20the%20ec2%20docker%20image.png)


3. **Pull and Run on EC2**:
```bash
# Inside EC2 terminal
aws configure  # Enter your credentials
aws ecr get-login-password --region [REGION] | sudo docker login ...

sudo docker run -d -p 8000:8000 [ACCOUNT_ID].dkr.ecr.[REGION][.amazonaws.com/weather-model:fastapi](https://.amazonaws.com/weather-model:fastapi)

```


**AWS EC2 Instance Running:**
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/proof%20aws%20ec2%20instance%20running%20(aws%20page).png)

---

## 🧪 Testing

### Test via AWS Console (Lambda)

1. Go to the **Test** tab in Lambda.
2. Create a new event with the following JSON:
```json
{
  "url": "[https://images.unsplash.com/photo-1626124902047-f3db8b02f740?q=80&w=1287&auto=format&fit=crop](https://images.unsplash.com/photo-1626124902047-f3db8b02f740?q=80&w=1287&auto=format&fit=crop)"
}
```

3. Click **Test**. You should see a JSON response with prediction probabilities.

![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/proof%20of%20aws%20lamda%20deployment%20and%20response.png)

### Local Testing (Optional)

```bash
# Run container interactively
sudo docker run -it --entrypoint /bin/bash weather-model

# Inside the container:
python3
>>> import lambda_function
>>> lambda_function.predict("[https://some-image-url.jpg](https://some-image-url.jpg)")
```
![alt text](https://github.com/k-mend/machine-learning/blob/main/Deep%20Learning%20Capstone%20Weather%20Project/screenshots/local%20docker%20image%20response%20before%20pushing%20to%20ecr.png)

---

## 🛠 Troubleshooting Common Errors

### 1. `Permission denied` / `no basic auth credentials`

**Cause:** Docker requires `sudo`, but credentials were saved to the user profile.
**Fix:** Run login with sudo: `aws ecr get-login-password ... | sudo docker login ...`

### 2. `Connection timed out` (SSH)

**Cause:** EC2 Security Group is blocking Port 22.
**Fix:** Add an inbound rule for SSH (Port 22) from `0.0.0.0/0`.

### 3. libpcre3-dev not found
**Cause**: Debian "Trixie" (Testing) deprecated this package. Fix: We switched the base image to python:3.10-slim-bookworm (Stable) and installed libpcre2-dev instead.

### 4. KeyError: 'AWS_LAMBDA_RUNTIME_API'
**Cause**: Running the container locally without the AWS Lambda emulator. Fix: This is normal during local builds. It confirms the container is built correctly for the cloud. Use the manual Python entry method (described above) to verify logic locally.

---

## 📜 Credits

Developed as part of the **Machine Learning Zoomcamp** by DataTalks.Club.

## Final thoughts and insights
This was a simple deep learning project that was heavily focused on notebook training and AWS configuration.

Made with love for the opensource community.
```
```

