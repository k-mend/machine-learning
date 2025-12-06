# Lead Scoring Prediction API

A production-ready machine learning service that predicts lead conversion probability for an educational platform. This API helps identify which potential customers are most likely to convert, enabling targeted marketing strategies.

## 🎯 Project Overview

This project implements a REST API that serves a logistic regression model trained to predict whether a lead will convert to a paying customer. The model considers factors like lead source, engagement level (courses viewed), and annual income to make predictions.

### Business Use Case

Marketing teams can use this API to:
- Prioritize high-probability leads for follow-up
- Optimize ad spend by focusing on promising lead sources
- Personalize engagement strategies based on conversion likelihood
- Track and analyze lead quality across different channels

## 🛠️ Tech Stack

- **Python 3.12/3.13** - Core language
- **uv** - Modern, fast Python package manager
- **Scikit-learn** - Machine learning model
- **FastAPI** - High-performance web framework
- **Uvicorn** - ASGI server
- **Docker** - Containerization for deployment

## 📋 Prerequisites

- Python 3.12 or 3.13
- Docker (for containerized deployment)
- Basic understanding of REST APIs

## 🚀 Getting Started

### 1. Environment Setup

First, install uv - a fast Python package manager:

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv

# Verify installation
uv --version
```

### 2. Project Initialization

Create and initialize your project:

```bash
# Create project directory
mkdir lead-scoring-api
cd lead-scoring-api

# Initialize uv project
uv init
```

### 3. Install Dependencies

Install required packages using uv:

```bash
# Install core ML library
uv add scikit-learn==1.6.1

# Install web framework and server
uv add fastapi uvicorn

# For testing the API
uv add requests
```

### 4. Download the Model

Download the pre-trained model:

```bash
wget https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin
```

Verify the model file integrity:

```bash
md5sum pipeline_v1.bin
# Expected: 7d17d2e4dfbaf1e408e1a62e6e880d49
```

## 💻 Implementation

### Model Loading Script

Create `predict.py` to test the model locally:

```python
import pickle

# Load the trained pipeline
with open('pipeline_v1.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Example lead data
lead = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Transform and predict
X = dv.transform([lead])
probability = model.predict_proba(X)[0, 1]

print(f"Conversion Probability: {probability:.3f}")
print(f"Predicted to Convert: {probability >= 0.5}")
```

Run the test:

```bash
python predict.py
```

### FastAPI Service

Create `api.py` to serve the model:

```python
from fastapi import FastAPI, Request
import pickle

app = FastAPI(title="Lead Scoring API", version="1.0")

# Load model at startup
with open('pipeline_v1.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

@app.get("/")
async def root():
    return {
        "message": "Lead Scoring API",
        "version": "1.0",
        "endpoints": {
            "predict": "/predict (POST)"
        }
    }

@app.post("/predict")
async def predict(request: Request):
    """
    Predict lead conversion probability
    
    Expected input:
    {
        "lead_source": str,
        "number_of_courses_viewed": int,
        "annual_income": float
    }
    """
    lead_data = await request.json()
    
    # Transform and predict
    X = dv.transform([lead_data])
    probability = model.predict_proba(X)[0, 1]
    
    return {
        "churn_probability": float(probability),
        "churn": bool(probability >= 0.5),
        "lead_quality": "high" if probability >= 0.6 else "medium" if probability >= 0.4 else "low"
    }
```

### Run Locally

Start the development server:

```bash
uvicorn api:app --reload --port 9696
```

Test the API:

```python
import requests

url = "http://localhost:9696/predict"

# Test case 1: Organic search lead
lead = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=lead).json()
print(f"Conversion Probability: {response['churn_probability']:.3f}")
print(f"Lead Quality: {response['lead_quality']}")
```

## 🐳 Docker Deployment

### Create Dockerfile

Create a `Dockerfile` for production deployment:

```dockerfile
FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

# Copy application files
COPY api.py .
COPY pipeline_v1.bin .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn

# Expose API port
EXPOSE 9696

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import requests; requests.get('http://localhost:9696')"

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "9696"]
```

### Build and Run Container

```bash
# Build the image
docker build -t lead-scoring-api:v1 .

# Check image size
docker images lead-scoring-api:v1

# Run the container
docker run -d -p 9696:9696 --name lead-scorer lead-scoring-api:v1

# View logs
docker logs lead-scorer

# Stop container
docker stop lead-scorer
docker rm lead-scorer
```

### Test Containerized API

```python
import requests

url = "http://localhost:9696/predict"

# High-quality lead example
lead = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=lead)
print(response.json())
```

## 📊 API Documentation

Once running, visit:
- API Docs: `http://localhost:9696/docs`
- Alternative Docs: `http://localhost:9696/redoc`

### Endpoints

#### `POST /predict`

Predicts lead conversion probability.

**Request Body:**
```json
{
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
```

**Response:**
```json
{
    "churn_probability": 0.534,
    "churn": true,
    "lead_quality": "medium"
}
```

**Lead Sources:**
- `organic_search` - Found through search engines
- `paid_ads` - Came from paid advertising
- `referral` - Referred by existing customers
- `social_media` - Social media campaigns
- `direct` - Direct website visits

## 🔍 Model Details

### Features Used

1. **lead_source** (categorical) - Marketing channel origin
2. **number_of_courses_viewed** (numeric) - Engagement indicator
3. **annual_income** (numeric) - Financial capacity indicator

### Model Pipeline

- **Preprocessing:** DictVectorizer for feature encoding
- **Algorithm:** Logistic Regression (liblinear solver)
- **Threshold:** 0.5 for binary classification

### Performance Considerations

The model provides probability scores (0-1), allowing flexible threshold adjustment based on business needs:
- **Conservative** (threshold > 0.6): Focus on highest-quality leads
- **Balanced** (threshold = 0.5): Standard approach
- **Aggressive** (threshold < 0.4): Maximize reach

## 🧪 Testing

Create `test_api.py`:

```python
import requests

def test_api():
    url = "http://localhost:9696/predict"
    
    test_cases = [
        {
            "name": "High engagement organic lead",
            "data": {
                "lead_source": "organic_search",
                "number_of_courses_viewed": 4,
                "annual_income": 80304.0
            }
        },
        {
            "name": "Low engagement paid ad",
            "data": {
                "lead_source": "paid_ads",
                "number_of_courses_viewed": 2,
                "annual_income": 79276.0
            }
        }
    ]
    
    for test in test_cases:
        response = requests.post(url, json=test["data"])
        result = response.json()
        print(f"\n{test['name']}:")
        print(f"  Probability: {result['churn_probability']:.3f}")
        print(f"  Quality: {result['lead_quality']}")

if __name__ == "__main__":
    test_api()
```

## 🚀 Production Deployment

### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "9696:9696"
    environment:
      - LOG_LEVEL=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9696"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Deploy:

```bash
docker-compose up -d
```

### Cloud Deployment Options

- **AWS:** ECS/Fargate or EC2
- **Google Cloud:** Cloud Run or GKE
- **Azure:** Container Instances or AKS
- **Heroku:** Container deployment

## 📈 Monitoring and Logging

Add basic logging to `api.py`:

```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(request: Request):
    lead_data = await request.json()
    
    X = dv.transform([lead_data])
    probability = model.predict_proba(X)[0, 1]
    
    # Log prediction
    logger.info(f"Prediction made: {probability:.3f} | Source: {lead_data.get('lead_source')}")
    
    return {
        "churn_probability": float(probability),
        "churn": bool(probability >= 0.5),
        "lead_quality": "high" if probability >= 0.6 else "medium" if probability >= 0.4 else "low"
    }
```

## 🤝 Contributing

This project is part of the ML Zoomcamp course. Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## 📝 License

Educational project - ML Zoomcamp 2025

## 🔗 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Docker Documentation](https://docs.docker.com/)
- [uv Documentation](https://github.com/astral-sh/uv)

---

**Built with ❤️ for ML Zoomcamp 2025**