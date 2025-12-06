FROM python:3.12-slim

RUN pip install pipenv

WORKDIR /app

# Copy dependency files
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

# Copy application files
COPY ["LOAD.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

# Start the application with Gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "LOAD:app"]