FROM agrigorev/zoomcamp-model:2025
WORKDIR /app

COPY api.py .
COPY pipeline_v1.bin .

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn==1.6.1

EXPOSE 9696
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "9696"]