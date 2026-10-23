FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
COPY requirements-mlops.txt requirements-mlops.txt
# The image serves the registry and the route as well as /v1/score, so it needs MLflow.
RUN pip install --no-cache-dir -r backend/requirements.txt -r requirements-mlops.txt
COPY backend backend
COPY features features
COPY mlops mlops
COPY agents agents
COPY rag rag
COPY models models
ENV PYTHONPATH=/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "backend.risklens_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
