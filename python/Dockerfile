FROM python:3.11-slim

WORKDIR /app

# Install system deps if needed (none for this stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory for DB/logs
RUN mkdir -p /data && chmod 777 /data

COPY . .

# Streamlit config for container
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV RISKLENS_DB_PATH=/data/risklens.db
ENV RISKLENS_LOG_PATH=/data/risklens.log

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
