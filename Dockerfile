# Python 3.12 slim base
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install CPU-only PyTorch first (smaller image), then other deps
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code only — model is downloaded from HuggingFace Hub at runtime
COPY app.py .
COPY templates/ templates/

# Render uses port 10000 by default
EXPOSE 10000

# Use gunicorn with eventlet worker for production (single worker for model in memory)
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:10000", "--timeout", "120", "app:app"]