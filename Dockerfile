FROM python:3.11-slim

# HuggingFace Space metadata
# title: IT Helpdesk OpenEnv
# sdk: docker
# pinned: false

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY environment.py .
COPY app.py .
COPY inference.py .
COPY openenv.yaml .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
