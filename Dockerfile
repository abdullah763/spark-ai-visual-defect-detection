FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install CPU-only PyTorch for inference.
RUN pip install --no-cache-dir \
    torch==2.14.1 \
    torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install only the packages required by the API.
COPY requirements-api.txt .

RUN pip install --no-cache-dir \
    -r requirements-api.txt

# Copy application code and model directory.
COPY src ./src
COPY api ./api
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
