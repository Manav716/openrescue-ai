FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py README.md openenv.yaml ./
COPY env ./env
COPY scripts ./scripts
COPY models ./models
COPY plots ./plots
COPY app.py main.py ./

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

