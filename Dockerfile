FROM python:3.13-slim
WORKDIR /app
COPY . /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cu126 \
    && pip install bitsandbytes accelerate \
    && pip install -r requirements.txt
EXPOSE 8001
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]