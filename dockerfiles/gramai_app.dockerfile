# Base image
FROM python:3.11.5-slim

EXPOSE $PORT

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

COPY pyproject.toml pyproject.toml
COPY src/gramai_app.py src/gramai_app.py
COPY src/config.yaml src/config.yaml
COPY models/ models/

WORKDIR /src
CMD exec uvicorn gramai_app:app --port $PORT --host 0.0.0.0 --workers 1
