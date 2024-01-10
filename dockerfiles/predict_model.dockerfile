# Base image
FROM python:3.11.5-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#We install requirements before any copying for better caching 
COPY requirements.txt requirements.txt
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY config/ config/ 
COPY models/ models/ #We could mount this instead.
COPY reports/ reports/

ENTRYPOINT ["python", "-u", "src/predict_model.py"]
#docker run -e WANDB_API_KEY=USEYOURKEYHERE predict:latest models/trained_model.pt data/processed/test_set.pt