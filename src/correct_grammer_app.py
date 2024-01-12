from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    """Root endpoint."""
    return "Grammer Correction API"


@app.get("/text/{text}")
def read_text(text: str):
    """Get a text."""
    return text