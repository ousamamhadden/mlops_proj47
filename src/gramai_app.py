from fastapi import FastAPI
from happytransformer import HappyTextToText, TTSettings
from pydantic import BaseModel
from typing import Optional
from enum import Enum
from omegaconf import OmegaConf

cfg = OmegaConf.load("config.yaml")

app = FastAPI()
model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

@app.get("/")
def read_root():
    """Root endpoint."""
    return "GramAI API"

class TTSettingsParams(BaseModel):
    """TTSettings params."""
    num_beams: int
    min_length: int
    max_length: int

@app.post("/text/")
def correct_grammar(input_sentence: str, params: Optional[TTSettingsParams] = TTSettingsParams(num_beams=cfg.num_beams, min_length=cfg.min_length, max_length=cfg.max_length)):
    """Return a grammarly corrected text."""
    settings = TTSettings(num_beams=params.num_beams, min_length=params.min_length, max_length=params.max_length)
    output_sentence = model.generate_text("grammar: " + input_sentence, args=settings).text
    return {"corrected": output_sentence}
