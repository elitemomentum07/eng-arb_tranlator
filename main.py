from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import MarianTokenizer
from onnxruntime import InferenceSession
import numpy as np
import os
from pathlib import Path

app = FastAPI()

# Paths
BASE_DIR = Path(__file__).resolve().parent
ENCODER_PATH = BASE_DIR / "encoder_model-quant.onnx"
DECODER_PATH = BASE_DIR / "decoder_model-quant.onnx"
MAX_LENGTH = 50

# Load tokenizer and ONNX models
tokenizer = MarianTokenizer.from_pretrained(BASE_DIR)
encoder = InferenceSession(str(ENCODER_PATH))
decoder = InferenceSession(str(DECODER_PATH))

class Input(BaseModel):
    text: str

@app.post("/translate")
async def translate(data: Input):
    text = data.text.strip()
    if not text:
        return {"error": "Empty input"}

    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    encoder_hidden = encoder.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })[0]

    decoder_input_ids = np.array([[tokenizer.pad_token_id]]).astype(np.int64)
    translated_tokens = []

    for _ in range(MAX_LENGTH):
        output = decoder.run(None, {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden,
            "encoder_attention_mask": attention_mask
        })[0]
        next_token = np.argmax(output[:, -1, :], axis=-1)
        if next_token[0] == tokenizer.eos_token_id:
            break
        decoder_input_ids = np.hstack([decoder_input_ids, next_token[:, None]])
        translated_tokens.append(next_token[0])

    translated_text = tokenizer.decode(translated_tokens, skip_special_tokens=True)
    return {"arabic": translated_text}
