from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI
from pydantic import BaseModel

tokenizer = AutoTokenizer.from_pretrained(Path.cwd() / 'model' / 'en_ru_local')
model = AutoModelForSeq2SeqLM.from_pretrained(Path.cwd() / 'model' / 'en_ru_local')
app = FastAPI()
def translate_phrase(phrase):
    inputs = tokenizer(phrase, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    out_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    return out_text[0]

class TextToTranslate(BaseModel): 
    text: str

@app.get("/") 
async def root(): 
    return {"message": "This is root page"}

@app.post("/en_ru/")
async def en_ru(_textToTranslate: TextToTranslate): 
    return {"translated_text": translate_phrase(_textToTranslate.text)}