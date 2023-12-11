from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-ru"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer.save_pretrained(Path.cwd() / 'model' / 'en_ru_local')
    model.save_pretrained(Path.cwd() / 'model' / 'en_ru_local')
    
load_model()