from unsloth import FastLanguageModel
from .config import Config

def load_model(cfg: Config):
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = cfg.MODEL_NAME,
    max_seq_length = cfg.MAX_SEQ_LENGTH,
    dtype = cfg.DTYPE,
    load_in_4bit = cfg.LOAD_IN_4BIT,
    )
    return model, tokenizer