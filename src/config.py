from dataclasses import dataclass

@dataclass
class Config:
    MODEL_NAME: str = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit"
    MAX_SEQ_LENGTH: int = 2048
    LOAD_IN_4BIT: bool = True
    SEED: int = 3407
    DTYPE: str | None = None

    TRAIN_FILE: str = "data/questoes.json"
    OUTPUT_DIR: str = "outputs"
    MODEL_OUTPUT_DIR: str = "models/llama-3.1-8b-4bit-questoes-programacao-ptbr"
    HUB_REPO_ID: str = "Fernandoez/llama-3.1-8b-4bit-questoes-programacao-ptbr"