from .model import load_model
from .data import load_and_prepare_dataset
from .train import train
from .inference import run_inference
from data.inferences_data import THEMES, TOPICS_RESTRICTIONS, QUESTION_FORMAT
from .utils import save_results
from pathlib import Path

OUT_DIR = Path("/app/output")
OUT_DIR.mkdir(exist_ok=True)

def main():
    model, tokenizer = load_model(train=True)
    dataset = load_and_prepare_dataset("data/questoes.json")
    train(model, tokenizer, dataset)
    results = run_inference(model, tokenizer, THEMES, TOPICS_RESTRICTIONS, QUESTION_FORMAT)
    save_results(results, outputh_path_xlsx = "/app/output/questoes_geradas_FT.xlsx", outputh_path_json= "/app/output/questoes_geradas_FT.json")

if __name__ == "__main__":
    main()