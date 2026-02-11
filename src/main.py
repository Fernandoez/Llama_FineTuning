from src.model import load_model
from src.data import load_and_prepare_dataset
from src.train import train
from src.inference import run_inference
from data.inferences_data import THEMES, TOPICS_RESTRICTIONS, QUESTION_FORMAT
from src.utils import save_results
from pathlib import Path

OUT_DIR = Path("/app/output")
OUT_DIR.mkdir(exist_ok=True)

def main():
    model, tokenizer = load_model()
    print("--- Início das inferências antes do FT ---")
    results = run_inference(model, tokenizer, THEMES, TOPICS_RESTRICTIONS, QUESTION_FORMAT)
    save_results(results, outputh_path_xlsx = "/app/output/questoes_geradas_ZS.xlsx", outputh_path_json= "/app/output/questoes_geradas_ZS.json")
    print("--- Fim das inferências antes do FT ---")
    print("--- Início do fine-tuning ---")
    dataset = load_and_prepare_dataset("data/questoes.json", tokenizer)
    train(model, tokenizer, dataset)
    print("--- Fim do fine-tuning ---")
    print("--- Início das inferências depois do FT ---")
    results = run_inference(model, tokenizer, THEMES, TOPICS_RESTRICTIONS, QUESTION_FORMAT)
    save_results(results, outputh_path_xlsx = "/app/output/questoes_geradas_FT.xlsx", outputh_path_json= "/app/output/questoes_geradas_FT.json")
    print("--- Fim das inferências depois do FT ---")

if __name__ == "__main__":
    main()