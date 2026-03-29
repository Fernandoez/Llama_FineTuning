from src.config import Config
from src.model import load_model
from src.data import load_and_prepare_dataset
from src.train import train
from src.inference import run_inference
from data.inferences_data import THEMES, TOPICS_RESTRICTIONS, QUESTION_FORMAT
from src.utils import save_model_local, push_model_to_hub, save_results
from dotenv import load_dotenv
import os

load_dotenv()
HUGGING_FACE_AUTENTICATION_TOKEN= os.getenv("HUGGING_FACE_AUTENTICATION_TOKEN")

def main():
    cfg = Config()
    model, tokenizer = load_model(cfg)

    print("--- Início das inferências antes do FT ---")
    results = run_inference(model, tokenizer, THEMES, TOPICS_RESTRICTIONS, QUESTION_FORMAT, "Resultado_ZS")
    save_results(
        results, 
        output_path_xlsx = "outputs/questoes_geradas_ZS.xlsx", 
        output_path_json= "outputs/questoes_geradas_ZS.json",
        stage="Resultado_ZS"
    )
    print("--- Fim das inferências antes do FT ---")

    print("--- Início do fine-tuning ---")
    dataset = load_and_prepare_dataset("data/questoes.json", tokenizer)
    trained_model = train(model, tokenizer, dataset, cfg)
    print("--- Fim do fine-tuning ---")

    print("--- Salvando modelo treinado ---")
    save_model_local(trained_model, tokenizer, "model/llama-3.1-8b-4bit-questoes-programacao-ptbr")
    push_model_to_hub(trained_model, tokenizer, "Fernandoez/llama-3.1-8b-4bit-questoes-programacao-ptbr", HUGGING_FACE_AUTENTICATION_TOKEN)

    print("--- Início das inferências depois do FT ---")
    results = run_inference(trained_model, tokenizer, THEMES, TOPICS_RESTRICTIONS, QUESTION_FORMAT, "Resultado_FT")
    save_results(
        results,
        output_path_xlsx = "outputs/questoes_geradas_FT.xlsx",
        output_path_json= "outputs/questoes_geradas_FT.json",
        stage="Resultado_FT"
    )
    print("--- Fim das inferências depois do FT ---")

if __name__ == "__main__":
    main()