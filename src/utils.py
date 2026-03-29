import pandas as pd
import json
from pathlib import Path

def save_results(results, output_path_xlsx, output_path_json, stage: str):
    Path(output_path_xlsx).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path_json).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_excel(output_path_xlsx, index=False)

    json_data = [
        {"output": item[stage]}
        for item in results
    ]
    
    with open(output_path_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

def save_model_local(model, tokenizer, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.save_pretrained_merged(
        output_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    tokenizer.save_pretrained(output_dir)

def push_model_to_hub(model, tokenizer, repo_id, token):
    if not token:
        raise ValueError("Token do Hugging Face não encontrado.")
    model.push_to_hub_merged(
        repo_id,
        tokenizer,
        save_method="merged_16bit",
        token=token,
    )