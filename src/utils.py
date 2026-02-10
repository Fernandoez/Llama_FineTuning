import pandas as pd
import json
from pathlib import Path

def save_results(results, outputh_path_xlsx, outputh_path_json):
    Path(outputh_path_xlsx).parent.mkdir(parents=True, exist_ok=True)
    Path(outputh_path_json).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_excel(outputh_path_xlsx, index=False)

    json_data = [
        {"output": item["Resultado_FT"]}
        for item in results
    ]
    
    with open(outputh_path_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)