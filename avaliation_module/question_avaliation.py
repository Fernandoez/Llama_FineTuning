from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import json
from data.inferences_data import AVALIATION_PROMPT_STR
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

arq_questions = "data/questoes.json"

# Prompt com rubrica de avaliação usado pela LLM
avaliation_prompt_str = AVALIATION_PROMPT_STR

# Juntando o prompt com a rubrica de avaliação e a primeira questão do banco para testar a avaliação do LLM
with open(arq_questions, "r", encoding="utf-8") as f:
  questions = json.load(f)

final_prompt = avaliation_prompt_str + questions[0]['output']
print(final_prompt)

# Instanciando modelo GEMINI 2.5 flash - usado no chat do gemini
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)

# resp = gemini.invoke("Me fale sobre você")
# print(resp.content)

resp = gemini.invoke(final_prompt)
print(resp.content)
clean = resp.content.replace("```json", "").replace("```", "").strip()
content = json.loads(clean)
flat = {**content["scores"], **content["comments"], "overall": content["overall"]}
print(flat)

# NOVA COM SEPARAÇÃO DOS CAMPOS
# Função que passa todas as questões com a rubrica de avaliação pelos modelos
import re

def evaluation(model, questions):
  evaluation = []
  i = 1
  for q in questions:
    print(f'Questão {i}')
    question = q['output']
    final_prompt = avaliation_prompt_str + question
    resp_model = model.invoke(final_prompt)

    resposta_str = resp_model.content
    resposta_str = re.sub(r"```[a-zA-Z]*\n", "", resposta_str)
    resposta_str = resposta_str.replace("```", "").strip()

    m = re.search(r"\{[\s\S]*\}", resposta_str)
    if m:
      resposta_str = m.group(0)

    try:
      resp_model_dict = json.loads(resposta_str)

      flat={}
      if "scores" in resp_model_dict:
        flat.update(resp_model_dict["scores"])
      if "comments" in resp_model_dict:
        flat.update(resp_model_dict["comments"])
      flat["overall"] = resp_model_dict.get("overall", "")

      evaluation_text = json.dumps(resp_model_dict, ensure_ascii=False)
    except json.JSONDecodeError:
      print(f"Erro ao decodificar JSON na questão {i}")
      flat = {}
      flat["overall"] = ""
      evaluation_text = resposta_str

    final_result = {
        "question": question,
        **flat,
        "evaluation_text": evaluation_text,
    }
    evaluation.append(final_result)
    i = i+1

  return evaluation

evaluation_gemini_parafrase = evaluation(gemini, questions)
df = pd.DataFrame(evaluation_gemini_parafrase)
df.to_excel('avaliation_questoesgeradas_gemini.xlsx', index=False)