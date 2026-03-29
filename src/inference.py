import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from data.inferences_data import SYSTEM_PROMPT

def prepare_tokenizer_for_inference(tokenizer):

    #Aplicando template de chat do LLaMA 3.1 ao tokenizer.

    return get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

def inference(model, tokenizer, instruction):
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids = inputs,
            max_new_tokens = 1024,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            use_cache=True
        )
  
    generated_ids = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def run_inference(model, tokenizer, themes, topics_restriction, question_format, stage: str):
    
    tokenizer = prepare_tokenizer_for_inference(tokenizer)

    results = []
    for theme in themes:
        for topic, restriction in topics_restriction.items():
            instruction = f"Crie uma questão de programação em Python, sobre {theme} e que seja do tópico de {topic}. {restriction} {question_format}"
            output = inference(model, tokenizer, instruction)
            results.append({
                "Temas": theme,
                "Tópicos": topic,
                "Prompt": instruction,
                stage: output
            })
    
    return results