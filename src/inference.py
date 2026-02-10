import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def prepare_tokenizer_for_inference(tokenizer):

    #Aplica o template de chat do LLaMA 3.1 ao tokenizer.

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    return tokenizer

def inference(model, tokenizer, instruction):
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "user", "content": instruction},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")
  
    outputs = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True)
    generated_ids = outputs[0][inputs.shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text

def run_inference(model, tokenizer, themes, topics_restriction, question_format):
    
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
                "Resultado_FT": output
            })
    
    return results