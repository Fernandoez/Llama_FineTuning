from datasets import load_dataset

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples, tokenizer):
    
    EOS_TOKEN = tokenizer.eos_token # Adicionado ao final de cada questão

    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
      # EOS_TOKEN tem que ser adicionado para o modelo não ficar gerando para sempre
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts}

def load_and_prepare_dataset(json_path, tokenizer):

    dataset = load_dataset("json", data_files={"train":json_path}, split = "train")

    #aplicando a formatação compatível para o Llama 3.1
    dataset = dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer), 
        batched = True,
        remove_columns = dataset.column_names
    )
    return dataset

