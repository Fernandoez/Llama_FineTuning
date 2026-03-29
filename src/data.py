from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from data.inferences_data import SYSTEM_PROMPT

REQUIRED_COLUMNS = {"instruction", "input", "output"}

def validate_dataset_columns(dataset):
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")
    
def formatting_prompts_func(examples, tokenizer):
    
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        user_content = instruction
        if input_text:
            user_content += "\nInput:\n" + input_text

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
        texts.append(
            tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

    return {"text": texts}


def load_and_prepare_dataset(json_path, tokenizer):
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    dataset = load_dataset("json", data_files={"train":json_path}, split = "train")
    validate_dataset_columns(dataset)

    #formatando no padrão do modelo
    dataset = dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer), 
        batched = True,
        remove_columns = dataset.column_names
    )
    return dataset

