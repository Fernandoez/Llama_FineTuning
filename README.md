# Fine-Tuning de LLM para Geração de Questões de Programação
(Dockerfile apenas para uso pessoal. Pode ser completamente ignorado)

Este repositório contém o código utilizado no projeto de mestrado desenvolvido no **Programa de Pós-Graduação em Ciência da Computação (PPGCC) da Universidade Federal de Ouro Preto (UFOP)**.

## Objetivo
Investigar a eficácia do fine-tuning de um modelo de linguagem de grande porte (LLM), especificamente o Llama 3.1-8B, utilizando um dataset reduzido (126 questões), para gerar automaticamente questões práticas de programação em português, respeitando restrições pedagógicas e estruturais.

## Principais Características
- **Modelo base**: unsloth/Llama-3.1-8B-Instruct-bnb-4bit
- **Framework**: [Unsloth](https://github.com/unslothai/unsloth)
- **Técnicas aplicadas**: 
    - QLoRA (quantização em 4 bits)
    - PEFT (LoRA).
- **Tarefa**: Geração de enunciados de questões de programação
- **Domínio**: Programação de Computadores
- **Formato de dados**:
    - Questões: Questões formatadas em LaTeX
    - Dados para treinamento: Chat template Llama 3.1
- **Idioma**: Português (PT-BR)
- **Ambiente de execução**: Google Colab (GPU T4).

## Pipeline
O fluxo principal do projeto é:
1. Carregamento do modelo base (4-bit)
2. Inferência zero-shot (baseline)
3. Preparação do dataset (chat template Llama 3.1)
4. Fine-tuning
5. Salvar o modelo ajustado
6. Inferência pós fine-tuning
7. Exportação dos resultados

## Execução
Executar o pipeline completo:
python -m src.main

O pipeline irá:
- gerar questões antes do fine-tuning;
- treinar o modelo;
- salvar o modelo localmente e no Huggin Face (Caso não deseje salvar no HF, comentar função "push_model_to_hub" no arquivo src/main.py);
- gerar questões após o fine-tuning.

## Dataset
O dataset contém 126 exemplos estruturados com:
- instruction: descrição da tarefa
- input: campo opcional, vazio no caso deste treinamento
- output: questão final formatada

Durante o treinamento, os dados são convertidos para o formato de chat:
system → contexto pedagógico
user → instrução + input
assistant → resposta esperada

## Licença
Este projeto é destinado a fins acadêmicos.

## Autor
Fernando Euzebio Zimerman
Projeto desenvolvido como parte de pesquisa de mestrado em Ciência da Computação (UFOP).