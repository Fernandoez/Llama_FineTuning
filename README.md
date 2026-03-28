# Fine-Tuning de LLM para Geração de Questões de Programação
(Dockerfile apenas para uso pessoal. Pode ser completamente ignorado)

Este repositório contém o código utilizado no meu projeto de mestrado no **Programa de Pós-Graduação em Ciência da Computação (PPGCC) da Universidade Federal de Ouro Preto (UFOP)**.

## Objetivo

Demonstrar a eficácia do fine-tuning de um modelo LLM (Llama 3.1-8B), mesmo com uma quantidade pequena de questões (126), para gerar automaticamente questões práticas de programação em português, seguindo uma estrutura padronizada.

## Principais Características

- **Modelo usado**: Llama 3.1-8B via [Unsloth](https://github.com/unslothai/unsloth).
- **Técnicas aplicadas**: QLoRA, PEFT (LoRA).
- **Tarefa**: Geração de enunciados de questões com tema e tópico definidos (foco em decisão simples em Python).
- **Formato de dados**: Alpaca prompt (instruction, input, output) e questões formatadas em LaTeX. //trocar para o usado no llama
- **Ambiente de execução**: Google Colab (GPU T4).

## Resultados
//Inserir resultados

## Conteúdo
//inserir conteúdo
