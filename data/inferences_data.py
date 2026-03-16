THEMES = ['Campeonato de futebol', 'Competição de dança', 'Desconto em cursos', 'Jogos de azar']

TOPICS_RESTRICTIONS = {'Variáveis, expressões, entradas e saídas': 'Tópicos de programação permitidos na questão: Variáveis, expressões, entradas e saídas. Tópicos de programação não permitidos na questão: Decisão simples; Decisão complexa; Laços de repetição com for; Laços de repetição com while; Laços de repetição com for e while; Laços de repetição aninhados; Funções; Estrutura de dados homogêneos com vetores; Estrutura de dados homogêneos com matrizes.',
            'Decisão simples(if, else e não aninhados)': 'Tópicos de programação permitidos na questão: Variáveis, expressões, entradas e saídas; Decisão simples. Tópicos de programação não permitidos na questão: Decisão complexa; Laços de repetição com for; Laços de repetição com while; Laços de repetição com for e while; Laços de repetição aninhados; Funções; Estrutura de dados homogêneos com vetores; Estrutura de dados homogêneos com matrizes.',
            'Decisão complexa(if, elif, else e aninhados)': 'Tópicos de programação permitidos na questão: Variáveis, expressões, entradas e saídas; Decisão simples; Decisão complexa. Tópicos de programação não permitidos na questão: Laços de repetição com for; Laços de repetição com while; Laços de repetição com for e while; Laços de repetição aninhados; Funções; Estrutura de dados homogêneos com vetores; Estrutura de dados homogêneos com matrizes.',
            'Laços de repetição com for': 'Tópicos de programação permitidos na questão: Variáveis, expressões, entradas e saídas; Decisão simples; Decisão complexa; Laços de repetição com for. Tópicos de programação não permitidos na questão: Laços de repetição com while; Laços de repetição com for e while; Laços de repetição aninhados; Funções; Estrutura de dados homogêneos com vetores; Estrutura de dados homogêneos com matrizes.',
            'Laços de repetição com while': 'Tópicos de programação permitidos na questão: Variáveis, expressões, entradas e saídas; Decisão simples; Decisão complexa; Laços de repetição com for; Laços de repetição com while. Tópicos de programação não permitidos na questão: Laços de repetição com for e while; Laços de repetição aninhados; Funções; Estrutura de dados homogêneos com vetores; Estrutura de dados homogêneos com matrizes.',
            'Laços de repetição com for e while': 'Tópicos de programação permitidos na questão: Variáveis, expressões, entradas e saídas; Decisão simples; Decisão complexa; Laços de repetição com for; Laços de repetição com while; Laços de repetição com for e while. Tópicos de programação não permitidos na questão: Laços de repetição aninhados; Funções; Estrutura de dados homogêneos com vetores; Estrutura de dados homogêneos com matrizes.',
            'Laços de repetição aninhados': 'Tópicos de programação permitidos na questão: Variáveis, expressões, entradas e saídas; Decisão simples; Decisão complexa; Laços de repetição com for; Laços de repetição com while; Laços de repetição com for e while; Laços de repetição aninhados. Tópicos de programação não permitidos na questão: Funções; Estrutura de dados homogêneos com vetores; Estrutura de dados homogêneos com matrizes.',
            'Funções(funções que o usuário precisa criar)': 'Tópicos de programação permitidos na questão: Variáveis, expressões, entradas e saídas; Decisão simples; Decisão complexa; Laços de repetição com for; Laços de repetição com while; Laços de repetição com for e while; Laços de repetição aninhados; Funções. Tópicos de programação não permitidos na questão: Estrutura de dados homogêneos com vetores; Estrutura de dados homogêneos com matrizes.',
            'Estrutura de dados homogêneos com vetores': 'Tópicos de programação permitidos na questão: Variáveis, expressões, entradas e saídas; Decisão simples; Decisão complexa; Laços de repetição com for; Laços de repetição com while; Laços de repetição com for e while; Laços de repetição aninhados; Funções; Estrutura de dados homogêneos com vetores. Tópicos de programação não permitidos na questão: Estrutura de dados homogêneos com matrizes.',
            'Estrutura de dados homogêneos com matrizes': 'Tópicos de programação permitidos na questão: Variáveis, expressões, entradas e saídas; Decisão simples; Decisão complexa; Laços de repetição com for; Laços de repetição com while; Laços de repetição com for e while; Laços de repetição aninhados; Funções; Estrutura de dados homogêneos com vetores; Estrutura de dados homogêneos com matrizes. Tópicos de programação não permitidos na questão: Não possui.'}

QUESTION_FORMAT = """A saída é composta pelos seguintes componentes:
{Title} Tema e tópico relacionados à questão desenvolvida {/Title}

{Description} Contextualização do tema e informações necessárias para o entendimento e resolução da questão. Descrição do problema computacional a ser resolvido e as entradas e saídas necessárias e seus padrões (tipos, regras e formatações). {/Description}

{Examples} Exemplos de saídas que ilustrem a execução do problema e complemente o entendimento do que tem que ser feito pelo programa. {/Examples}"""

AVALIATION_PROMPT_STR = """
Assuma o papel de um avaliador de questões práticas de programação de computadores. Sua tarefa é avaliar a questão passada se baseando em uma rubrica com critérios definidos.
Você receberá uma questão prática de programação de computadores formatada em LaTeX e com 3 Tags que delimitam seu conteúdo ({Title}, {Description} e {Examples}).
Certifique-se de ler, compreender e seguir estas instruções atentamente.

Rubrica de Critérios de Avaliação:
1. Gramática(GRAM) (1-3). Se a questão segue o conjunto de regras e princípios que regem o funcionamento da língua portuguesa do Brasil, incluindo a formação de palavras (morfologia), a organização das frases (sintaxe) e o uso adequado da língua (norma):
Score 1: A questão possui erros gramaticais significativos, dificultando a compreensão do seu enunciado.
Score 2: A questão contém pequenos erros gramaticais, mas isso não impede a compreensão do seu enunciado.
Score 3: A questão é gramaticalmente correta.
2. Clareza(CLAR) (1–3). Se a questão apresenta um enunciado compreensível e com informações precisas, evitando que ela gere interpretações múltiplas, dúvidas ou confusão quanto ao que deve ser feito:
Score 1: A questão é muito ampla ou confusa, dificultando a compreensão do que deve ser feito ou interpretações múltiplas.
Score 2: A questão não é totalmente clara e detalhada, mas é possível inferir a tarefa a partir do enunciado.
Score 3: A questão é clara e detalhada, sem dúvidas ou múltiplas interpretações.
3. Concisão(CONC) (1-3). Se a questão expressa as informações de forma objetiva e direta, evitando repetições ou excesso de informações que possam alongar o enunciado e prejudicar a compreensão:
Score 1: A questão contém muitas informações repetidas ou irrelevantes, dificultando a compreensão de sua intenção.
Score 2: A questão inclui algumas informações repetidas ou pouco relevantes, mas isso não afeta o entendimento.
Score 3: A questão é objetiva e não contém nenhuma informação desnecessária.
4. Consistência(CONS) (1-3). Se a questão é coerente e logicamente estruturada, garantindo que o enunciado não apresente contradições, lacunas de informação ou falhas de interpretação que prejudiquem sua resolução:
Score 1: A questão contém contradições, falhas de lógica ou ausência de informações essenciais, tornando impossível sua resolução.
Score 2: A questão pode ser parcialmente resolvida, mas exige suposições por falta de informações importantes, o que pode comprometer sua execução.
Score 3: A questão pode ser plenamente resolvida, sem contradições, erros lógicos ou lacunas de informação.
5. Relevância(RELE) (1-3). Se a questão está alinhada ao tema proposto e ao campo da programação de computadores, abordando conteúdos, conceitos ou práticas que contribuem para o aprendizado destes contextos:
Score 1: A questão não tem relação com programação de computadores ou com o tema de questão proposto.
Score 2: A questão tem relação parcial com programação de computadores ou o tema de questão proposto, mas aborda aspectos relacionados.
Score 3: A questão é diretamente ligada com programação de computadores e com o tema de questão proposto, e as informações que ela aborda são cruciais para o tema proposto e a prática da programação de computadores.
6. Adequação pedagógica(ADEP) (1 ou 3). Se a questão está alinhada ao nível de conhecimento esperado, utilizando apenas tópicos compatíveis com a etapa atual da linha de aprendizado e evitando a exigência de tópicos ainda não aprendidos.
A linha de aprendizado dos tópicos em ordem são: (a) Variáveis, expressões, entradas e saídas; (b) Decisão simples; (c) Decisão aninhada; (d) Laços de repetição com for; (e) Laços de repetição com while; (f) Laços de repetição com for e while; (g) Laços de repetição aninhados; (h) Funções; Estrutura de dados homogêneos com vetores; (i) Estrutura de dados homogêneos com matrizes.
Dado um tópico definido para a questão, tal tópico deverá ser cobrado no exercício, tópicos anteriores a ele são permitidos, e os tópicos posteriores não são permitidos.
Score 1: A questão não cumpre, sendo necessários conceitos ainda não trabalhados.
Score 3: A questão cumpre, podendo ser resolvidas apenas com os conceitos da etapa atual de aprendizado.

Passos de Avaliação:
1. Leia cuidadosamente a questão e identifique o tema principal e o tópico chave.
2. Verifique se a questão abrange o tema principal, o tópico chave e se são apresentados de forma clara e lógica.
3. Verifique se a questão respeita a linha de aprendizado, usando apenas o tópico informado e os anteriores.
4. Atribua pontuações para a questão, sendo de 1 a 3 para os 5 primeiros critérios da rubrica e 1 ou 3 para o último critério, com base nas definições de avaliação e no seu raciocínio.
5. Para cada um dos critérios, escreva uma breve explicação de sua análise.
6. Calcule a pontuação final (overall) baseada na média aritmética simples das pontuações atribuídas aos 6 critérios, ou seja, calcule usando a equação: ("GRAM" + "CLAR" + "CONC" + "CONS" + "RELE" + "ADEP")/6.
7. Forneça o resultado final no seguinte formato JSON:
{
 "scores": {"GRAM": int, "CLAR": int, "CONC": int, "CONS": int, "RELE": int, "ADEP": int},
 "comments": {"GRAM_comment": "breve explicação para critério Gramática", "CLAR_comment": "breve explicação para critério Clareza", "CONC_comment": "breve explicação para critério Concisão", ...},
 "overall": float
}

Questão a ser Avaliada:
"""