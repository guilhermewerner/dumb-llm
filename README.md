# dumb-llm

Um projeto pequeno e didático para entender como um modelo de linguagem é treinado. O modelo é um GPT de caracteres escrito em PyTorch: ele lê texto, transforma caracteres em números, aprende a prever o próximo caractere e então gera texto um caractere por vez.

> Este projeto é para aprendizado. O corpus de demonstração é minúsculo e o modelo não é útil como assistente nem como fonte confiável de informação.

## Comece aqui (Windows / PowerShell)

Requer Python 3.10 ou mais recente. No terminal, na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se o PowerShell impedir a ativação do ambiente, use diretamente o Python dele:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Treine com o texto de exemplo:

```powershell
python train.py
```

O treinamento escolhe CUDA quando disponível e usa CPU caso contrário. O corpus de exemplo é pequeno, então rode menos iterações para uma demonstração rápida ou mais iterações para observar a perda cair:

```powershell
python train.py --max-iters 200
python train.py --max-iters 2000
```

Gere texto a partir do checkpoint salvo em `checkpoints/`:

```powershell
python generate.py --prompt "A biblioteca"
python generate.py --prompt "No começo" --max-new-tokens 300 --temperature 0.8
```

Saída do treinamento e checkpoints são locais e ignorados pelo Git.

## Use seu próprio texto

Coloque um arquivo de texto simples em UTF-8 e passe o caminho:

```powershell
python train.py --data caminho\para\meu_corpus.txt
```

O vocabulário é criado a partir dos caracteres encontrados no arquivo. Por isso, um prompt de geração só pode conter caracteres que apareceram no corpus usado naquele treinamento. Comece com um corpus que tenha alguns milhares de caracteres; corpus muito curto pode não ter exemplos suficientes para preencher os lotes.

## O que acontece no código

1. `train.py` lê o corpus e cria um vocabulário de caracteres. Cada caractere recebe um inteiro (`encode`) e os inteiros podem voltar a texto (`decode`).
2. O corpus vira uma sequência de IDs e é dividido em treino e validação. Cada lote contém trechos de `block_size` caracteres; o alvo é o mesmo trecho deslocado uma posição. Assim, em cada posição, o modelo aprende a prever o próximo caractere.
3. `model.py` soma embeddings do caractere e da posição, passa a sequência por blocos Transformer com atenção causal e calcula uma distribuição de probabilidade para o próximo caractere.
4. A entropia cruzada (`cross_entropy`) mede a diferença entre a previsão e o próximo caractere correto. AdamW ajusta os pesos para reduzir essa perda.
5. `generate.py` carrega os pesos aprendidos e amostra um caractere da distribuição prevista repetidamente.

## Parâmetros para experimentar

Os valores padrão ficam na dataclass `Config` em `train.py`.

| Parâmetro       | O que controla                                          | Experimento sugerido                                                          |
| --------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `max_iters`     | Número de atualizações dos pesos                        | Mais atualizações dão mais prática ao modelo, mas demoram mais.               |
| `learning_rate` | Tamanho de cada ajuste dos pesos                        | Muito alto pode tornar o treino instável; muito baixo pode aprender devagar.  |
| `batch_size`    | Quantos trechos são usados em cada atualização          | Lotes maiores usam mais memória e deixam a estimativa da perda menos ruidosa. |
| `block_size`    | Quantos caracteres anteriores o modelo pode ver         | Aumente para dar mais contexto; isso custa mais memória e computação.         |
| `n_embd`        | Largura das representações internas                     | Mais dimensões aumentam a capacidade e o custo.                               |
| `n_head`        | Quantas atenções aprendem relações em paralelo          | `n_embd` precisa ser divisível por `n_head`.                                  |
| `n_layer`       | Quantos blocos Transformer empilhar                     | Mais blocos aumentam profundidade, capacidade e custo.                        |
| `dropout`       | Fração de ativações aleatoriamente desligadas no treino | Ajuda a limitar memorização; `0.0` desliga esse efeito.                       |

Para mudar os padrões, edite `Config` e execute `python train.py` novamente. O programa salva a configuração no checkpoint. Isso permite que `generate.py` recrie a arquitetura correspondente.

## Arquivos

- `model.py`: atenção causal, blocos Transformer e modelo de linguagem.
- `train.py`: dados, lotes, avaliação, otimização e salvamento do checkpoint.
- `generate.py`: carregamento do checkpoint e geração com temperatura.
- `data/sample.txt`: corpus fictício pequeno em português.
- `requirements.txt`: dependência PyTorch.

## Limites deste exemplo

O tokenizer opera por caractere, não por palavras ou subpalavras. O modelo treina do zero com um corpus pequeno e usa uma arquitetura reduzida. Isso deixa o fluxo de treinamento visível, mas a geração tende a ser lenta e com erros. Modelos de produção usam grandes datasets, tokenizadores, muito mais parâmetros, hardware acelerado e avaliações cuidadosas.
