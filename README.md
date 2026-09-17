# Benchmark TensorFlow: Metal vs. CUDA

Este repositório investiga uma pergunta específica: a diferença observada no treino real entre o iMac M4 e o computador com RTX A2000 vem da execução TensorFlow na GPU ou de outras partes do treinamento?

Ele compara duas pilhas de software reais, sempre com TensorFlow:

- **Mac:** TensorFlow 2.18.1 com o plug-in `tensorflow-metal` 1.2.0, executando na GPU integrada do Apple M4;
- **Windows/WSL2:** TensorFlow 2.21.0 com CUDA 12.5.1 e cuDNN 9, executando em uma NVIDIA RTX A2000 de 12 GB.

O resultado consolidado está em [reports/relatorio_consolidado.md](reports/relatorio_consolidado.md) e os dados calculados automaticamente estão em [results/tf-comparison-20260917T130256-0300/](results/tf-comparison-20260917T130256-0300/).

## Em uma frase: o que o benchmark responde?

Ele mede quanto tempo a GPU leva para executar dois trechos pequenos e controlados de TensorFlow: uma multiplicação grande de matrizes e um passo completo de treino de uma CNN simples. Assim, ele ajuda a separar o custo do backend GPU do custo do pipeline de dados e de outras etapas do experimento.

Ele **não** mede o tempo total de uma pesquisa nem decide qual computador é melhor em qualquer uso. Um treinamento real também pode incluir leitura de disco, decodificação de imagens, augmentations, `tf.data`, validação, callbacks, salvamento de pesos e operações executadas na CPU.

## Conceitos usados no relatório

| Termo | Significado neste repositório |
|---|---|
| **TensorFlow Metal** | Caminho pelo qual o TensorFlow envia operações à GPU Apple usando a API Metal do macOS. |
| **TensorFlow CUDA** | Caminho pelo qual o TensorFlow envia operações à GPU NVIDIA usando CUDA e bibliotecas como cuDNN. |
| **FP32** | Números de ponto flutuante de 32 bits (`float32`). É a mesma precisão nos dois computadores. |
| **Batch (lote)** | Número de imagens processadas em uma atualização do modelo. Foram testados lotes de 32, 64, 128 e 256 imagens. |
| **Passo de treino** | Uma atualização dos pesos: previsão, cálculo de erro, gradientes e otimização. |
| **Aquecimento** | Execuções não cronometradas antes da medição. Elas evitam que criação de kernels, alocações iniciais e compilação distorçam o resultado. |
| **Mediana** | Valor central das cinco repetições. Ela é menos sensível a uma execução ocasionalmente mais lenta que a média simples. |
| **Throughput** | Quantidade de trabalho por segundo: GFLOP/s no `matmul` e imagens/s na CNN. Maior é melhor. |

## Protocolo comum aos dois computadores

Os seguintes parâmetros foram iguais nos resultados comparados:

| Item | Configuração | Por que importa |
|---|---|---|
| Precisão | FP32 | Evita comparar `float16`/mixed precision em um lado com FP32 no outro. |
| Compilação XLA | Desativada | XLA pode alterar fortemente kernels e fusões de operações; deixá-la desligada reduz uma variável extra. |
| Repetições | 5 por cenário | Permite usar mediana e observar variação entre execuções. |
| Dispositivo | `/GPU:0` confirmado nos metadados | Garante que a intenção era executar na GPU, não na CPU. |
| Dados | Sintéticos e gerados no dispositivo | Remove download, leitura de disco e cópia CPU→GPU do intervalo medido. |
| Sincronização | Materialização do resultado ao fim de cada bloco | Garante que uma GPU assíncrona terminou o trabalho antes de o tempo ser registrado. |

Cada execução salva `metadata.json` (ambiente e GPU), `samples.csv` (as cinco amostras), `summary.json` (medianas e especificação) e `report.md` (leitura rápida). O comando `compare` só aceita dois `summary.json` com a mesma especificação; se um parâmetro for diferente, ele aborta em vez de produzir uma comparação enganosa.

## Teste 1 — multiplicação de matrizes (`tf.linalg.matmul`)

### O que é calculado

O script cria duas matrizes aleatórias `A` e `B`, cada uma com formato `4096 × 4096`, em `float32`, já na GPU. Em seguida calcula:

```text
C = A × B
```

Cada posição de `C` é a soma de 4.096 produtos. O cálculo completo envolve aproximadamente `2 × 4096³`, ou **137,4 bilhões de operações de ponto flutuante**, por multiplicação. É um workload grande, denso e altamente paralelo, muito comum em camadas densas, projeções e partes de modelos de aprendizado profundo.

### Como o tempo é medido

1. As matrizes são criadas antes da medição.
2. O `matmul` é executado 10 vezes para aquecimento.
3. São executadas 100 multiplicações consecutivas dentro de uma repetição cronometrada.
4. O último resultado é materializado no Python, esperando a GPU terminar.
5. O processo completo é repetido 5 vezes; o relatório usa a mediana em milissegundos por multiplicação e em GFLOP/s.

Esse teste não é um treino de rede neural e não envolve dados ou gradientes. Ele responde apenas: **quão rápido este backend TensorFlow executa uma grande operação matemática na GPU?**

### Resultado atual

| Métrica | Mac M4 / Metal | RTX A2000 / CUDA | Leitura correta |
|---|---:|---:|---|
| Tempo mediano | 47,4441 ms/op | 10,1791 ms/op | Menor tempo é melhor; CUDA foi 4,66× mais rápida neste `matmul`. |
| Throughput mediano | 2.896,9 GFLOP/s | 13.502,0 GFLOP/s | Maior é melhor; a mesma razão aparece no trabalho concluído por segundo. |

O fator `4,66×` vem de `47,4441 ÷ 10,1791`. Ele não significa que toda aplicação do Mac é 4,66× mais lenta; significa somente que essa multiplicação, com esse tamanho e essas versões, levou 4,66 vezes mais tempo no Mac.

## Teste 2 — passo de treino de uma CNN pequena

Este teste se aproxima mais de um treino real: há convoluções, função de perda, retropropagação e atualização dos pesos. Mesmo assim, ele continua propositalmente pequeno e controlado para não misturar o resultado com carregamento de dados ou com detalhes do projeto de pesquisa.

### Dados de entrada

Para cada repetição, o script gera na GPU:

- imagens aleatórias `float32` com formato `[batch, 28, 28, 1]`;
- um rótulo inteiro aleatório de 0 a 9 para cada imagem.

`28 × 28 × 1` representa uma imagem em escala de cinza, semelhante em formato ao MNIST. As imagens são sintéticas: não possuem significado visual e não servem para medir acurácia. Elas servem somente para fornecer tensores com forma e tipo definidos à rede. O mesmo lote permanece residente na GPU durante os 100 passos medidos de uma repetição.

### Arquitetura da CNN

`CNN` significa *Convolutional Neural Network* (rede neural convolucional). A arquitetura fixa do benchmark é a seguinte:

```text
Entrada [batch, 28, 28, 1]
  ↓
Conv2D: 16 filtros 3×3, padding="same", ReLU
  ↓
MaxPooling2D: janela 2×2
  ↓
Conv2D: 32 filtros 3×3, padding="same", ReLU
  ↓
GlobalAveragePooling2D
  ↓
Dense: 10 saídas (logits)
```

| Camada | Saída para cada imagem | Parâmetros treináveis | Função |
|---|---|---:|---|
| Entrada | `28 × 28 × 1` | 0 | Recebe uma imagem em escala de cinza. |
| `Conv2D(16, 3×3, same, ReLU)` | `28 × 28 × 16` | 160 | Aplica 16 filtros locais para extrair padrões; `same` preserva altura e largura; ReLU introduz não linearidade. |
| `MaxPooling2D(2×2)` | `14 × 14 × 16` | 0 | Reduz cada região 2×2 ao maior valor, diminuindo custo e resolução espacial. |
| `Conv2D(32, 3×3, same, ReLU)` | `14 × 14 × 32` | 4.640 | Extrai padrões mais abstratos a partir dos 16 mapas anteriores. |
| `GlobalAveragePooling2D` | `32` | 0 | Calcula a média dos 14×14 valores de cada um dos 32 mapas de características. |
| `Dense(10)` | `10` | 330 | Produz uma pontuação, ou *logit*, para cada uma das 10 classes. |

A rede possui **5.130 parâmetros treináveis**. É pequena de propósito: o objetivo não é obter alta acurácia, mas repetir o mesmo passo de treino em Metal e CUDA com poucas variáveis externas.

### O que acontece em um passo de treino

Para cada batch, o `train_step` executa:

1. **Forward pass:** as imagens passam pelas camadas e produzem 10 logits por imagem.
2. **Loss:** `SparseCategoricalCrossentropy(from_logits=True)` compara os logits com o rótulo de classe de 0 a 9. “Sparse” significa que o rótulo é um único inteiro, e não um vetor one-hot.
3. **Gradientes:** `tf.GradientTape` calcula como cada um dos 5.130 parâmetros deve mudar para reduzir a loss.
4. **Atualização:** `Adam(learning_rate=0.001)` aplica os gradientes aos pesos. Adam é um otimizador adaptativo comum em redes neurais.

Modelo, pesos, estado do otimizador, imagens e rótulos são criados antes da janela cronometrada de cada repetição. Depois há 20 passos de aquecimento. Somente então 100 passos consecutivos são medidos. O tempo, portanto, inclui cálculo direto, retropropagação e Adam, mas exclui construção do modelo, criação dos dados e aquecimento.

### Por que vários tamanhos de batch?

Um batch maior faz mais trabalho em cada passo, mas também costuma usar a GPU com maior eficiência. Por isso não é suficiente observar apenas `ms/passo`: também é necessário olhar `imagens/s`.

| Batch | O que significa | Uso no relatório |
|---:|---|---|
| 32 | 32 imagens por atualização | Mostra desempenho com trabalho menor por passo. |
| 64 | 64 imagens por atualização | Ponto intermediário. |
| 128 | 128 imagens por atualização | Aumenta a ocupação da GPU. |
| 256 | 256 imagens por atualização | Maior lote testado; maximiza imagens processadas por passo, desde que a memória comporte o lote. |

### Resultados atuais da CNN

| Batch | Mac M4 / Metal | RTX A2000 / CUDA | Razão Mac/Windows |
|---:|---:|---:|---:|
| 32 | 3,7815 ms/passo; 8.462,1 imagens/s | 1,8653 ms/passo; 17.155,8 imagens/s | 2,03× |
| 64 | 3,7060 ms/passo; 17.269,4 imagens/s | 1,9213 ms/passo; 33.310,1 imagens/s | 1,93× |
| 128 | 3,9019 ms/passo; 32.804,2 imagens/s | 2,0259 ms/passo; 63.180,4 imagens/s | 1,93× |
| 256 | 4,4148 ms/passo; 57.986,5 imagens/s | 2,2422 ms/passo; 114.172,1 imagens/s | 1,97× |

Neste teste, a RTX A2000 processou cerca do dobro de imagens por segundo em todos os batches. A diferença é menor do que a vista no `matmul` porque um passo de CNN mistura operações de formatos e custos diferentes: convoluções, pooling, redução, perda, gradientes e atualização Adam. Nenhum único número de `matmul` descreve integralmente o comportamento de um treino.

## O que entra e o que não entra no cronômetro

| Etapa | Matmul | CNN | Motivo |
|---|---|---|---|
| Criação de matrizes/imagens/rótulos | Não | Não | Alocação e geração de dados não devem ser confundidas com execução do workload. |
| Construção do modelo e do otimizador | Não se aplica | Não | São custos de inicialização, não de cada passo de treino. |
| Aquecimento | Não | Não | Estabiliza a execução antes da coleta. |
| Operação matemática / forward | Sim | Sim | É o objeto principal da medição. |
| Gradientes e Adam | Não se aplica | Sim | Fazem parte de um passo completo de treino. |
| Sincronização final com a GPU | Sim | Sim | Necessária para que o tempo represente trabalho realmente concluído. |
| Download, disco, `tf.data`, cópia CPU→GPU | Não | Não | Foram removidos para isolar o backend TensorFlow/GPU. |
| Validação, callbacks e checkpoints | Não | Não | Fazem parte de um experimento real, mas não deste microbenchmark. |

## Como interpretar os resultados sem extrapolar

As conclusões suportadas pelos dados são:

- ambos os ambientes expuseram uma GPU ao TensorFlow e os tensores medidos ficaram em `/GPU:0`;
- para o `matmul` grande, CUDA/RTX foi 4,66× mais rápida;
- para o passo completo da CNN pequena, CUDA/RTX foi entre 1,93× e 2,03× mais rápida;
- portanto, o núcleo da CNN isolado não reproduziu uma diferença de 5–6×.

O que os dados **não** permitem concluir:

- que o M4 é “4,66× mais lento” em qualquer modelo ou aplicativo;
- que 16 GB de memória unificada do Mac equivalem diretamente aos 12 GB de VRAM dedicada da RTX;
- que o restante da diferença no treino real é necessariamente culpa da GPU;
- que o resultado isola somente hardware: TensorFlow 2.18.1/Metal e TensorFlow 2.21.0/CUDA são versões e backends diferentes.

Se o treino da pesquisa ainda é 5–6× mais lento no Mac, a diferença adicional está provavelmente em uma parte que este teste eliminou: pipeline `tf.data`, leitura e preparação de imagens, augmentations, validação, callbacks, checkpoints, formato do modelo ou operações sem suporte Metal que são executadas na CPU. A próxima investigação deve cronometrar essas etapas separadamente no código real.

## Estrutura de branches e evidências

| Branch | Conteúdo |
|---|---|
| `main` | Script do benchmark, instruções e relatório técnico do iMac. |
| `results/mac` | Dados brutos da execução Metal: metadados, amostras e resumo. |
| `results/windows` | Dados brutos da execução CUDA/RTX A2000. |
| `results/consolidated` | Merge dos resultados, comparação validada e relatório final. |

As execuções individuais estão em:

- Mac: `results/tf-benchmark-20260916T221446-0300-mac-m4/`;
- Windows/WSL2: `results/tf-benchmark-20260917T124917-0300-wsl-rtx-a2000/`;
- comparação: `results/tf-comparison-20260917T130256-0300/`.

## Reproduzir o benchmark

As instruções de instalação e todos os parâmetros de linha de comando estão em [outputs/README.md](outputs/README.md). Em resumo:

```bash
# iMac M4, com tensorflow-metal instalado
python3 outputs/tensorflow_benchmark.py run --label mac-m4 --expect metal --output-dir results

# Windows/WSL2, com TensorFlow/CUDA e RTX A2000 visível
python3 outputs/tensorflow_benchmark.py run --label wsl-rtx-a2000 --expect cuda --require-gpu-name "RTX A2000" --output-dir results

# Com os dois diretórios de resultados no mesmo computador
python3 outputs/tensorflow_benchmark.py compare \
  --mac-run results/tf-benchmark-AAAAmmddTHHMMSS-0000-mac-m4 \
  --windows-run results/tf-benchmark-AAAAmmddTHHMMSS-0000-wsl-rtx-a2000 \
  --output-dir results
```

Use alimentação AC, feche processos pesados e preserve as versões do ambiente nos metadados. O relatório técnico do Mac, sem identificadores pessoais, está em [outputs/relatorio_mac.md](outputs/relatorio_mac.md).
