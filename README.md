# Benchmark TensorFlow: Metal vs. CUDA

Repositório para comparar, de forma reproduzível, o desempenho de duas pilhas TensorFlow no mesmo conjunto de microbenchmarks:

- **macOS / Apple Silicon:** TensorFlow com o plug-in `tensorflow-metal`;
- **Windows via WSL2 / NVIDIA:** TensorFlow com CUDA, direcionado à RTX A2000.

O objetivo é identificar se a diferença de tempo observada no treino real vem do backend TensorFlow/GPU ou de outras partes do pipeline, como `tf.data`, validação, callbacks ou I/O. O repositório não faz uma afirmação de desempenho geral entre os hardwares.

## O que é testado

1. **Matmul FP32:** `tf.linalg.matmul` com matrizes `4096×4096`, após aquecimento e com sincronização por materialização do resultado.
2. **Passo de treino residente:** uma CNN `tf.keras` pequena para entradas `1×28×28`, dados sintéticos criados previamente no dispositivo e medição restrita a forward, gradientes e Adam.

Os dois testes usam FP32, XLA desativado por padrão, 5 repetições e salvam os dados de cada repetição. Não baixam datasets nem medem `tf.data`, disco ou cópia de dados CPU→GPU durante a janela cronometrada.

## Estrutura de branches

| Branch | Finalidade |
|---|---|
| `main` | Script, instruções e relatório técnico da configuração do iMac. |
| `results/mac` | Execuções produzidas no iMac M4 com TensorFlow Metal. |
| `results/windows` | Execuções produzidas no Windows/WSL2 com TensorFlow CUDA/RTX A2000. |

Cada branch de resultados deve receber a pasta completa criada pelo comando `run` (`metadata.json`, `samples.csv`, `summary.json` e `report.md`). Depois, use `compare` para gerar o comparativo com parâmetros idênticos.

## Uso rápido

Veja as instruções detalhadas em [`outputs/README.md`](outputs/README.md). Em resumo:

```bash
# iMac M4
python3 outputs/tensorflow_benchmark.py run --label mac-m4 --expect metal --output-dir results

# Windows/WSL2 com RTX A2000
python3 outputs/tensorflow_benchmark.py run --label wsl-rtx-a2000 --expect cuda --require-gpu-name "RTX A2000" --output-dir results
```

O relatório da configuração fornecida para o Mac está em [`outputs/relatorio_mac.md`](outputs/relatorio_mac.md), sem identificadores pessoais ou números de série.
