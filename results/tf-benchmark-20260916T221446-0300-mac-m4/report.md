# Resultado do benchmark TensorFlow

- Executado em: `2026-09-17T01:15:32.116433+00:00`
- Sistema: `Darwin 24.5.0` (arm64)
- TensorFlow: `2.18.1`; tensorflow-metal: `1.2.0`
- GPU TensorFlow: `/device:GPU:0`
- XLA: `desativado`

## Resultados

| Teste | Parâmetros | Tempo mediano | Throughput mediano |
|---|---|---:|---:|
| Matmul FP32 | 4096×4096, 100 operações | 47.4441 ms/op | 2896.9 GFLOP/s |
| CNN, lote residente | batch 32, 100 passos | 3.7815 ms/passo | 8462.1 imagens/s |
| CNN, lote residente | batch 64, 100 passos | 3.7060 ms/passo | 17269.4 imagens/s |
| CNN, lote residente | batch 128, 100 passos | 3.9019 ms/passo | 32804.2 imagens/s |
| CNN, lote residente | batch 256, 100 passos | 4.4148 ms/passo | 57986.5 imagens/s |

## Interpretação

Este teste exclui download de dataset, DataLoader/tf.data e cópias de dados do intervalo cronometrado. Portanto, ele isola a execução TensorFlow do backend GPU; não representa o tempo total de treinamento de um experimento real.

Consulte `metadata.json` para as versões completas do runtime e `samples.csv` para todas as repetições individuais.
