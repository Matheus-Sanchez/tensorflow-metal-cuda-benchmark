# Resultado do benchmark TensorFlow

- Executado em: `2026-09-17T15:49:37.817756+00:00`
- Sistema: `Linux 6.18.33.2-microsoft-standard-WSL2` (x86_64)
- TensorFlow: `2.21.0`; tensorflow-metal: `não instalado`
- GPU TensorFlow: `/device:GPU:0`
- XLA: `desativado`

## Resultados

| Teste | Parâmetros | Tempo mediano | Throughput mediano |
|---|---|---:|---:|
| Matmul FP32 | 4096×4096, 100 operações | 10.1791 ms/op | 13502.0 GFLOP/s |
| CNN, lote residente | batch 32, 100 passos | 1.8653 ms/passo | 17155.8 imagens/s |
| CNN, lote residente | batch 64, 100 passos | 1.9213 ms/passo | 33310.1 imagens/s |
| CNN, lote residente | batch 128, 100 passos | 2.0259 ms/passo | 63180.4 imagens/s |
| CNN, lote residente | batch 256, 100 passos | 2.2422 ms/passo | 114172.1 imagens/s |

## Interpretação

Este teste exclui download de dataset, DataLoader/tf.data e cópias de dados do intervalo cronometrado. Portanto, ele isola a execução TensorFlow do backend GPU; não representa o tempo total de treinamento de um experimento real.

Consulte `metadata.json` para as versões completas do runtime e `samples.csv` para todas as repetições individuais.
