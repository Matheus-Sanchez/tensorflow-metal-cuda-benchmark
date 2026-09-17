# Relatório consolidado: TensorFlow Metal vs. CUDA

**Atualizado em:** 17/09/2026
**Branch de consolidação:** `results/consolidated`
**Fontes:** `results/mac` e `results/windows`

## Estado das evidências

| Plataforma | Branch de origem | Execução encontrada | Situação |
|---|---|---|---|
| iMac M4 / TensorFlow Metal | `results/mac` | Não | Pendente |
| RTX A2000 / TensorFlow CUDA no WSL2 | `results/windows` | Sim | Consolidada |

`results/mac` ainda contém somente a base do benchmark; não há `summary.json`, `samples.csv` ou `metadata.json` de uma execução Metal. Por isso, este relatório registra os dados Windows individualmente e deixa os campos comparativos como **não calculáveis**. Não é válido inferir ou estimar os valores do Mac a partir do hardware.

## Resultado individual — Windows/WSL2

| Campo | Valor |
|---|---|
| GPU | NVIDIA RTX A2000 12 GB |
| Driver NVIDIA | 596.51 |
| Sistema | Linux 6.18.33.2-microsoft-standard-WSL2 |
| TensorFlow | 2.21.0 |
| Precisão | FP32 |
| XLA | Desativado |
| Repetições | 5 por cenário |

### Matmul FP32

| Parâmetros | Tempo mediano | Throughput mediano | Dispersão entre repetições |
|---|---:|---:|---:|
| 4096×4096; 100 operações por repetição | 10,1791 ms/op | 13.502,0 GFLOP/s | 10,91 ms de desvio-padrão por bloco de 100 operações |

### Treino de CNN com lote residente na GPU

Os dados sintéticos foram criados antes da janela de medição. Cada passo inclui somente forward, gradientes e atualização Adam.

| Batch | Tempo mediano | Passos/s | Imagens/s |
|---:|---:|---:|---:|
| 32 | 1,8653 ms | 536,1 | 17.155,8 |
| 64 | 1,9213 ms | 520,5 | 33.310,1 |
| 128 | 2,0259 ms | 493,6 | 63.180,4 |
| 256 | 2,2422 ms | 446,0 | 114.172,1 |

Os arquivos-fonte deste resultado estão em `results/tf-benchmark-20260917T124917-0300-wsl-rtx-a2000/`.

## Comparação Metal vs. CUDA

| Teste | Mac M4 / Metal | Windows RTX A2000 / CUDA | Razão Mac/Windows |
|---|---:|---:|---:|
| Matmul FP32 4096×4096 | Pendente | 10,1791 ms/op | Não calculável |
| CNN, batch 32 | Pendente | 1,8653 ms/passo | Não calculável |
| CNN, batch 64 | Pendente | 1,9213 ms/passo | Não calculável |
| CNN, batch 128 | Pendente | 2,0259 ms/passo | Não calculável |
| CNN, batch 256 | Pendente | 2,2422 ms/passo | Não calculável |

## Próximo passo para fechar o comparativo

1. Execute o mesmo `tensorflow_benchmark.py run` no iMac com TensorFlow e `tensorflow-metal`, mantendo os parâmetros padrão e XLA desativado.
2. Adicione o diretório gerado à branch `results/mac`.
3. Na branch `results/consolidated`, execute `tensorflow_benchmark.py compare` apontando para os dois diretórios de resultado. O comando valida os parâmetros antes de calcular medianas e razões de desempenho.

Somente após essa execução será apropriado concluir se a diferença observada no treino real persiste quando `tf.data`, I/O, callbacks e validação são removidos da medição.
